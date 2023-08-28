//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <unordered_map>

#include "GenericML/GraphCompiler/KGENStubBuilder/Builder.h"
#include "GenericML/GraphCompiler/KGENStubBuilder/StubCompiler.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/Support/MOGGTypeTraits.h"
#include "GenericML/Support/Tensor.h"

#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/StringExtras.h"
#include <sstream>
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_JITCOMPILEKERNELS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Directly embed the mojo file rather than loading dynamically.
#include MOGG_MOJO_BINDING_H

// We lower static dimensions as the dimension directly, obviously, but symbolic
// dimensions we track by their index in the loaded dimensions i.e <D1314,
// D2424> will become <P0, P1>. This is so that kernels which use the parameters
// in the same way, i.e that the parameters maintain the same internal
// relationship, will match and we avoid compiling a new novel kernel.
void addShapeDimsToName(MOSH::ShapeAttr dims, std::string &name,
                        DenseMap<KGEN::ParamDeclRefAttr, size_t> &params) {
  for (auto dim : dims.getValues()) {
    if (auto asInt = dyn_cast<IntegerAttr>(dim))
      name += std::to_string(asInt.getInt());
    else if (auto asSymbolic = dyn_cast<KGEN::ParamDeclRefAttr>(dim))
      name += "P" + std::to_string(params[asSymbolic]);
    name += "x";
  }
}

ErrorOr<std::string>
mangleBlock(mlir::Block &block,
            DenseMap<Value, std::string> &valueToPositionInBlock,
            DenseMap<M::ElementsAttr, size_t> &seenConsts,
            DenseMap<KGEN::ParamDeclRefAttr, size_t> &params) {
  std::string name = "";
  size_t blockIndex = 0;
  size_t numConstants = 0;

  // Each operation in the block of ops contributes to the name.
  for (mlir::Operation &innerOp : block) {
    // Handle the output as a special case, report which operations returned a
    // value from the block.
    if (isa<MOGG::OutputOp>(innerOp)) {
      name += "outs_";
      for (Value operand : innerOp.getOperands())
        name += valueToPositionInBlock[operand] + "_";
    } else if (auto lambda = dyn_cast<MOGG::LambdaOp>(innerOp)) {
      mlir::Block &lambdaBlock = lambda.getBody().front();

      // Mangle the lambda block arguments.
      for (size_t argIndex = 0, e = lambdaBlock.getNumArguments(); argIndex < e;
           ++argIndex)
        valueToPositionInBlock[lambdaBlock.getArgument(argIndex)] =
            "L" + std::to_string(argIndex);

      auto mangledLambdaOr =
          mangleBlock(lambdaBlock, valueToPositionInBlock, seenConsts, params);
      if (mangledLambdaOr.isError())
        return mangledLambdaOr.takeError();
      name += *mangledLambdaOr;
    } else if (auto out = dyn_cast<MOGG::OutputPlaceholdersOp>(innerOp)) {
      for (Value output : out.getResults())
        valueToPositionInBlock[output] = "O" + std::to_string(blockIndex);
    } else if (auto out = dyn_cast<MOGG::ParamShapeToValueOp>(innerOp)) {
      MOSH::ShapeAttr shape =
          dyn_cast<MOSH::ShapeAttr>(out.getResult().getType().getShapeAttr());

      name += "PV";
      addShapeDimsToName(shape, name, params);
      valueToPositionInBlock[out.getResult()] = std::to_string(blockIndex);
    } else if (auto invokeOp = dyn_cast<MOGG::LambdaInvokeOp>(innerOp)) {
      name += "invoke_" + invokeOp.getLambda().getLeafReference().str() + "_";

      // Add operands to the mangling.
      for (Value input : innerOp.getOperands())
        name += valueToPositionInBlock[input] + "_";

      // Track the outputs for future mangling.
      for (Value output : innerOp.getResults())
        valueToPositionInBlock[output] = std::to_string(blockIndex);

    } else if (auto call = dyn_cast<MOGG::CallOp>(innerOp)) {
      name += "call_" + call.getMojoKernelTarget().str();

      // If we have parameters we have to mangle those too.
      if (call.getParams()) {
        for (SymbolRefAttr sym :
             call.getParamsAttr().getAsRange<SymbolRefAttr>())
          name += sym.getLeafReference().str() + "_";
      }

      // Mangle the kgen parameters into the calls as well.
      if (call.getKgenParams()) {
        mlir::NamedAttrList attrs{*call.getKgenParams()};
        for (NamedAttribute named : attrs) {
          name += named.getName().str() + "_";
          if (auto asBool = dyn_cast<BoolAttr>(named.getValue())) {
            name += std::to_string(asBool.getValue()) + "_";

          } else if (auto asInt = dyn_cast<IntegerAttr>(named.getValue())) {
            name += std::to_string(asInt.getInt()) + "_";

          } else if (auto asType = dyn_cast<TypeAttr>(named.getValue())) {
            std::string typeStr;
            llvm::raw_string_ostream(typeStr) << asType;
            name += typeStr + "_";

          } else {
            assert(false && "Unsupported parameter attribute type");
          }
        }
      }

      for (Value input : innerOp.getOperands())
        name += valueToPositionInBlock[input] + "_";

      // Track the outputs name for future mangling.
      for (Value output : innerOp.getResults())
        valueToPositionInBlock[output] = std::to_string(blockIndex);
    } else if (auto cst = dyn_cast<MO::ConstantOp>(innerOp)) {
      // Add the local constant ID. (i.e the constant index within this block).
      std::string constantID = "CST" + std::to_string(numConstants);
      numConstants++;

      valueToPositionInBlock[cst.getResult()] = constantID;

      // Add to the mangling soup.
      name += constantID + "_";

      // Give the constant a unique name based on the hash of the name.
      auto itr = seenConsts.find(cst.getValueAttr());

      if (itr != seenConsts.end()) {
        name += std::to_string(itr->second);
      } else {
        name += std::to_string(seenConsts.size());
        seenConsts[cst.getValueAttr()] = seenConsts.size();
      }

      // Add the type, just incase.
      name += "_";
      Type res = cst.getResult().getType();
      auto tensor = cast<MO::TensorType>(res);
      addShapeDimsToName(tensor.getTypedShapeAttr(), name, params);
      auto traitsOr = MOGGTypeTraits::lookupType(tensor.getElementType());
      if (traitsOr.isError())
        return traitsOr.takeError();
      name += traitsOr->moggType.getAsString() + "_";
    }

    // Add double seperator between operations.
    name += "_";
    blockIndex++;
  }

  return {name};
}

// We create a mangled name which exactly matches the operation. The number of
// inputs, the contents of the block, and which are returned. It needs to be
// this verbose to ensure we don't recompile for identical blocks but also allow
// for blocks of the same operations with different returns to compile legally.
ErrorOr<std::string>
createMangledName(MOGG::KernelOp op,
                  DenseMap<KGEN::ParamDeclRefAttr, size_t> &params,
                  DenseMap<M::ElementsAttr, size_t> &seenConsts,
                  size_t &kernelsDefiningParams) {
  std::string name = "__gen__";

  // Each kernel which defines params can be assumed to be unique. No params can
  // be defined twice.
  auto outParams = op.getOutputParamDecls();
  if (!outParams.empty())
    name += "PDEF_" + std::to_string(kernelsDefiningParams++);

  name += "ins_";

  // Track where each value is in the block order. This is to mangle the returns
  // to ensure legality.
  DenseMap<Value, std::string> valueToPositionInBlock;
  size_t argIndex = 0;

  // Get the MLIR block.
  mlir::Block &block = op.getBody().front();
  for (Value v : block.getArguments()) {
    valueToPositionInBlock[v] = "i" + std::to_string(argIndex);
    argIndex++;

    // Add input id to the name.
    name += valueToPositionInBlock[v] + "_";

    // Mangle the type into the name.
    auto tensor = cast<MO::TensorType>(v.getType());
    addShapeDimsToName(tensor.getTypedShapeAttr(), name, params);
    auto traitsOr = MOGGTypeTraits::lookupType(tensor.getElementType());
    if (traitsOr.isError())
      return traitsOr.takeError();
    name += traitsOr->moggType.getAsString() + "_";
  }

  // Add the output type as the kernel could reference this independently.
  name += "_out_ty_";
  for (Type res : op->getResultTypes()) {
    auto tensor = cast<MO::TensorType>(res);
    addShapeDimsToName(tensor.getTypedShapeAttr(), name, params);
    auto traitsOr = MOGGTypeTraits::lookupType(tensor.getElementType());
    if (traitsOr.isError())
      return traitsOr.takeError();
    name += traitsOr->moggType.getAsString() + "_";
  }
  name += "_";

  auto mangledBlockOr = mangleBlock(op.getBody().front(),
                                    valueToPositionInBlock, seenConsts, params);
  if (mangledBlockOr.isError())
    return mangledBlockOr.takeError();
  name += *mangledBlockOr;
  return {name};
}

SmallVector<KGEN::ParamDeclRefAttr> getUsedParameters(MOGG::KernelOp kernel) {
  DenseSet<KGEN::ParamDeclRefAttr> seenParams;
  SmallVector<KGEN::ParamDeclRefAttr> usedParameters;

  auto params = kernel.getOutputParamDecls();
  auto isInOutputParams = [&](StringRef name) {
    for (KGEN::ParamDeclAttr attr : params) {
      if (attr.getName().str() == name)
        return true;
    }
    return false;
  };

  // Track all unique parameters from a given tensor that we haven't seen
  // already.
  auto addParamsFromTensor = [&](Type t) {
    MOSH::ShapeAttr shape;

    // The type can either be a shape or a tensor.
    MO::TensorType asTensor = dyn_cast<MO::TensorType>(t);
    if (asTensor)
      shape = asTensor.getTypedShapeAttr();
    MOGG::ShapeType asShape = dyn_cast<MOGG::ShapeType>(t);
    if (asShape)
      shape = cast<MOSH::ShapeAttr>(asShape.getShapeAttr());

    if (!shape)
      return;

    for (auto dim : shape.getValues()) {
      if (auto param = dyn_cast<KGEN::ParamDeclRefAttr>(dim)) {
        if (seenParams.contains(param))
          continue;

        if (isInOutputParams(param.getName()))
          continue;

        seenParams.insert(param);
        usedParameters.push_back(param);
      }
    }
  };

  // Get all parameters used by an operation.
  auto addParamsFromOp = [&](Operation *op) {
    for (Type t : op->getResultTypes())
      addParamsFromTensor(t);

    for (Type t : op->getOperandTypes())
      addParamsFromTensor(t);
  };

  // Start with the inputs / outputs on the kernel. Generally this would be all
  // parameters.
  addParamsFromOp(kernel);

  // However rarely we could create a kernel which internally uses a parameter
  // which isn't known from the inputs. I.E

  // clang-format off
  // %0 = broadcast(%input) : tensor<[N]> -> tensor<N, J>
  // %1 = broadcast(%0) : tensor<[N, J]> -> tensor<N, B>
  // %3 = broadcast(%1) : tensor<[N, B, K]> -> tensor<N, B, K, M>
  // clang-format on

  // We would see `N` on the input and `B` / `K` / `M` on the outputs but the
  // intermediate `J` is not on either.

  // To capture the above legal composition we walk the kernel ops as well.
  kernel.walk(addParamsFromOp);

  return usedParameters;
}

// Convert all possible operations directly to the avaliable operation.
struct JitCompileKernels
    : public impl::JitCompileKernelsBase<JitCompileKernels> {
  using JitCompileKernelsBase::JitCompileKernelsBase;

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    // Load the mojo kernel library info.
    FailureOr<MOGG::MojoRegisterOp> registryOr =
        getOrLoadMojoLibrary(m, extraLibPaths.getValue());
    if (failed(registryOr)) {
      m.emitError() << "[JitCompileKernels] Couldn't load mojo kernel registry";
      return signalPassFailure();
    }
    MOGG::MojoRegisterOp registry = *registryOr;

    // Run the jit required passes.
    mlir::PassManager pm(m->getContext());
    pm.addPass(createLowerSimdLoads());
    pm.addNestedPass<MO::GraphOp>(createInlineLambdas());
    if (failed(pm.run(m)))
      return signalPassFailure();

    for (auto graph : m.getOps<MO::GraphOp>()) {
      auto errOr = this->runPass(graph, registry);
      if (errOr.isError()) {
        graph.emitError() << "[JitCompileKernels] " << errOr.getError();
        return signalPassFailure();
      }
    }
  }

  ErrorOr<mlir::OwningOpRef<mlir::ModuleOp>> loadMojoLibrary() {
    // Load the module. The stub will be appended to this.
    auto errorOr = loadMojoBindings(getOperation()->getContext(),
                                    extraLibPaths.getValue());
    if (errorOr.isError())
      return errorOr.takeError();

    mlir::OwningOpRef<mlir::ModuleOp> baseModuleRef = errorOr.takeValue();

    // Mark all functions as private. The builder will create new exported
    // functions.
    for (auto func : baseModuleRef->getOps<KGEN::FuncOp>())
      func.setNotExported();

    for (auto func : baseModuleRef->getOps<KGEN::GeneratorOp>())
      func.setNotExported();

    return baseModuleRef;
  }

  ErrorOrSuccess runPass(MO::GraphOp graph, MOGG::MojoRegisterOp registry) {
    // Load the mojo library. We use this KGEN module as the starting point to
    // build up the stub functions. It contains all the kernels and functions we
    // will need to call.
    auto kgenModuleOr = loadMojoLibrary();
    if (kgenModuleOr.isError())
      return kgenModuleOr.takeError();
    auto kgenModule = std::move(*kgenModuleOr);

    // Helper to track all the types to prevent constant reallocation.
    Stub::TypeManager manager{graph.getContext()};

    // Map operations with the same output sizes onto the same kernel. Longer
    // term we may want fancier optimizations like unifying aliased inputs but
    // this is legal for now.
    std::unordered_map<std::string, MOGG::ExecuteCompiledOp> jittedKernels;

    // Track all the attributes we have added.
    SmallVector<mlir::Attribute> symbolNamesAdded;

    // Start building up the KGEN kernel.
    Stub::KGENBuilder builder{manager, *kgenModule, registry,
                              minimumCpuAlignment.getValue()};

    // Give each unique constant an identifier globally across all kernels.
    DenseMap<M::ElementsAttr, size_t> seenConsts;

    ErrorOrSuccess error;

    size_t kernelsDefiningParams = 0;

    // Build stubs for all JIT operations and replace them with symbolic calls
    // to the kernels in the KGEN stub.
    graph.walk([&](MOGG::KernelOp op) {
      // We collect all the parameters used by this operation and track them on
      // the kernel.
      SmallVector<KGEN::ParamDeclRefAttr> usedParameters =
          getUsedParameters(op);

      // Helper so we can make arrays from it.
      SmallVector<Attribute> usedParamsAsAttr;
      usedParamsAsAttr.reserve(usedParameters.size());

      // Parameters will be passed in as an array so we need to relate to them
      // by their index in that array (which is the same order as the above
      // vector).
      DenseMap<KGEN::ParamDeclRefAttr, size_t> paramToIndex;

      size_t paramIndex = 0;
      for (auto param : usedParameters) {
        paramToIndex[param] = paramIndex++;
        usedParamsAsAttr.push_back(param);
      }

      // Create a unique name for the kernel.
      auto stubKernelName = createMangledName(op, paramToIndex, seenConsts,
                                              kernelsDefiningParams);
      if (stubKernelName.isError()) {
        error = Error(stubKernelName.takeError());
        return WalkResult::interrupt();
      }

      // Don't duplicate KGEN kernels for the same call.
      auto itr = jittedKernels.find(stubKernelName.get());
      if (itr != jittedKernels.end()) {
        MOGG::ExecuteCompiledOp cachedExeOp = itr->second;

        // We cannot just clone the cached operation because its result type may
        // be different from the currently processed kernel's result type, if
        // they are parameterized.
        mlir::IRRewriter rewriter{OpBuilder(op)};
        auto newOp = rewriter.replaceOpWithNewOp<MOGG::ExecuteCompiledOp>(
            op, op->getResultTypes(), op->getOperands(),
            cachedExeOp->getAttrDictionary().getValue());

        // Add the new used parameters which may be different than before.
        newOp.setUsedParamsAttr(rewriter.getArrayAttr(usedParamsAsAttr));

        return WalkResult::skip();
      }

      // Lower the basic stub function.
      error = builder.startLowering(op, stubKernelName.get(), usedParameters);
      if (error.isError())
        return WalkResult::interrupt();

      // Lower the operation using a vectorized map over an elementwise lambda.
      error = builder.lowerToElementwiseLambda(op, usedParameters.size());
      if (error.isError())
        return WalkResult::interrupt();

      // Replace the original kernel with a call to the compiled jit code.
      MOGG::ExecuteCompiledOp jitExecute = builder.createExecuteOperation(
          stubKernelName.get(), usedParamsAsAttr, op.getOutputParamDeclsAttr());
      op->replaceAllUsesWith(jitExecute.getResults());
      jittedKernels.insert({stubKernelName.get(), jitExecute});

      // Tracking attributes makes it easier to create an array.
      symbolNamesAdded.push_back(jitExecute.getKernelNameAttr());

      if (createMOGGReproducers.getValue()) {
        ModuleOp modu = cast<ModuleOp>(graph.getParentOp());
        static size_t i = 0;
        std::string name = "MOGGReducedGraph_" + std::to_string(i++);

        llvm::errs() << "Graph for generated kernel: \n";
        llvm::errs() << stubKernelName.get() << "\n";
        llvm::errs() << "MO Reproducer: \n";

        // Create a new graph.
        OpBuilder b{modu};
        b.setInsertionPointToEnd(&modu.getBodyRegion().front());
        MO::GraphOp toDump = b.create<MO::GraphOp>(
            graph->getLoc(), name, op->getOperandTypes(), op->getResultTypes());

        // Attach input decls.
        SmallVector<KGEN::ParamDeclAttr> decls;
        for (auto param : usedParameters)
          decls.push_back(
              KGEN::ParamDeclAttr::get(param.getName(), param.getType()));
        toDump.setInputParams(
            KGEN::ParamDeclArrayAttr::get(graph->getContext(), decls));
        b.setInsertionPointToStart(toDump.getBodyBlock());

        // Clone the kernel into the new graph.
        mlir::IRMapping mapping;
        for (auto [oldV, newV] : llvm::zip(
                 op->getOperands(), toDump.getBodyBlock()->getArguments())) {
          mapping.map(oldV, newV);
        }
        auto cloned = b.clone(*op, mapping);
        b.create<MO::OutputOp>(graph->getLoc(), cloned->getResults());

        // Print it out.
        toDump.dump();

        // Remove it so we don't keep traversing.
        toDump.erase();
      }

      op->erase();
      if (dumpStub.getValue())
        builder.dumpStub();

      return WalkResult::advance();
    });

    if (error)
      return error;

    if (!symbolNamesAdded.empty()) {
      // Convert our list into a combined attribute.
      auto allSyms =
          mlir::ArrayAttr::get(graph->getContext(), symbolNamesAdded);

      // Add the compiled binary to the file.
      Stub::StubCompiler compiler{*kgenModule, graph, useSearch.getValue(),
                                  saveTempPrefix.getValue()};
      if (auto err = compiler.compile(allSyms))
        return err.takeError();
    }
    return success();
  }
};

} // namespace

} // namespace M::MOGG
