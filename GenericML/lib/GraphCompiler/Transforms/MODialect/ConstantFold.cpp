//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstantFold.h"
#include "GenericML/Driver/CXXDriverAPI.h"
#include "GenericML/Driver/CompileDriver.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "GenericML/Support/Assert.h"
#include "GenericML/Support/Model.h"
#include "GenericML/Support/ResourceBufferInterface.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "MEFModel.h"
#include "Support/Buffer.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/FileUtilities.h"

#include <numeric>

using namespace M;
using namespace M::MO;

namespace M::MO {
#define GEN_PASS_DEF_CONSTANTFOLD
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"
} // namespace M::MO

namespace {

// Holds all the information necessary to execute and replace all constant
// subgraphs in a single graph op.
struct SubgraphsInfo {
  GraphOp orig;
  std::string constantSubgraphName;
  // Need the subgraph for its output values.
  SmallVector<Subgraph> subgraphs;
};

struct ValueParamCompare {
  bool operator()(const Operation *lhs, const Operation *rhs) const {
    return hash_value(lhs) < hash_value(rhs);
  }
};

/// Simple struct holding results of a model execution.
struct ModelResult {
  /// Mapping from subgraph output operation to the constants that should
  /// replace it. Also includes the output shape so that unknown shapes in the
  /// constant graph can be resolved.
  llvm::DenseMap<Operation *, SmallVector<std::pair<ElementsAttr, TensorShape>>>
      constants;
  /// Mapping from parameter name in subgraph to the value that it should be
  /// defined as.
  llvm::DenseMap<StringAttr, int64_t> parameters;
};

// Models need to ensure that their backing buffer lives as long as they do.
struct ModelsWithBuffer {
  std::unique_ptr<ModelCollection> models;
  BufferRef buffer;
};

} // namespace

llvm::SetVector<StringAttr> M::MO::Subgraph::paramDeclarations() const {
  llvm::SetVector<StringAttr> params;
  for (auto op : ops) {
    op->walk([&](KGEN::ParamOpInterface paramOp) {
      paramOp.walkDeclarations(
          [&](KGEN::ParamDeclAttr decl) { params.insert(decl.getName()); });
      paramOp.walkDefinitions(
          [&](KGEN::ParamDeclAttr decl, const KGEN::ParamDefValue &val) {
            params.insert(decl.getName());
          });
    });
  }
  return params;
}

llvm::SmallVector<ConstantOp> M::MO::Subgraph::constants() const {
  auto filtered = llvm::make_filter_range(
      ops, [](Operation *op) { return dyn_cast<ConstantOp>(op); });
  return llvm::to_vector(llvm::map_range(
      filtered, [](Operation *op) { return dyn_cast<ConstantOp>(op); }));
}

/// Topologically sort parameters in a list of ops.
static llvm::SmallVector<Operation *>
sortParams(llvm::ArrayRef<Operation *> ops) {
  // ParameterUseDefGraph doesn't map from parameter to operations using it, so
  // we have to construct our own.
  llvm::DenseMap<StringAttr, llvm::SetVector<StringAttr>> uses;
  llvm::DenseMap<StringAttr, Operation *> defs;
  for (auto op : ops) {
    op->walk([&](KGEN::ParamOpInterface paramOp) {
      paramOp.walkDeclarations([&](KGEN::ParamDeclAttr decl) {
        defs[decl.getName()] = op;
        uses[decl.getName()] = {};
      });
      paramOp.walkDefinitions(
          [&](KGEN::ParamDeclAttr decl, const KGEN::ParamDefValue &val) {
            llvm::SetVector<StringAttr> params;
            for (auto expr : val.exprs) {
              expr.walk([&](KGEN::ParamDeclRefAttr ref) {
                params.insert(ref.getName());
              });
            }
            defs[decl.getName()] = op;
            uses[decl.getName()] = params;
          });
    });
  }

  llvm::SetVector<Operation *> sorted;
  llvm::SetVector<StringAttr> inserted;
  // Sort topologically. Probably not quite optimal, but should do.
  llvm::SmallVector<StringAttr> workQueue;
  for (auto [k, _] : uses)
    workQueue.push_back(k);
  while (!workQueue.empty()) {
    StringAttr param = workQueue.back();
    workQueue.pop_back();

    if (inserted.contains(param))
      continue;
    auto notHandled =
        llvm::make_filter_range(uses.at(param), [&](StringAttr param) {
          return !inserted.contains(param);
        });
    if (notHandled.empty()) {
      inserted.insert(param);
      sorted.insert(defs[param]);
      continue;
    }

    workQueue.push_back(param);
    for (auto dependentParam : notHandled)
      workQueue.push_back(dependentParam);
  }

  for (auto op : ops)
    if (!sorted.contains(op))
      sorted.insert(op);
  return sorted.takeVector();
}

// Convert a TensorType to a ShapedType so it can be used with functions in
// Support. TensorType does not implement the ShapedType protocol so this is
// necessary.
static ShapedType typeToShaped(TensorType type) {
  auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(type.getShapeAttr());
  GENERICML_ASSERT(shapeAttr)
      << "Shapes in constants subgraph should all be infered";
  auto intDims = shapeAttr.getIntDims();
  int64_t size = std::accumulate(intDims.begin(), intDims.end(), 1,
                                 std::multiplies<int64_t>());
  return ArrayType::get(type.getContext(), size, type.getElementType());
}

/// Marker for ops that are constant foldable. Currently MO ops and KGEN param
/// ops.
static bool isConstantFoldable(Operation *op) {
  return (isa<MO::MODialect>(op->getDialect()) ||
          dyn_cast<KGEN::ParamOpInterface>(op) ||
          dyn_cast<KGEN::ParamConstantOp>(op) ||
          dyn_cast<KGEN::ParamDeclareOp>(op)) &&
         !dyn_cast<MOControlOpInterface>(op) &&
         !dyn_cast<MO::LayoutTransformOp>(op);
}

/// Creates a a tensor from a `dense_resource`.
/// If the constant op's users are all in this subgraph, then this returns a
/// tensor that owns the underlying resource blob.
/// Otherwise the returned tensor's data is borrowed.
static GML::Tensor
denseResourceAsTensor(ConstantOp constant,
                      DenseResourceElementsAttr denseResource,
                      const Subgraph &subgraph,
                      GML::ForeignBufferInterfaceID resourceInterfaceID) {
  mlir::AsmResourceBlob *blob = denseResource.getRawHandle().getBlob();
  TensorType ttype = dyn_cast<TensorType>(constant.getType());
  TensorSpec spec(dyn_cast<MOSH::ShapeAttr>(ttype.getShapeAttr()).getIntDims(),
                  ttype.getDtype());

  // Find all users of this constant op.
  llvm::SetVector<Operation *> constantUsers;
  for (Operation *user : constant->getUsers())
    constantUsers.insert(user);

  llvm::SetVector<Operation *> usersOutsideSubgraph(constantUsers);
  usersOutsideSubgraph.set_subtract(subgraph.ops);
  if (usersOutsideSubgraph.empty()) {
    // It is safe to take ownership of blob data if all users are in this
    // subgraph.
    GML::BufferRef blobBuffer = GML::BufferRef::createForeign(
        resourceInterfaceID, denseResource.getAsOpaquePointer(),
        blob->getData().size(), blob->getDataAlignment());
    return GML::Tensor(std::move(blobBuffer), std::move(spec));
  }

  // Otherwise borrow the blob data since it has other users.
  return GML::Tensor::createBorrowed(blob->getData().data(), spec,
                                     blob->getDataAlignment(),
                                     GML::BufferRef::Lifetime::kConstant);
}

/// Creates or retrieves an `ElementsAttr` with the same data as `tensor`.
static ElementsAttr
elementsAttrFromTensor(GML::Tensor &tensor, MLIRContext *ctx,
                       GML::ForeignBufferInterfaceID resourceInterfaceID) {
  // Output shape has runtime-dependent size so we need to read out what
  // the size is.
  TensorShape shape = tensor.getShape();
  size_t numElements = shape.getNumElements();
  ArrayType arrayType = ArrayType::get(
      ctx, numElements, toMLIRType(ctx, tensor.getEltType()).get());
  ArrayRef<char> tensorData(tensor.getBuffer<char>(), tensor.getSizeInBytes());
  size_t elementByteAlign = llvm::PowerOf2Ceil(
      llvm::divideCeil(arrayType.getElementTypeBitWidth(), CHAR_BIT));
  // 1. Create an inline attribute for small tensor data.
  if (!shouldUseOutOfLineAttrStorage(arrayType.getNumElements()))
    return getInlineAttrForTensorDataCopy(arrayType, tensorData);

  GML::BufferRef &bufferRef = tensor.getBufferRef();

  // 2. Return the original `dense_resource` if it already exists.
  if (void *opaqueResource =
          bufferRef.maybeReleaseForeignObject(resourceInterfaceID)) {
    // If `bufferRef` holds a foreign resource buffer we end up creating two
    // `dense_resource`s that hold onto the same blob:
    // - The original `dense_resource`.
    // - The new `dense_resource` that holds onto a foreign buffer interface.
    // Hence when the MLIRContext is destroyed, it wants to call the same blob
    // destructor twice, causing a heap-buffer-overflow.
    // To avoid this, return the original `dense_resource` instead of creating a
    // new one.
    return GML::Support::denseResourceFromOpaque(opaqueResource);
  }

  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);
  mlir::AsmResourceBlob blob;
  if (GML::BufferRef::Ownership ownership = bufferRef.getOwnership();
      ownership == GML::BufferRef::kBorrowed) {
    // 3. Copy underlying data to a new `dense_resource` if the buffer is
    //    borrowed.
    blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
        tensorData, elementByteAlign);
  } else {
    GENERICML_ASSERT(llvm::is_contained(
        {GML::BufferRef::kOwned, GML::BufferRef::kForeign}, ownership))
        << "invalid buffer ref ownership";

    // 4. Otherwise, transfer ownership of the buffer ref to a new
    // `dense_resource` if the buffer is owned.
    blob = mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
        tensorData, elementByteAlign,
        [bufferRef = std::move(bufferRef)](void *tensorData, size_t, size_t) {
          // No-op that drops the `bufferRef` when the closure is
          // destroyed.
        });
  }

  return DenseResourceElementsAttr::get(
      arrayType, resourceManager.insert("folded_const", std::move(blob)));
}

/// Run a Model to get the constant outputs to a subgraph. Returns a mapping
/// from output Value to ElementsAttr containing the constant resulting from
/// folding.
static LLCL::ErrorDiagnosticOr<ModelResult>
executeModel(MLIRContext *ctx, const Model *model,
             const SubgraphsInfo &subgraphsInfo, LLCL::Runtime &runtime,
             GML::ForeignBufferInterfaceID resourceInterfaceID) {
  auto executableModelOr = model->setup(runtime);
  if (executableModelOr.isError())
    return executableModelOr.takeError();

  auto executableModel = executableModelOr.takeValue();
  auto setupErrorOrSuccess = executableModel->waitForModelSetup();
  if (setupErrorOrSuccess.isError())
    return LLCL::EncodedDiagnostic(
        Error("Model setup failed, not ready for execution"),
        model->getLocation().copy());

  SmallVector<AnyAsyncValueRef> inputTensorsInMEFIndexOrder;
  for (const Subgraph &subgraph : subgraphsInfo.subgraphs) {
    for (ConstantOp constant : subgraph.constants()) {
      if (auto denseResource =
              dyn_cast<DenseResourceElementsAttr>(constant.getValueAttr())) {
        inputTensorsInMEFIndexOrder.push_back(
            AsyncValueRef<GML::Tensor>::createReady(
                runtime, denseResourceAsTensor(constant, denseResource,
                                               subgraph, resourceInterfaceID)));
      }
    }
  }
  auto results = executableModel->execute(inputTensorsInMEFIndexOrder);

  if (results.empty())
    return LLCL::EncodedDiagnostic("Failed to execute model",
                                   model->getLocation().copy());

  LLCL::await(results);
  for (auto &result : results)
    if (result.isError())
      return LLCL::EncodedDiagnostic(result.takeDiagnostic());

  MutableArrayRef<AnyAsyncValueRef> resultCopy(results.data(), results.size());

  const AnyAsyncValueRef &ch = resultCopy.front();
  GENERICML_ASSERT(ch.isReady() && isa<Chain>(ch))
      << "first result must be a ready chain";
  resultCopy = resultCopy.drop_front();
  llvm::SmallVector<AnyAsyncValueRef> outputTensorRefs;
  outputTensorRefs.reserve(results.size());
  for (auto &result : resultCopy) {
    outputTensorRefs.push_back(
        executableModel->convertToGenericTensor(std::move(result)));
  }
  LLCL::await(outputTensorRefs);

  // Get the constant values.
  llvm::DenseMap<Operation *, SmallVector<std::pair<ElementsAttr, TensorShape>>>
      opToTensors;
  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);
  size_t outIdx = 0;
  for (const Subgraph &subgraph : subgraphsInfo.subgraphs) {
    for (auto op : subgraph.outputs) {
      SmallVector<std::pair<ElementsAttr, TensorShape>> opTensors;
      for (size_t i = 0; i < op->getNumResults(); i++) {
        AnyAsyncValueRef &tensorRef = outputTensorRefs[outIdx];
        if (tensorRef.isError())
          return LLCL::EncodedDiagnostic(tensorRef.takeDiagnostic());

        // Move output tensor into storage.
        GENERICML_ASSERT(tensorRef.isReady())
            << "We already awaited, tensor should be ready";
        GML::Tensor &tensor = tensorRef.get<GML::Tensor>();

        // Create an `ElementsAttr`, possibly by taking ownership of `tensor`'s
        // buffer.
        opTensors.push_back(
            {elementsAttrFromTensor(tensor, ctx, resourceInterfaceID),
             tensor.getShape()});
        outIdx++;
      }
      opToTensors[op] = opTensors;
    }
  }

  // Get parameters now. They come after the constants.
  llvm::DenseMap<StringAttr, int64_t> params;
  for (const Subgraph &subgraph : subgraphsInfo.subgraphs) {
    for (auto paramName : subgraph.paramDeclarations()) {
      auto &tensor = outputTensorRefs[outIdx];
      if (tensor.isError())
        return LLCL::EncodedDiagnostic(tensor.takeDiagnostic());
      GENERICML_ASSERT(tensor.isReady())
          << "We already awaited, tensor should be ready";
      GML::Tensor &t = tensor.get<GML::Tensor>();
      params[paramName] = t.getBuffer<const int64_t>()[0];
      outIdx++;
    }
  }
  return ModelResult{opToTensors, params};
}

/// The use-def graph has both declarations (no associated value) and
/// definitions (x = y). We just need any parameter that is created so we search
/// through both for the defining operation.
static Operation *lookupParamDef(const KGEN::ParameterUseDefGraph &paramUseDef,
                                 StringAttr name) {
  Operation *def;
  auto defIt = paramUseDef.defs.find(name);
  auto declIt = paramUseDef.decls.find(name);
  if (defIt != paramUseDef.defs.end())
    def = defIt->second.defOp;
  else if (declIt != paramUseDef.decls.end())
    def = declIt->second.declOp;
  else
    GENERICML_ASSERT(false) << "Could not find def with name " << name;
  return def;
}

SmallVector<Subgraph>
M::MO::constantSubgraphs(MO::GraphOp graph,
                         KGEN::ParameterCollector::Analysis &paramCache) {
  // Recurse down consumers of ConstantOps, marking ops with all constant
  // inputs as constant. If inputs have not been visited yet, then we
  // recurse up the use chain on inputs until we determine if they are const
  // or not. This process also applies to parameter in the outputs of each op.
  // We track operations instead of values so that we can track parameters in
  // addition to regular ops.

  // Set containing operations that are known to not be constant. That is one or
  // more inputs to the operation are not constant.
  llvm::DenseSet<Operation *> nonConst;

  // Disjoint set containing constant operations. A constant operations is
  // either a mo.constant or an operation who has all constant inputs.
  // Each equivalence class is a connected DAG of values which depend on each
  // other.
  llvm::EquivalenceClasses<Operation *, ValueParamCompare> constSubgraphs;
  // Stack of operations that need to be visited. Values may be popped and then
  // pushed back on the queue if all parent values have not been resolved.
  // New values are added when the parents of an operation are unknown or
  // the users of a value when it has been determined to be constant.
  llvm::SmallVector<Operation *> workQueue;
  graph.walk([&](ConstantOp op) {
    auto dense = dyn_cast<DenseResourceElementsAttr>(op.getValueAttr());
    // If dense is present but doesn't have a resource behind it
    // i.e., the blob is a nullptr, then skip it for constant folding
    if (dense && !(dense.getRawHandle().getBlob())) {
      nonConst.insert(op);
      return;
    }
    constSubgraphs.insert(op);
    for (auto user : op.getResult().getUsers())
      workQueue.push_back(user);
  });
  graph.walk([&](KGEN::ParamConstantOp op) {
    // ParamConstantOp can have parameters that are not constant...
    workQueue.push_back(op);
  });

  // Parameters are not SSA values so we need to construct our own use-def graph
  // for them.
  KGEN::ParameterUseDefGraph paramUseDef(graph.getRegion());
  paramUseDef.calculate(paramCache);

  // Run for a fixed amount of iterations. This avoids unexpected infinite
  // recursion within a graphop.
  size_t iter = 0;
  size_t bound_for_maximum_iterations = 100000;
  for (; iter < bound_for_maximum_iterations; iter++) {
    if (workQueue.empty())
      break;

    Operation *op = workQueue.back();
    workQueue.pop_back();

    // Value is already non-const, skip to next on work queue.
    if (nonConst.contains(op))
      continue;

    // already visited
    if (constSubgraphs.findValue(op) != constSubgraphs.end())
      continue;

    // Don't constant fold any op with regions.
    if (op->getNumRegions() != 0) {
      nonConst.insert(op);
      continue;
    }

    // Skip non-foldable ops
    if (!isConstantFoldable(op)) {
      nonConst.insert(op);
      continue;
    }

    llvm::DenseSet<Operation *> unvisited;
    bool foundNonConst = false;
    for (auto operand : op->getOperands()) {
      // If any operand is non-const then this op/value is also non-const
      Operation *definition = operand.getDefiningOp();
      if (!definition || nonConst.contains(definition)) {
        nonConst.insert(op);
        foundNonConst = true;
        break;
      }
      if (constSubgraphs.findValue(definition) == constSubgraphs.end()) {
        // Operand has not been visited yet.
        unvisited.insert(definition);
      }
    }
    if (foundNonConst)
      continue;

    // Check that all new definitions are const
    llvm::SetVector<Operation *> paramOperands;
    auto walkDeclRef = [&](KGEN::ParamDeclRefAttr paramRef) {
      Operation *def = lookupParamDef(paramUseDef, paramRef.getName());
      if (nonConst.contains(def)) {
        nonConst.insert(op);
        foundNonConst = true;
        return WalkResult::interrupt();
      }
      paramOperands.insert(def);
      // Skip params that are defined by this op.
      if (def != op && constSubgraphs.findValue(def) == constSubgraphs.end())
        unvisited.insert(def);
      return WalkResult::advance();
    };
    auto paramOpInterface = dyn_cast<KGEN::ParamOpInterface>(op);
    if (paramOpInterface) {
      paramOpInterface.walkDefinitions(
          [&](KGEN::ParamDeclAttr param, const ParamDefValue &val) {
            for (auto expr : val.exprs) {
              expr.walk(walkDeclRef);
              if (foundNonConst)
                break;
            }
          });
      if (foundNonConst)
        continue;
    }

    // Go through all parameters and make sure they are const. We only do
    // this for outputs as inputs were already checked by their defining op.
    for (auto resultType : op->getResultTypes()) {
      resultType.walk(walkDeclRef);
      if (foundNonConst)
        break;
    }
    if (foundNonConst)
      continue;

    // Check that all parameters in attrs are const.
    for (auto attr : op->getAttrs())
      attr.getValue().walk(walkDeclRef);
    if (foundNonConst)
      continue;

    if (!unvisited.empty()) {
      // All operands have not been visited, push the current value, then
      // the unvisited operands (so they get handled first).
      workQueue.push_back(op);
      for (auto definition : unvisited)
        workQueue.push_back(definition);
      continue;
    }

    // All operands are const, union this value with all operands.
    constSubgraphs.insert(op);
    for (auto operand : op->getOperands())
      constSubgraphs.unionSets(op, operand.getDefiningOp());
    for (auto param : paramOperands)
      constSubgraphs.unionSets(op, param);

    // Check if users of this now constant value are themselves constant.
    for (auto result : op->getResults())
      for (auto user : result.getUsers())
        workQueue.push_back(user);
  }
  GENERICML_ASSERT(iter < bound_for_maximum_iterations)
      << "Hit bound on maximum number of iterations for detecting constant "
         "subgraphs. Either this graph op is VERY large (in which case this "
         "bound should be increased), or else somehow there is infinite "
         "recursion (which is a bug)";

  // Collect subgraphs from equivalence sets.
  SmallVector<Subgraph> subgraphs;
  for (auto it = constSubgraphs.begin(), end = constSubgraphs.end(); it != end;
       ++it) {
    if (!it->isLeader())
      continue;

    // Ignore subgraphs that only contain constant ops.
    if (llvm::all_of(llvm::iterator_range(constSubgraphs.member_begin(it),
                                          constSubgraphs.member_end()),
                     [](Operation *op) {
                       return dyn_cast<ConstantOp>(op) ||
                              dyn_cast<KGEN::ParamDeclareOp>(op);
                     }))
      continue;
    llvm::SetVector<Operation *> members(constSubgraphs.member_begin(it),
                                         constSubgraphs.member_end());

    subgraphs.push_back(Subgraph(members));
  }

  return subgraphs;
}

/// Copy a subgraph out of its original GraphOp and into a new GraphOp in a
/// different module. Adds OutputOps for all the outputs of the subgraph.
/// Return information to identify the new GraphOp.
/// Large constants in the subgraph are transformed into inputs to the new
/// graph.
static SubgraphsInfo cloneSubgraphs(OpBuilder builder, GraphOp orig,
                                    ModuleOp module,
                                    const SmallVector<Subgraph> &subgraphs,
                                    int64_t &name_counter) {
  // Add large constants as inputs
  llvm::SmallVector<Type> argTypes;
  for (const Subgraph &subgraph : subgraphs) {
    for (ConstantOp constant : subgraph.constants()) {
      auto dense = dyn_cast<DenseResourceElementsAttr>(constant.getValueAttr());
      if (dense)
        argTypes.push_back(constant.getType());
    }
  }
  llvm::SmallVector<Type> resultTypes;
  for (const Subgraph &subgraph : subgraphs) {
    for (auto op : subgraph.outputs) {
      for (auto result : op->getResults()) {
        resultTypes.push_back(result.getType());
      }
    }
  }
  // This name cannot be too long otherwise something breaks :(.
  auto graphop = builder.create<GraphOp>(
      module.getLoc(), (orig.getName() + "_constant_subgraphs").str(), argTypes,
      resultTypes);
  builder.setInsertionPointToEnd(graphop.getBodyBlock());

  llvm::MapVector<ConstantOp, BlockArgument> inputConstants;
  int64_t idx = 0;
  for (const Subgraph &subgraph : subgraphs) {
    for (ConstantOp constant : subgraph.constants()) {
      auto dense = dyn_cast<DenseResourceElementsAttr>(constant.getValueAttr());
      if (dense) {
        inputConstants[constant] = graphop.getBody().getArgument(idx);
        idx++;
      }
    }
  }

  SmallVector<Operation *, 8> opsSorted;
  for (const Subgraph &subgraph : subgraphs)
    for (auto op : subgraph.ops)
      opsSorted.push_back(op);
  // Topologically sort so definitions come before uses.
  // Sort param first, doesn't necessarily maintain topological order.
  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();
  opsSorted = sortParams(opsSorted);
  mlir::computeTopologicalSorting(opsSorted);
  IRMapping oldToNew;
  for (auto oldOp : opsSorted) {
    ConstantOp oldConst = dyn_cast<ConstantOp>(oldOp);
    // Replace large constants with the inputs to this new graph
    if (oldConst) {
      auto it = inputConstants.find(oldConst);
      if (it != inputConstants.end()) {
        oldToNew.map(oldOp->getResult(0), it->second);
        continue;
      }
    }
    builder.clone(*oldOp, oldToNew);
  }
  llvm::SmallVector<Value> newOutputs;
  llvm::SetVector<KGEN::ParamDeclAttr> outputParams;
  for (const Subgraph &subgraph : subgraphs) {
    for (auto op : subgraph.outputs) {
      for (auto result : op->getResults()) {
        Value newOutput = oldToNew.lookup<Value>(result);
        if (newOutput.getType().isa<MOSH::ShapeType>()) {
          // Can't output shapes so we convert it to a rank 1 tensor of unknown
          // size.
          newOutput = builder.create<Shape::ToTensorOp>(
              loc,
              TensorType::get(
                  MOSH::ShapeAttr::get(
                      SmallVector<TypedAttr>(
                          {KGEN::UnknownAttr::get(builder.getIndexType())}),
                      MOSH::ShapeType::get(ctx)),
                  DType::si64),
              newOutput);
        }
        newOutput.getType().walk([&](KGEN::ParamDeclRefAttr ref) {
          outputParams.insert(KGEN::ParamDeclAttr::get(ref));
        });
        newOutputs.push_back(newOutput);
      }
    }
  }

  // Add parameter declaration/definitions that are used in the subgraph. They
  // are converted to indexes and then to tensors so that we can output them.
  Type indexType = IndexType::get(ctx);
  for (const Subgraph &subgraph : subgraphs) {
    for (auto param : subgraph.paramDeclarations()) {
      Value index = builder.create<KGEN::ParamConstantOp>(
          loc, indexType, KGEN::ParamDeclRefAttr::get(ctx, param, indexType));
      newOutputs.push_back(builder.create<IndexToTensorOp>(
          loc,
          TensorType::get(ArrayRef<int64_t>({}),
                          IntegerType::get(ctx, 64, IntegerType::Unsigned)),
          index));
    }
  }
  // No output parameters because we bind them all locally.
  auto out =
      builder.create<OutputOp>(loc, newOutputs, KGEN::ParameterExprArrayAttr{});
  // Have to update the GraphOp's signature.
  graphop.setSignature(
      SignatureType::get(ctx, argTypes, out.getOperandTypes()));
  graphop.setFunctionType(
      mlir::FunctionType::get(ctx, argTypes, out.getOperandTypes()));
  return {orig, std::string(graphop.getName()), subgraphs};
}

/// Replace a constant subgraph with results from executing it. Replaces outputs
/// with constant values, removes ops internal to the subgraph (DCE), and add
/// parameters defined within the subgraph.
static void replaceConstantOps(GraphOp graph, OpBuilder &builder,
                               const ModelResult &modelResult,
                               const SubgraphsInfo &subgraphsInfo) {
  MLIRContext *ctx = graph.getContext();
  // Find ops that are not used outside of the subgraph. Collect them now before
  // we have removed anything, then remove them later when they are not depended
  // on.
  SmallVector<Operation *> allOps;
  for (const Subgraph &subgraph : subgraphsInfo.subgraphs)
    for (auto op : subgraph.ops)
      allOps.push_back(op);
  auto filtered = llvm::make_filter_range(allOps, [&](Operation *op) {
    return llvm::all_of(op->getResults(), [&](OpResult r) {
      return llvm::all_of(
          subgraphsInfo.subgraphs,
          [&](const Subgraph &subgraph) { return subgraph.fullyContains(r); });
    });
  });
  llvm::SmallVector<Operation *> internalOps(filtered.begin(), filtered.end());
  mlir::computeTopologicalSorting(internalOps);

  for (auto [op, tensors] : modelResult.constants) {
    builder.setInsertionPoint(op);
    llvm::SmallVector<Value> newResults;
    for (auto [opResult, constantTensor] :
         llvm::zip(op->getResults(), tensors)) {
      // Each result can either be a tensor or a shape.
      if (opResult.getType().isa<TensorType>()) {
        auto [constant, shape] = constantTensor;
        Type type = opResult.getType();
        // Resolve instances where the output shape is not statically
        // known (mo.tensor<[?], f32> for example) by using the size of
        // the output.
        if (dyn_cast<TensorType>(type).hasUnknownShape()) {
          auto resolvedType = TensorType::get(
              MOSH::ShapeAttr::get(
                  SmallVector<int64_t>(shape.begin(), shape.end()),
                  MOSH::ShapeType::get(ctx)),
              dyn_cast<TensorType>(opResult.getType()).getDtype());
          auto constantOp =
              builder.create<ConstantOp>(op->getLoc(), resolvedType, constant);
          // Rebind the shape so it matches what is expected. For example,
          // transform [1, 2] to [1, A] so that types shapes with
          // parameters match.
          newResults.push_back(
              builder.create<RebindOp>(op->getLoc(), type, constantOp));
        } else {
          newResults.push_back(
              builder.create<ConstantOp>(op->getLoc(), type, constant));
        }
      } else if (opResult.getType().isa<ShapeType>()) {
        auto intElems = dyn_cast<IntArrayElementsAttr>(constantTensor.first);
        newResults.push_back(builder.create<KGEN::ParamConstantOp>(
            op->getLoc(), ShapeAttr::get(intElems.asArrayRef<int64_t>(),
                                         MOSH::ShapeType::get(ctx))));
      } else {
        GENERICML_ASSERT(false)
            << "constants must be tensors or shapes but one is a "
            << opResult.getType();
      }
    }
    op->replaceAllUsesWith(newResults);
    op->erase();
  }
  // Remove internal ops (DCE). Ops are removed in reverse topological ordering
  // so we don't erase a dependency of another op.
  for (auto op = internalOps.rbegin(); op < internalOps.rend(); op++)
    (*op)->erase();

  // Add parameters that were defined in the subgraph.
  builder.setInsertionPointToStart(graph.getBodyBlock());
  for (auto [name, val] : modelResult.parameters) {
    builder.create<KGEN::ParamDeclareOp>(
        graph.getLoc(),
        KGEN::ParamDeclAttr::get(ctx, name, IndexType::get(ctx)),
        builder.getIndexAttr(val));
  }
}

namespace M::MO {

// Interface for controlling whether or not a subgraph is constant folded.
struct ConstantFoldPolicy {
  // Determines which subgraphs should be constant folded. Must return the
  // passed in subgraphs or subsets of each of them. Inputs subgraphs not
  // included in the output will not be constant folded at all. I.e. only
  // the returned subgraphs will be constant folded.
  virtual SmallVector<Subgraph>
  shouldFold(SmallVector<Subgraph> subgraphs) const = 0;
  virtual ~ConstantFoldPolicy() = default;
};

// Fold the largest subgraph that does not increase memory used by more than
// a factor of the space used by the non-folded subgraph.
struct FactorMemIncrease : public ConstantFoldPolicy {
  FactorMemIncrease(double factor, int64_t threshold)
      : factor(factor), threshold(threshold) {}

  // A factor of 1 indicates no increase, a factor of 2 indicates 100% more
  // memory usage max.
  double factor;
  /// Fold everything under this size in bytes.
  int64_t threshold;

  SmallVector<Subgraph>
  shouldFold(SmallVector<Subgraph> subgraphs) const override {
    SmallVector<Subgraph> foldThese;
    for (auto &subgraph : subgraphs) {
      int64_t initialMem = constantMemUsage(subgraph);
      // Greedily remove outputs with the largest size. Might be a smarter
      // way to do this.
      int64_t outputMem = outputMemUsage(subgraph);
      while (!subgraph.outputs.empty() && outputMem > initialMem * factor &&
             outputMem > threshold) {
        // TODO: this is N^2, we should improve it #18475
        Operation *maxValue = *std::max_element(
            subgraph.outputs.begin(), subgraph.outputs.end(),
            [&subgraph](Operation *a, Operation *b) {
              return valueSize(subgraph, a) < valueSize(subgraph, b);
            });
        // TODO: this is N^2, we should improve it #18475
        subgraph.remove(maxValue);
        outputMem = outputMemUsage(subgraph);
      }
      if (!llvm::all_of(subgraph.outputs, [](Operation *op) {
            return dyn_cast<ConstantOp>(op) ||
                   dyn_cast<KGEN::ParamDeclareOp>(op);
          }))
        foldThese.push_back(subgraph);
    }
    return foldThese;
  }
  ~FactorMemIncrease() override = default;

  // Size of Value in bytes.
  static int64_t valueSize(Value v) {
    auto tensor = dyn_cast<TensorType>(v.getType());
    if (tensor)
      return typeToShaped(tensor).getNumElements() *
             tensor.getDtype().getSizeInBytes();

    // index or shape have nominal sizes
    return 0;
  }

  static int64_t valueSize(const Subgraph &subgraph, Operation *op) {
    int64_t sum = 0;
    for (auto result : op->getResults()) {
      if (!subgraph.fullyContains(result))
        sum += valueSize(result);
    }
    return sum;
  }

  // Size of ConstantOp data in bytes.
  static int64_t valueSize(ConstantOp op) { return valueSize(op.getResult()); }

  // Calculate the size in bytes of memory used by the outputs of a
  // subgraph.
  static int64_t outputMemUsage(const Subgraph &subgraph) {
    int64_t outputSize = 0;
    for (auto output : subgraph.outputs)
      outputSize += valueSize(subgraph, output);
    return outputSize;
  }

  // Calculate the size in bytes of memory used by all constant ops in the
  // subgraph.
  static int64_t constantMemUsage(const Subgraph &subgraph) {
    int64_t internalSize = 0;
    for (auto op : subgraph.ops) {
      if (!subgraph.outputs.contains(op)) {
        auto constant = dyn_cast<ConstantOp>(op);
        if (constant)
          internalSize += valueSize(constant);
      }
    }
    return internalSize;
  }
};

struct ConstantFoldPass
    : public M::MO::impl::ConstantFoldBase<ConstantFoldPass> {
  using ConstantFoldBase::ConstantFoldBase;

  void runOnOperation() override;

  ConstantFoldPass()
      : ConstantFoldBase(), policy(new FactorMemIncrease(2, 100)) {}

  ConstantFoldPass(std::string_view label)
      : ConstantFoldBase(), policy(new FactorMemIncrease(2, 100)),
        compiledFrameworkLabel(label) {}

private:
  // Lower a ModuleOp all the way to MGP, export it as MEF, then load it
  // for execution. Returns the models along with a buffer that needs to
  // live as long as they do. This needs to be a member of the pass so it
  // can access runPipeline.
  ErrorOr<ModelsWithBuffer> lowerSubgraphs(ModuleOp module,
                                           LLCL::Runtime &runtime);

  std::shared_ptr<const ConstantFoldPolicy> policy;
  std::string compiledFrameworkLabel;
};

ErrorOr<ModelsWithBuffer>
ConstantFoldPass::lowerSubgraphs(ModuleOp module, LLCL::Runtime &runtime) {
  // Lower module to MGP.
  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);
  MO::MoToMgpOptions moOpts;
  // Avoid infinite recursion.
  moOpts.constantFold = false;
  // Use MOF executor so we don't depend on a framework.
  moOpts.compiledFrameworkLabel = "mof";
  MO::buildMoToMgpPipeline(pm, moOpts, runtime);
  // Run pipeline needs to run on a module.
  if (failed(runPipeline(pm, module)))
    return Error("Failed to run MOToMGP pipeline");

  auto buf = M::WriteableBuffer::get(0, /*alignment=*/0x4000);
  if (failed(convertToMEF(module, *buf, false)))
    return Error("unable to convert file to a MEF binary");

  auto mefFileOr = MEFFile::open(buf->Buffer::getBuffer());
  if (mefFileOr.isError())
    return mefFileOr.takeError();
  auto modelsOr = loadMGPMEFFile(std::move(*mefFileOr));
  if (modelsOr.isError())
    return modelsOr.takeError();
  return {ModelsWithBuffer{std::move(*modelsOr), std::move(buf)}};
}

void ConstantFoldPass::runOnOperation() {
  // At a high level, this pass:
  // 1. Extracts all constant subgraphs.
  // 2. Clones each subgraph into its own GraphOp.
  // 3. Lowers and executes each GraphOp from the previous step.
  // 4. Replaces subgraph outputs with outputs from the previous step.

  MLIRContext *ctx = &getContext();

  // Create separate resource buffer interfaces for different framework drivers.
  // Otherwise crossing framework boundaries when calling
  // `ResourceBufferInterface` member functions causes TypeID issues.
  GML::ForeignBufferInterfaceID resourceInterfaceID =
      GML::Support::registerResourceBufferInterface(compiledFrameworkLabel);

  for (int i = 0; i < iterations.getValue(); i++) {
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(getOperation().getBody());
    // We have to create the copy of each subgraph somewhere under the
    // original model because passes can't create ops outside of what they
    // are operating on. We create a nested module in the current module
    // where we do everything. At the end we remove it.
    auto module = builder.create<ModuleOp>(getOperation().getLoc(),
                                           "constant_fold_subgraphs");

    // For each constant subgraph, we clone it into our submodule as a new
    // GraphOp. The outputs of the GraphOp are the outputs of the subgraph.
    //
    // An alternative approach could be to combine each subgraph into one
    // large constant graph so we only have one thing to execute.
    SmallVector<SubgraphsInfo> subgraphReplacements;
    int64_t name_counter = 0;
    getOperation().walk([&](GraphOp graph) {
      // Can't constant fold cuda yet
      if (graph.getDeviceRef() && graph.getDeviceRef().getLabel() != kCPULabel)
        return mlir::WalkResult::skip();
      auto subgraphs = M::MO::constantSubgraphs(
          graph, getAnalysis<KGEN::ParameterCollector::Analysis>());
      if (subgraphs.empty())
        return mlir::WalkResult::advance();
      subgraphs = policy->shouldFold(subgraphs);
      if (subgraphs.empty())
        return mlir::WalkResult::advance();
      builder.setInsertionPointToEnd(module.getBody());

      subgraphReplacements.push_back(
          cloneSubgraphs(builder, graph, module, subgraphs, name_counter));
      return WalkResult::advance();
    });

    if (module.getBody()->getOperations().empty()) {
      // Remove module to not clutter the output.
      module.erase();
      return;
    }

    // TODO: #18077 this should be an argument to the pass.
    auto ownedRuntime = LLCL::createRuntimeIfNeeded(
        LLCL::RuntimeOptions().withLeakCheckedAllocator());

    ErrorOr<ModelsWithBuffer> modelsOr = lowerSubgraphs(module, *ownedRuntime);
    if (failed(modelsOr)) {
      mlir::emitError(module.getLoc(),
                      "[ConstantFold] Unable to load models: " +
                          Twine(modelsOr.getError()));
      return signalPassFailure();
    }

    // We have to consolidate rewrites to the same graphop as the rewrites of
    // one subgraph may invalidate ops in another subgraph (due to fold hooks).
    llvm::DenseMap<GraphOp,
                   llvm::SmallVector<std::pair<ModelResult, SubgraphsInfo>>>
        executionResults;
    for (const SubgraphsInfo &subgraphsInfo : subgraphReplacements) {
      const Model *model = modelsOr.get().models->getModelNamed(
          subgraphsInfo.constantSubgraphName);
      GENERICML_ASSERT(model != nullptr)
          << "Model should exist, make sure names above match.";
      LLCL::ErrorDiagnosticOr<ModelResult> valuesToTensorsOr = executeModel(
          ctx, model, subgraphsInfo, *ownedRuntime, resourceInterfaceID);
      if (valuesToTensorsOr.isError()) {
        LLCL::EncodedDiagnostic err = valuesToTensorsOr.takeError();
        auto decoded = err.decodeLocation();
        auto loc =
            mlir::FileLineColLoc::get(module.getContext(), decoded.filename,
                                      decoded.line, decoded.column);
        mlir::emitError(loc, "[ConstantFold] " + Twine(err.getMessage().get()));
        return signalPassFailure();
      }
      auto valuesToTensors = valuesToTensorsOr.takeValue();
      auto it = executionResults.find(subgraphsInfo.orig);
      if (it != executionResults.end()) {
        it->second.push_back({valuesToTensors, subgraphsInfo});
      } else {
        executionResults[subgraphsInfo.orig] = {
            {valuesToTensors, subgraphsInfo}};
      }
    }

    // Replace output expressions with constants. Also handles DCE.
    for (auto [graph, results] : executionResults) {
      for (const auto &r : results) {
        const ModelResult &modelResult = r.first;
        const SubgraphsInfo &subgraphsInfo = r.second;
        replaceConstantOps(graph, builder, modelResult, subgraphsInfo);
      }
    }

    // Remove module to not clutter the output.
    module.erase();

    // Run canonicalization to uncover more constant folding.
    mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
        ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      mlir::emitError(getOperation().getLoc(),
                      "[ConstantFold] Failed to run canonicalization pass");
      return;
    }
  }
}

} // namespace M::MO

std::unique_ptr<mlir::Pass>
M::MO::createConstantFoldPass(std::string_view compiledFrameworkLabel) {
  return std::make_unique<ConstantFoldPass>(compiledFrameworkLabel);
}
