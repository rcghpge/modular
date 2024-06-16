//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstantFold.h"
#include "GenericML/Driver/CXXDriverAPI.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "GenericML/Support/BlobLifetimeManager.h"
#include "GenericML/Support/List.h"
#include "GenericML/Support/Model.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "MEFModel.h"
#include "Support/Buffer.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "Support/Context.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/FileUtilities.h"

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

using RuntimeTensor = std::pair<ElementsAttr, TensorShape>;
using RuntimeList = std::vector<RuntimeTensor>;
using RuntimeOutput = std::variant<RuntimeTensor, RuntimeList>;

/// Simple struct holding results of a model execution.
struct ModelResult {
  /// Mapping from subgraph output operation to the constants that should
  /// replace it. Also includes the output shape so that unknown shapes in the
  /// constant graph can be resolved.
  llvm::DenseMap<Operation *, SmallVector<RuntimeOutput>> constants;
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

/// Marker for ops that are constant foldable. Currently MO ops and KGEN param
/// ops. If true, might be constant foldable, if false, definitely not.
static bool isMaybeConstantFoldable(Operation *op) {
  // Don't constant fold any op with regions.
  if (op->getNumRegions() != 0)
    return false;

  // Don't include transfers between devices.
  if (isa<MO::TransferOp>(op))
    return false;

  // Skip ops with parameterized shape
  auto findParamDeclRefAttr = [](KGEN::ParamDeclRefAttr paramRef) {
    return WalkResult::interrupt();
  };
  auto findUnknownAttr = [](KGEN::UnknownAttr param) {
    return WalkResult::interrupt();
  };
  for (auto operandTy : op->getOperandTypes()) {
    if (operandTy.walk(findParamDeclRefAttr, findUnknownAttr)
            .wasInterrupted()) {
      return false;
    }
  }
  for (auto resultTy : op->getResultTypes()) {
    if (resultTy.walk(findParamDeclRefAttr, findUnknownAttr).wasInterrupted()) {
      return false;
    }
  }
  for (auto attr : op->getAttrs()) {
    if (attr.getValue()
            .walk(findParamDeclRefAttr, findUnknownAttr)
            .wasInterrupted()) {
      return false;
    }
  }

  // Skip ops which declare/define parameters (NOTE: they may not use them)
  auto paramOpInterface = dyn_cast<KGEN::ParamOpInterface>(op);
  if (paramOpInterface) {
    bool definesOrDeclaresParams = false;
    paramOpInterface.walkDefinitions(
        [&](KGEN::ParamDeclAttr param, const ParamDefValue &val) {
          definesOrDeclaresParams = true;
        });
    paramOpInterface.walkDeclarations(
        [&](KGEN::ParamDeclAttr param) { definesOrDeclaresParams = true; });
    if (definesOrDeclaresParams)
      return false;
  }

  return ((isa<MO::MODialect, MOGG::MOGGDialect>(op->getDialect()) &&
           // TODO (#29484): Graphs with mo.list outputs can't be lowered to
           // MGP.
           llvm::all_of(
               op->getResultTypes(),
               [](Type ty) {
                 return isa<MO::TensorType, MOSH::ShapeType, MO::ListType>(ty);
               })) ||
          isa<KGEN::ParamOpInterface, MOSH::ParamToValueOp,
              KGEN::ParamDeclareOp>(op)) &&
         !isa<MOControlOpInterface>(op);
}

/// Creates a a tensor from a `dense_resource`.
/// If the constant op's users are all in this subgraph, then this returns a
/// tensor that owns the underlying resource blob.
/// Otherwise the returned tensor's data is borrowed.
static Tensor denseResourceAsTensor(LLCL::Runtime &runtime, ConstantOp constant,
                                    DenseResourceElementsAttr denseResource,
                                    const Subgraph &subgraph) {
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

  // This optimization is used to aggressively free DenseResource allocations
  // during execution. The buffers borrow dense resources which have no other
  // uses and we free the dense resources when the buffers are dead rather
  // than waiting for the MLIRContext to die.
  if (usersOutsideSubgraph.empty() && blob->getDeleter()) {
    // Take ownership of the deleting function.
    auto async = AsyncValue::createReady<GML::BlobLifetimeManager>(
        runtime.getCompactPtr(), std::move(blob->getDeleter()), blob->getData(),
        blob->getDataAlignment());
    blob->getDeleter() = nullptr;
    auto ref = AnyAsyncValueRef::take(async);

    // It is safe to take ownership of blob data if all users are in this
    // subgraph.
    TensorBufferRef blobBuffer = TensorBufferRef::create(
        reinterpret_cast<void *>(const_cast<char *>(blob->getData().data())),
        blob->getData().size(), std::move(ref), blob->getDataAlignment());
    return Tensor(std::move(blobBuffer), std::move(spec));
  }

  return Tensor::createBorrowed(blob->getData().data(), spec,
                                blob->getDataAlignment(),
                                TensorBufferRef::Lifetime::kConstant);
}

/// Creates or retrieves an `ElementsAttr` with the same data as `ref`
/// `ref` is expected to hold a reference to a tensor object.
static ElementsAttr elementsAttrFromTensor(const AnyAsyncValueRef &ref,
                                           MLIRContext *ctx) {
  // Output shape has runtime-dependent size so we need to read out what
  // the size is.
  Tensor &tensor = ref.get<Tensor>();
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

  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);
  TensorBufferRef &bufferRef = tensor.getBufferRef();

  // Reuses the data within the BufferRef and extends the lifetime of it.
  mlir::AsmResourceBlob blob =
      mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
          tensorData, elementByteAlign,
          [storage = bufferRef.getMemStorageHandle().copy()](void *, size_t,
                                                             size_t) {
            // No-op that drops the `bufferRef` when the closure is
            // destroyed.
          });

  return DenseResourceElementsAttr::get(
      arrayType, resourceManager.insert("folded_const", std::move(blob)));
}

static RuntimeOutput extractRuntimeOutput(
    const AnyAsyncValueRef &ref, MLIRContext *ctx,
    llvm::DenseMap<M::LLCL::AsyncValue *, ElementsAttr> &cachedConstants) {
  if (ref.isType<Tensor>()) {
    Tensor &tensor = ref.get<Tensor>();
    ElementsAttr constantsArr;
    if (auto it = cachedConstants.find(ref.getPointer());
        it != cachedConstants.end()) {
      // The same output value was returned multiple times by the graph.
      // There is no need to create the same attribute another time, just reuse
      // the existing one.
      // This avoids creating twice the same dense_resources.
      // For example, if we constant-fold the following IR:
      //  mo.if(...) {
      //     %1 = some_op(%cst0)
      //  } else {
      //     %1 = some_op(%cst0)
      // }
      // Without this cache, it would generate:
      // mo.if(...) {
      //   %1 = constant { value=<folded_constant_1> }
      // } else {
      //   %1 = constant { value=<folded_constant_2> }
      // }
      constantsArr = it->second;
    } else {
      // Create an `ElementsAttr`, possibly by taking ownership of
      // `ref/tensor`'s buffer.
      constantsArr = elementsAttrFromTensor(ref, ctx);
      cachedConstants.try_emplace(ref.getPointer(), constantsArr);
    }
    return RuntimeOutput(RuntimeTensor(constantsArr, tensor.getShape()));
  } else if (ref.isType<GML::List>()) {
    GML::List &list = ref.get<GML::List>();
    RuntimeList out;
    for (auto &v : list.getElements()) {
      out.push_back(std::get<RuntimeTensor>(
          extractRuntimeOutput(v.getValue(), ctx, cachedConstants)));
    }
    return out;
  } else {
    assert(false && "Unexpected GML type");
  }
}

/// Run a Model to get the constant outputs to a subgraph. Returns a mapping
/// from output Value to ElementsAttr containing the constant resulting from
/// folding.
static LLCL::ErrorDiagnosticOr<ModelResult>
executeModel(MLIRContext *ctx, const Model *model,
             const SubgraphsInfo &subgraphsInfo) {
  LLCL::Runtime &runtime = *loadContext(ctx)->get<LLCL::Runtime>();
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
            AsyncValueRef<Tensor>::createReady(
                runtime, denseResourceAsTensor(runtime, constant, denseResource,
                                               subgraph)));
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
  ASSERT_STREAM(ch.isReady() && isa<Chain>(ch),
                << "first result must be a ready chain");
  resultCopy = resultCopy.drop_front();

  // Get the constant values.
  llvm::DenseMap<M::LLCL::AsyncValue *, ElementsAttr> cachedConstants;
  llvm::DenseMap<Operation *, SmallVector<RuntimeOutput>> opToTensors;
  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);
  size_t outIdx = 0;
  for (const Subgraph &subgraph : subgraphsInfo.subgraphs) {
    for (auto op : subgraph.outputs) {
      SmallVector<RuntimeOutput> opTensors;
      for (size_t i = 0; i < op->getNumResults(); i++) {
        AnyAsyncValueRef &tensorRef = resultCopy[outIdx];
        if (tensorRef.isError())
          return LLCL::EncodedDiagnostic(tensorRef.takeDiagnostic());

        // Move output tensor into storage.
        ASSERT_STREAM(tensorRef.isReady(),
                      << "We already awaited, tensor should be ready");
        opTensors.push_back(
            extractRuntimeOutput(tensorRef, ctx, cachedConstants));
        outIdx++;
      }
      opToTensors[op] = opTensors;
    }
  }

  // Get parameters now. They come after the constants.
  llvm::DenseMap<StringAttr, int64_t> params;
  for (const Subgraph &subgraph : subgraphsInfo.subgraphs) {
    for (auto paramName : subgraph.paramDeclarations()) {
      auto &tensor = resultCopy[outIdx];
      if (tensor.isError())
        return LLCL::EncodedDiagnostic(tensor.takeDiagnostic());
      ASSERT_STREAM(tensor.isReady(),
                    << "We already awaited, tensor should be ready");
      ASSERT_STREAM(tensor.isType<Tensor>(),
                    << "Subgraph shape output should be a tensor");
      Tensor &t = tensor.get<Tensor>();
      params[paramName] = t.getBuffer<const int64_t>()[0];
      outIdx++;
    }
  }
  return ModelResult{opToTensors, params};
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

  // Run for a fixed amount of iterations. This avoids unexpected infinite
  // recursion within a graphop.
  size_t iter = 0;
  size_t bound_for_maximum_iterations = 100000;
  for (; iter < bound_for_maximum_iterations; iter++) {
    if (workQueue.empty())
      break;

    Operation *op = workQueue.back();
    workQueue.pop_back();

    // already visited
    if (constSubgraphs.findValue(op) != constSubgraphs.end())
      continue;

    // Value is already non-const, skip to next on work queue.
    if (nonConst.contains(op))
      continue;

    if (!isMaybeConstantFoldable(op)) {
      nonConst.insert(op);
      continue;
    }

    // Check all operands are const
    llvm::DenseSet<Operation *> unvisitedOperands;
    bool foundNonConst = false;
    for (auto operand : op->getOperands()) {
      // If any operand is non-const then this op/value is also non-const
      Operation *definition = operand.getDefiningOp();
      if (!definition || nonConst.contains(definition)) {
        foundNonConst = true;
        break;
      }
      if (constSubgraphs.findValue(definition) == constSubgraphs.end()) {
        // Operand has not been visited yet.
        unvisitedOperands.insert(definition);
      }
    }
    if (foundNonConst) {
      nonConst.insert(op);
      continue;
    }

    // All operands have not been visited, push the current value, then
    // the unvisited operands (so they get handled first).
    if (!unvisitedOperands.empty()) {
      workQueue.push_back(op);
      for (auto definition : unvisitedOperands)
        workQueue.push_back(definition);
      continue;
    }

    // All operands are const, union this value with all operands.
    constSubgraphs.insert(op);
    for (auto operand : op->getOperands())
      constSubgraphs.unionSets(op, operand.getDefiningOp());

    // Check if users of this now constant value are themselves constant.
    for (auto result : op->getResults())
      for (auto user : result.getUsers())
        workQueue.push_back(user);
  }
  ASSERT_STREAM(
      iter < bound_for_maximum_iterations,
      << "Hit bound on maximum number of iterations for detecting constant "
         "subgraphs. Either this graph op is VERY large (in which case this "
         "bound should be increased), or else somehow there is infinite "
         "recursion (which is a bug)");

  // Collect subgraphs from equivalence sets.
  SmallVector<Subgraph> subgraphs;
  for (auto it = constSubgraphs.begin(), end = constSubgraphs.end(); it != end;
       ++it) {
    if (!it->isLeader())
      continue;

    llvm::SetVector<Operation *> members(constSubgraphs.member_begin(it),
                                         constSubgraphs.member_end());
    Subgraph subgraph(members);
    // Remove any mo.list.create that are the output of the op as they will get
    // recreated after executing the subgraph.
    llvm::SmallVector<Operation *> toRemove;
    for (auto output : subgraph.outputs)
      if (isa<ListCreateOp>(output))
        toRemove.push_back(output);
    for (auto op : toRemove)
      subgraph.remove(op);

    // Ignore subgraphs that only contain constant ops.
    if (llvm::all_of(subgraph.ops, [](Operation *op) {
          return isa<ConstantOp>(op) || isa<KGEN::ParamDeclareOp>(op);
        }))
      continue;

    subgraphs.push_back(subgraph);
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
        if (llvm::isa<MOSH::ShapeType>(newOutput.getType())) {
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
      Value index = builder.create<MOSH::ParamToValueOp>(
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

  // Rewrite all device attrs to target CPU.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](DeviceRefAttr ref) { return DeviceRefAttr::get(ctx, kCPULabel, 0); });
  replacer.recursivelyReplaceElementsIn(graphop, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);

  return {orig, std::string(graphop.getName()), subgraphs};
}

static Value runtimeTensorToConstant(OpBuilder &builder, Location loc,
                                     RuntimeTensor tensor, Type type) {
  auto [constant, shape] = std::move(tensor);
  // Resolve instances where the output shape is not statically
  // known (mo.tensor<[?], f32> for example) by using the size of
  // the output.
  if (cast<TensorType>(type).hasUnknownShape()) {
    auto resolvedType = TensorType::get(
        MOSH::ShapeAttr::get(SmallVector<int64_t>(shape.begin(), shape.end()),
                             MOSH::ShapeType::get(builder.getContext())),
        cast<TensorType>(type).getDtype());
    auto constantOp = builder.create<ConstantOp>(loc, resolvedType, constant);
    // Rebind the shape so it matches what is expected. For example,
    // transform [1, 2] to [1, A] so that types shapes with
    // parameters match.
    return builder.create<RebindOp>(loc, type, constantOp);
  } else {
    return builder.create<ConstantOp>(loc, type, constant);
  }
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

  for (auto constant : modelResult.constants) {
    auto &op = constant.first;
    auto &tensors = constant.second;
    builder.setInsertionPoint(op);
    llvm::SmallVector<Value> newResults;
    for (auto [opResult, constantTensor] :
         llvm::zip(op->getResults(), tensors)) {
      Type type = opResult.getType();
      // Each result can either be a tensor or a shape.
      if (llvm::isa<TensorType>(type)) {
        newResults.push_back(runtimeTensorToConstant(
            builder, op->getLoc(), std::get<RuntimeTensor>(constantTensor),
            type));
      } else if (llvm::isa<ShapeType>(type)) {
        auto intElems = dyn_cast<IntArrayElementsAttr>(
            std::get<RuntimeTensor>(constantTensor).first);
        newResults.push_back(builder.create<MOSH::ParamToValueOp>(
            op->getLoc(), ShapeAttr::get(intElems.asArrayRef<int64_t>(),
                                         MOSH::ShapeType::get(ctx))));
      } else if (llvm::isa<ListType>(type)) {
        auto tensors =
            llvm::map_to_vector(std::get<RuntimeList>(constantTensor),
                                [&](const RuntimeTensor &tensor) {
                                  return runtimeTensorToConstant(
                                      builder, op->getLoc(), tensor,
                                      cast<ListType>(type).getElementType());
                                });
        newResults.push_back(
            builder.create<ListCreateOp>(op->getLoc(), type, tensors));
      } else {
        ASSERT_STREAM(false,
                      << "constants must be tensors or shapes but one is a "
                      << opResult.getType());
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

struct ConstantFoldPass
    : public M::MO::impl::ConstantFoldBase<ConstantFoldPass> {
  using ConstantFoldBase::ConstantFoldBase;

  void runOnOperation() override;

  ConstantFoldPass()
      : ConstantFoldBase(), policy(new FactorMemIncrease(2, 100)) {}

  ConstantFoldPass(std::string_view label, const MO::MoToMgpOptions &origOpts)
      : ConstantFoldBase(), policy(new FactorMemIncrease(2, 100)),
        compiledFrameworkLabel(label) {
    // Copy some of the options used for main compilation to compilation of the
    // subgraph. We don't need to copy all options, we only want the ones that
    // affect correctness.
    cacheDir = origOpts.cacheDir;
    kernelLibraryPaths =
        origOpts.kernelLibraryPaths.operator ArrayRef<std::string>();
  }

private:
  // Lower a ModuleOp all the way to MGP, export it as MEF, then load it
  // for execution. Returns the models along with a buffer that needs to
  // live as long as they do. This needs to be a member of the pass so it
  // can access runPipeline.
  ErrorOr<ModelsWithBuffer> lowerSubgraphs(ModuleOp module);

  std::shared_ptr<const ConstantFoldPolicy> policy;
  // Label used to load and handle buffers created by the framework.
  std::string compiledFrameworkLabel;
  std::optional<std::string> cacheDir;
  std::optional<std::vector<std::string>> kernelLibraryPaths;
};

ErrorOr<ModelsWithBuffer> ConstantFoldPass::lowerSubgraphs(ModuleOp module) {
  // Lower module to MGP.
  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);
  MO::MoToMgpOptions moOpts;
  if (cacheDir)
    moOpts.cacheDir = cacheDir.value();
  if (kernelLibraryPaths)
    moOpts.kernelLibraryPaths = kernelLibraryPaths.value();
  // Avoid infinite recursion.
  moOpts.constantFold = false;
  // Reduce Memory overhead by not fusing compiled-regions primitives.
  moOpts.enableCompiledRegionsFusion = false;
  // Use MOF executor so we don't depend on a framework.
  moOpts.compiledFrameworkLabel = "mof";
  // Set default device label to cpu
  moOpts.defaultDeviceLabel = kCPULabel;
  // Force all kernels to be synchronous. This prevents higher memory usage.
  moOpts.forceSync = true;
  MO::buildMoToMgpPipeline(pm, moOpts);
  // Run pipeline needs to run on a module.
  if (failed(runPipeline(pm, module)))
    return Error("Failed to run MOToMGP pipeline");

  // NOTE: `convertMGPToMEF` destroys `dense_resource`s, so it is only safe to
  // call here without further precaution because all `dense_resource`s in
  // constant folding are operands, not attributes in the subgraph.
  auto bufOr = MO::convertMGPToMEF(module, false);
  if (bufOr)
    return bufOr.takeError();
  BufferRef buf = bufOr.takeValue();

  auto mefFileOr = MEFFile::open(buf->getBuffer());
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

  ContextRef maxContext = loadContext(ctx);
  ASSERT_STREAM(maxContext.getPointer() != nullptr,
                << "expected max context to be setup here");
  LLCL::Runtime *runtimeFromCtx = maxContext->get<Runtime>();

  auto readyChain = AsyncValueRef<Chain>::allocate(runtimeFromCtx);

  LLCL::addTask(*runtimeFromCtx, [&, readyChain = readyChain.copy()]() mutable {
    auto setError = [&](mlir::Location loc, const Twine &errorMessage) {
      auto encodedLocation = LLCL::MLIRLocationDecoder::getEncodedLocation(loc);
      return std::move(readyChain)
          .setToError({Error(errorMessage), std::move(encodedLocation)});
    };
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
        return std::move(readyChain).emplace();
      }

      ErrorOr<ModelsWithBuffer> modelsOr = lowerSubgraphs(module);
      if (failed(modelsOr)) {
        return setError(module.getLoc(),
                        "[ConstantFold] Unable to load models: " +
                            Twine(modelsOr.getError()));
      }

      // We have to consolidate rewrites to the same graphop as the rewrites of
      // one subgraph may invalidate ops in another subgraph (due to fold
      // hooks).
      llvm::DenseMap<GraphOp,
                     llvm::SmallVector<std::pair<ModelResult, SubgraphsInfo>>>
          executionResults;
      for (const SubgraphsInfo &subgraphsInfo : subgraphReplacements) {
        const Model *model = modelsOr.get().models->getModelNamed(
            subgraphsInfo.constantSubgraphName);
        ASSERT_STREAM(model != nullptr,
                      << "Model should exist, make sure names above match.");
        LLCL::ErrorDiagnosticOr<ModelResult> valuesToTensorsOr =
            executeModel(ctx, model, subgraphsInfo);
        if (valuesToTensorsOr.isError()) {
          LLCL::EncodedDiagnostic err = valuesToTensorsOr.takeError();
          auto decoded = err.decodeLocation();
          auto loc =
              mlir::FileLineColLoc::get(module.getContext(), decoded.filename,
                                        decoded.line, decoded.column);
          return setError(loc,
                          "[ConstantFold] " + Twine(err.getMessage().get()));
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
        return setError(getOperation().getLoc(),
                        "[ConstantFold] Failed to run canonicalization pass");
      }
    }
    return std::move(readyChain).emplace();
  });
  LLCL::await(readyChain);
  if (readyChain.isError()) {
    std::string errorMessage = readyChain.getDiagnostic().getMessage().get();
    LLCL::DecodedLocation decoded = readyChain.getDiagnostic().decodeLocation();
    auto loc = mlir::FileLineColLoc::get(ctx, decoded.filename, decoded.line,
                                         decoded.column);
    mlir::emitError(loc, errorMessage);
    return signalPassFailure();
  }
}

} // namespace M::MO

std::unique_ptr<mlir::Pass>
M::MO::createConstantFoldPass(std::string_view compiledFrameworkLabel,
                              const MoToMgpOptions &opts) {
  return std::make_unique<ConstantFoldPass>(compiledFrameworkLabel, opts);
}
