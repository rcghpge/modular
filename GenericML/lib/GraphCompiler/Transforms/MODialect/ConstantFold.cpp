//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstantFold.h"
#include "GenericML/Driver/CXXDriverAPI.h"
#include "GenericML/Driver/CompileDriver.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "GenericML/Support/Model.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "MEFModel.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/FileUtilities.h"

#include <numeric>

namespace M::MO {
#define GEN_PASS_DEF_CONSTANTFOLD
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

// Holds all the information necessary to execute and replace a constant
// subgraph
struct SubgraphInfo {
  GraphOp orig;
  std::string constantSubgraphName;
  // Need the subgraph for its output values.
  Subgraph subgraph;
};

struct ValueCompare {
  bool operator()(const Value &lhs, const Value &rhs) const {
    return hash_value(lhs) < hash_value(rhs);
  }
};

// Convert a TensorType to a ShapedType so it can be used with functions in
// Support. TensorType does not implement the ShapedType protocol so this is
// necessary.
ShapedType typeToShaped(TensorType type) {
  auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(type.getShapeAttr());
  assert(shapeAttr && "Shapes in constants subgraph should all be infered");
  auto intDims = shapeAttr.getIntDims();
  int64_t size = std::accumulate(intDims.begin(), intDims.end(), 1,
                                 std::multiplies<int64_t>());
  return ArrayType::get(type.getContext(), size, type.getElementType());
}

/// Rewrite ops to the results of constant folding them.
struct RewriteConstantOps : public RewritePattern {
  RewriteConstantOps(MLIRContext *context,
                     const llvm::DenseMap<Value, ElementsAttr> &constants)
      : RewritePattern(mlir::Pattern::MatchInterfaceOpTypeTag(),
                       MO::MOTensorOpInterface::getInterfaceID(), 1, context),
        constants(constants) {}
  using RewritePattern::matchAndRewrite;
  const llvm::DenseMap<Value, ElementsAttr> &constants;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0)
      return rewriter.notifyMatchFailure(op, "Op has no results");

    auto it = constants.find(op->getResult(0));
    if (it == constants.end())
      return rewriter.notifyMatchFailure(op, "Op is not constant");

    rewriter.setInsertionPoint(op);
    llvm::SmallVector<Value> new_results;
    for (auto r : op->getResults()) {
      new_results.push_back(rewriter.create<ConstantOp>(
          op->getLoc(), r.getType(), constants.at(r)));
    }
    rewriter.replaceOp(op, new_results);
    return LogicalResult::success();
  }
};

/// Run a Model to get the constant outputs to a subgraph. Returns a mapping
/// from output Vale to ElementsAttr containing the constant resulting from
/// folding.
ErrorOr<llvm::DenseMap<Value, ElementsAttr>>
executeModel(MLIRContext *ctx, const Model *model, SubgraphInfo subgraphInfo,
             LLCL::Runtime &runtime) {
  auto executableModelOr = model->setup(runtime);
  if (executableModelOr.isError())
    return executableModelOr.takeError();

  auto executableModel = executableModelOr.takeValue();
  auto setupErrorOrSuccess = executableModel->waitForModelSetup();
  if (setupErrorOrSuccess.isError())
    return Error("Model setup failed, not ready for execution");

  SmallVector<AnyAsyncValueRef> inputTensorsInMEFIndexOrder;
  auto results = executableModel->execute(inputTensorsInMEFIndexOrder);

  if (results.empty())
    return Error("Failed to execute model");

  LLCL::await(results);
  for (auto &result : results)
    if (result.isError())
      return Error(result.getDiagnostic().getMessage().get());

  MutableArrayRef<AnyAsyncValueRef> resultCopy(results.data(), results.size());
  resultCopy = resultCopy.drop_front();
  llvm::SmallVector<AnyAsyncValueRef> outputTensorRefs;
  outputTensorRefs.reserve(results.size());
  for (auto &result : resultCopy) {
    outputTensorRefs.push_back(
        executableModel->convertToGenericTensor(std::move(result)));
  }
  LLCL::await(outputTensorRefs);

  llvm::DenseMap<Value, ElementsAttr> valuesToTensors;
  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);
  for (auto [outIdx, tensor] : llvm::enumerate(outputTensorRefs)) {
    if (tensor.isError())
      return Error(tensor.getDiagnostic().getMessage().get());

    // Copy output tensor into storage. Ideally we could preallocate the
    // output, but the interface does not support it.
    assert(tensor.isReady() && "We already awaited, tensor should be ready");
    Value outValue = subgraphInfo.subgraph.outputs[outIdx];
    Tensor t = std::move(tensor.get<Tensor>());
    valuesToTensors[outValue] = getAttrForTensorDataCopy(
        typeToShaped(dyn_cast<TensorType>(outValue.getType())),
        subgraphInfo.constantSubgraphName + "_" + std::to_string(outIdx),
        {reinterpret_cast<const char *>(t.getBuffer()), t.getSizeInBytes()},
        resourceManager);
  }
  return valuesToTensors;
}

SmallVector<Subgraph> ConstantSubgraphs(MO::GraphOp graph) {
  // Recurse down consumers ConstantOps, marking ops with all constant inputs as
  // constant. If inputs have not been visited yet, then we recurse up the use
  // chain on inputs until we determine if they are const or not.

  // Set containing values that are known to not be constant. That is one or
  // more inputs to the operation creating the values are not constant.
  llvm::DenseSet<Value> non_const;

  // Disjoint set containing constant values. A constant value is either a
  // mo.constant or a value whose defining operation has all constant value
  // inputs. Each equivalence class is a connected DAG of values which depend on
  // each other.
  llvm::EquivalenceClasses<Value, ValueCompare> const_subgraphs;
  // Stack of values that need to be visited. Values may be popped and then
  // pushed back on the queue if all parent values have not been resolved. New
  // values are added when the parents of an operation are unknown or the users
  // of a value when it has been determined to be constant.
  llvm::SmallVector<Value> work_queue;
  graph.walk([&](ConstantOp op) {
    for (auto user : op.getResult().getUsers())
      for (auto result : user->getResults())
        work_queue.push_back(result);
    const_subgraphs.insert(op.getResult());
  });

  // Run for a fixed amount of iterations. This avoids unexpected infinite
  // recursion within a graphop.
  size_t iter = 0;
  size_t bound_for_maximum_iterations = 100000;
  for (; iter < bound_for_maximum_iterations; iter++) {
    if (work_queue.empty())
      break;

    Value v = work_queue.back();
    work_queue.pop_back();

    // Value is already non-const, skip to next on work queue.
    if (non_const.contains(v))
      continue;

    // already visited
    if (const_subgraphs.findValue(v) != const_subgraphs.end())
      continue;

    auto op = v.getDefiningOp();
    // If there is no defining op, then this value is not constant.
    if (!op) {
      non_const.insert(v);
      continue;
    }

    // Don't constant fold any op with regions.
    if (op->getNumRegions() != 0) {
      non_const.insert(v);
      continue;
    }

    // We have hit a root op (does not have input operands). For now we treat
    // this as a non-const. Currently includes kgen.param.constant.
    if (op->getNumOperands() == 0) {
      non_const.insert(v);
      continue;
    }

    // If an op declares parameters then it is not constant.
    auto paramDeclInterface = dyn_cast<ParamDeclarationInterface>(op);
    if (paramDeclInterface &&
        paramDeclInterface.getOutputParamDecls().size() != 0) {
      non_const.insert(v);
      continue;
    }

    // Non constant output shapes are not constant.
    if (!dyn_cast<TensorType>(v.getType()).hasStaticShape()) {
      non_const.insert(v);
      continue;
    }

    llvm::DenseSet<Value> unvisited;
    bool foundNonConst = false;
    for (auto operand : op->getOperands()) {
      // If any operand is non-const then this op/value is also non-const
      if (non_const.contains(operand)) {
        non_const.insert(v);
        foundNonConst = true;
        break;
      }
      if (const_subgraphs.findValue(operand) == const_subgraphs.end()) {
        // Operand has not been visited yet.
        unvisited.insert(operand);
      }
    }
    if (foundNonConst)
      continue;

    if (!unvisited.empty()) {
      // All operands have not been visited, push the current value, then the
      // unvisited operands (so they get handled first).
      work_queue.push_back(v);
      for (auto operand : unvisited)
        work_queue.push_back(operand);
      continue;
    }

    // All operands are const, union this value with all operands.
    for (auto operand : op->getOperands())
      const_subgraphs.unionSets(v, operand);

    // Check if users of this now constant value are themselves constant.
    for (auto result : op->getResults())
      for (auto user : result.getUsers())
        for (auto user_result : user->getResults())
          work_queue.push_back(user_result);
  }
  assert(iter < bound_for_maximum_iterations &&
         "Hit bound on maximum number of iterations for detecting constant "
         "subgraphs. Either this graph op is VERY large (in which case this "
         "bound should be increased), or else somehow there is infinite "
         "recursion (which is a bug)");

  // Collect subgraphs from equivalence sets.
  SmallVector<Subgraph> subgraphs;
  for (auto it = const_subgraphs.begin(), end = const_subgraphs.end();
       it != end; ++it) {
    if (!it->isLeader())
      continue;
    // Ignore subgraphs that only contain constant ops.
    if (llvm::all_of(
            llvm::iterator_range(const_subgraphs.member_begin(it),
                                 const_subgraphs.member_end()),
            [](Value v) { return dyn_cast<ConstantOp>(v.getDefiningOp()); }))
      continue;
    llvm::SetVector<Value> members(const_subgraphs.member_begin(it),
                                   const_subgraphs.member_end());

    subgraphs.push_back(Subgraph(members));
  }

  return subgraphs;
}

/// Copy a subgraph out of its original GraphOp and into a new GraphOp in a
/// different module. Adds OutputOps for all the outputs of the subgraph. Return
/// information to identify the new GraphOp.
SubgraphInfo cloneSubgraph(OpBuilder builder, GraphOp orig, ModuleOp module,
                           Subgraph subgraph, int64_t &name_counter) {
  auto resultTypes = llvm::map_to_vector(subgraph.outputs,
                                         [](Value v) { return v.getType(); });
  int64_t counter = orig.getProperties().getCounter();
  // This name cannot be too long otherwise something breaks :(.
  auto graphop = builder.create<GraphOp>(
      module.getLoc(),
      (orig.getName() + "_subgraph_" + std::to_string(name_counter)).str(),
      SmallVector<Type>{}, resultTypes);
  name_counter++;
  orig.getProperties().setCounter(counter + 1);
  builder.setInsertionPointToEnd(graphop.getBodyBlock());

  SmallPtrSet<Operation *, 8> ops;
  for (auto v : subgraph.vals)
    ops.insert(v.getDefiningOp());
  SmallVector<Operation *, 8> ops_sorted(ops.begin(), ops.end());
  // Topologically sort so definitions come before uses.
  mlir::computeTopologicalSorting(ops_sorted);
  IRMapping oldToNew;
  for (auto old_op : ops_sorted)
    builder.clone(*old_op, oldToNew);
  auto new_outputs = llvm::map_to_vector(
      subgraph.outputs, [&](Value v) { return oldToNew.lookup<Value>(v); });
  builder.create<OutputOp>(module.getLoc(), new_outputs,
                           KGEN::ParameterExprArrayAttr{});
  return {orig, std::string(graphop.getName()), subgraph};
}

// Models need to ensure that their backing buffer lives as long as they do.
struct ModelsWithBuffer {
  std::unique_ptr<ModelCollection> models;
  std::unique_ptr<llvm::MemoryBuffer> buffer;
};

// Interface for controlling whether or not a subgraph is constant folded.
struct ConstantFoldPolicy {
  // Determines which subgraphs should be constant folded. Must return the
  // passed in subgraphs or subsets of each of them. Inputs subgraphs not
  // included in the output will not be constant folded at all. I.e. only the
  // returned subgraphs will be constant folded.
  virtual SmallVector<Subgraph>
  shouldFold(SmallVector<Subgraph> subgraphs) const = 0;
  virtual ~ConstantFoldPolicy(){};
};

// Fold the largest subgraph that does not increase memory used by more than a
// factor of the space used by the non-folded subgraph.
struct FactorMemIncrease : public ConstantFoldPolicy {
  FactorMemIncrease(double factor) : factor(factor) {}

  // A factor of 1 indicates no increase, a factor of 2 indicates 100% more
  // memory usage max.
  double factor;

  SmallVector<Subgraph>
  shouldFold(SmallVector<Subgraph> subgraphs) const override {
    SmallVector<Subgraph> foldThese;
    for (auto &subgraph : subgraphs) {
      int64_t initialMem = constantMemUsage(subgraph);
      // Greedily remove outputs with the largest size. Might be a smarter way
      // to do this.
      while (!subgraph.vals.empty() &&
             outputMemUsage(subgraph) > initialMem * factor) {
        // TODO: this is N^2, we should improve it #18475
        Value maxValue = *std::max_element(
            subgraph.outputs.begin(), subgraph.outputs.end(),
            [](Value a, Value b) { return valueSize(a) < valueSize(b); });
        // TODO: this is N^2, we should improve it #18475
        subgraph.remove(maxValue);
      }
      if (!llvm::all_of(subgraph.outputs,
                        [](Value v) { return v.getDefiningOp<ConstantOp>(); }))
        foldThese.push_back(subgraph);
    }
    return foldThese;
  }
  ~FactorMemIncrease() override {}

  // Size of Value in bytes.
  static int64_t valueSize(Value v) {
    auto tensor = dyn_cast<TensorType>(v.getType());
    return typeToShaped(tensor).getNumElements() *
           tensor.getDtype().getSizeInBytes();
  }

  // Size of ConstantOp data in bytes.
  static int64_t valueSize(ConstantOp op) { return valueSize(op.getResult()); }

  // Calculate the size in bytes of memory used by the outputs of a subgraph.
  static int64_t outputMemUsage(Subgraph subgraph) {
    int64_t outputSize = 0;
    for (auto output : subgraph.outputs)
      outputSize += valueSize(output);
    return outputSize;
  }

  // Calculate the size in bytes of memory used by all constant ops in the
  // subgraph.
  static int64_t constantMemUsage(Subgraph subgraph) {
    int64_t internalSize = 0;
    for (auto value : subgraph.vals) {
      if (!subgraph.outputs.contains(value)) {
        auto constant = value.getDefiningOp<ConstantOp>();
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

  ConstantFoldPass() : ConstantFoldBase(), policy(new FactorMemIncrease(2)) {}

private:
  // Lower a ModuleOp all the way to MGP, export it as MEF, then load it
  // for execution. Returns the models along with a buffer that needs to
  // live as long as they do. This needs to be a member of the pass so it
  // can access runPipeline.
  ErrorOr<ModelsWithBuffer> lowerSubgraphs(ModuleOp module);

  std::shared_ptr<const ConstantFoldPolicy> policy;
};

ErrorOr<ModelsWithBuffer> ConstantFoldPass::lowerSubgraphs(ModuleOp module) {
  // Lower module to MGP.
  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);
  MO::MoToMgpOptions moOpts;
  // Avoid infinite recursion.
  moOpts.constantFold = false;
  // Use MOF executor so we don't depend on a framework.
  moOpts.compiledFrameworkLabel = "mof";
  MO::buildMoToMgpPipeline(pm, moOpts);
  // Run pipeline needs to run on a module.
  if (failed(runPipeline(pm, module)))
    return Error("Failed to run MOToMOGG pipeline");

  // Convert to MEF. We write it to a file and then read it back in. LLVM
  // doesn't seem to provide a way to write and read it directly from a buffer.
  llvm::SmallString<128> filename;
  std::error_code e =
      llvm::sys::fs::createTemporaryFile("mo", "subgraph_mef", filename);
  if (e.value() != 0)
    return Error("Cannot open temp file: " + e.message());
  {
    std::error_code e;
    llvm::raw_fd_ostream os(filename, e);
    if (failed(convertToMEF(module, os, false)))
      return Error("unable to convert file to a MEF binary");
  }
  // Delete file when we leave scope.
  llvm::FileRemover remover(filename);

  std::string errorMsg;
  std::unique_ptr<llvm::MemoryBuffer> file =
      mlir::openInputFile(filename, llvm::Align(0x4000), &errorMsg);
  if (!file)
    return Error(errorMsg);
  UNWRAP_ERROR(mefFile, MEFFile::open(file->getBuffer()));
  UNWRAP_ERROR(models, loadMGPMEFFile(std::move(mefFile)));
  return {ModelsWithBuffer{std::move(models), std::move(file)}};
}

void ConstantFoldPass::runOnOperation() {
  // At a high level, this pass:
  // 1. Extracts all constant subgraphs.
  // 2. Clones each subgraph into its own GraphOp.
  // 3. Lowers and executes each GraphOp from the previous step.
  // 4. Replaces subgraph outputs with outputs from the previous step.

  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(getOperation().getBody());
  // We have to create the copy of each subgraph somewhere under the original
  // model because passes can't create ops outside of what they are operating
  // on. We create a nested module in the current module where we do everything.
  // At the end we remove it.
  auto module = builder.create<ModuleOp>(getOperation().getLoc());

  // For each constant subgraph, we clone it into our submodule as a new
  // GraphOp. The outputs of the GraphOp are the outputs of the subgraph.
  //
  // An alternative approach could be to combine each subgraph into one large
  // constant graph so we only have one thing to execute.
  SmallVector<SubgraphInfo> subgraphReplacements;
  int64_t name_counter = 0;
  getOperation().walk([&](GraphOp graph) {
    auto subgraphs = ConstantSubgraphs(graph);
    subgraphs = policy->shouldFold(subgraphs);
    for (auto subgraph : subgraphs) {
      builder.setInsertionPointToEnd(module.getBody());
      subgraphReplacements.push_back(
          cloneSubgraph(builder, graph, module, subgraph, name_counter));
    }
  });

  ErrorOr<ModelsWithBuffer> modelsOr = lowerSubgraphs(module);
  if (failed(modelsOr)) {
    mlir::emitError(module.getLoc(),
                    "unable to load models: " + Twine(modelsOr.getError()));
    return signalPassFailure();
  }

  // Execute MEF one subgraph at a time.
  // TODO: #18077 this should be an argument to the pass.
  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());
  for (auto [i, subgraphInfo] : llvm::enumerate(subgraphReplacements)) {
    const Model *model =
        modelsOr.get().models->getModelNamed(subgraphInfo.constantSubgraphName);
    assert(model != nullptr &&
           "Model should exist, make sure names above match.");
    ErrorOr<llvm::DenseMap<Value, ElementsAttr>> valuesToTensorsOr =
        executeModel(ctx, model, subgraphInfo, runtime);
    if (failed(valuesToTensorsOr)) {
      mlir::emitError(module.getLoc(),
                      Twine(valuesToTensorsOr.takeError().get()));
      return signalPassFailure();
    }

    // Replace output expressions with constants. Also handles DCE.
    RewritePatternSet patterns{ctx};
    patterns.add<RewriteConstantOps>(ctx, valuesToTensorsOr.get());
    if (failed(applyPatternsAndFoldGreedily(subgraphInfo.orig,
                                            std::move(patterns))))
      return signalPassFailure();
  }

  // Remove module to not clutter the output.
  module.erase();
}

} // namespace M::MO
