//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstantFold.h"
#include "GenericML/Driver/CXXDriverAPI.h"
#include "GenericML/Driver/CompileDriver.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/MGPDialect/MGPDialect.h"
#include "GenericML/Support/Model.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
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

struct ValueParamCompare {
  bool operator()(const Operation *lhs, const Operation *rhs) const {
    return hash_value(lhs) < hash_value(rhs);
  }
};

llvm::SetVector<StringAttr> Subgraph::paramDeclarations() const {
  llvm::SetVector<StringAttr> params;
  for (auto op : ops) {
    op->walk([&](KGEN::ParamOpInterface paramOp) {
      paramOp.walkDeclarations(
          [&](KGEN::ParamDeclAttr decl) { params.insert(decl.getName()); });
      paramOp.walkDefinitions(
          [&](KGEN::ParamDeclAttr decl, KGEN::ParamDefValue val) {
            params.insert(decl.getName());
          });
    });
  }
  return params;
}

/// Topologically sort parameters in a list of ops.
llvm::SmallVector<Operation *> sortParams(llvm::ArrayRef<Operation *> ops) {
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
          [&](KGEN::ParamDeclAttr decl, KGEN::ParamDefValue val) {
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
  RewriteConstantOps(
      MLIRContext *context,
      const llvm::DenseMap<Value, std::pair<ElementsAttr, TensorShape>>
          &constants,
      const llvm::SetVector<Operation *> &constantOps)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context), constants(constants),
        constantOps(constantOps) {}
  using RewritePattern::matchAndRewrite;
  const llvm::DenseMap<Value, std::pair<ElementsAttr, TensorShape>> &constants;
  const llvm::SetVector<Operation *> &constantOps;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() > 0) {
      auto it = constants.find(op->getResult(0));
      if (it != constants.end()) {
        rewriter.setInsertionPoint(op);
        llvm::SmallVector<Value> new_results;
        for (auto r : op->getResults()) {
          // Each result can either be a tensor or a shape.
          if (r.getType().isa<TensorType>()) {
            auto [constant, shape] = constants.at(r);
            Type type = r.getType();
            // Resolve instances where the output shape is not statically
            // known (mo.tensor<[?], f32> for example) by using the size of
            // the output.
            if (dyn_cast<TensorType>(type).hasUnknownShape()) {
              auto resolvedType = TensorType::get(
                  MOSH::ShapeAttr::get(
                      SmallVector<int64_t>(shape.begin(), shape.end()),
                      MOSH::ShapeType::get(op->getContext())),
                  dyn_cast<TensorType>(r.getType()).getDtype());
              auto constantOp = rewriter.create<ConstantOp>(
                  op->getLoc(), resolvedType, constant);
              // Rebind the shape so it matches what is expected. For example,
              // transform [1, 2] to [1, A] so that types shapes with parameters
              // match.
              new_results.push_back(
                  rewriter.create<RebindOp>(op->getLoc(), type, constantOp));
            } else {
              new_results.push_back(
                  rewriter.create<ConstantOp>(op->getLoc(), type, constant));
            }
          } else if (r.getType().isa<ShapeType>()) {
            auto intElems =
                dyn_cast<IntArrayElementsAttr>(constants.at(r).first);
            new_results.push_back(rewriter.create<KGEN::ParamConstantOp>(
                op->getLoc(),
                ShapeAttr::get(intElems.asArrayRef<int64_t>(),
                               MOSH::ShapeType::get(op->getContext()))));
          } else {
            assert(false && "constants must be tensors or shapes");
          }
        }
        rewriter.replaceOp(op, new_results);
        return LogicalResult::success();
      }
    }
    // Match any op that was not replaced with a constant to check if it should
    // be DCEd.
    if (constantOps.contains(op)) {
      if (llvm::all_of(op->getUsers(), [&](Operation *user) {
            return constantOps.contains(user) ||
                   llvm::all_of(user->getResults(), [&](Value result) {
                     return constants.contains(result);
                   });
          }))
        rewriter.eraseOp(op);
      return LogicalResult::success();
    }
    return rewriter.notifyMatchFailure(op, "Op is not constant");
  }
};

/// Simple struct holding results of a model execution.
struct ModelResult {
  /// Mapping from subgraph output value to the constants that should replace
  /// it. Also includes the output shape so that unknown shapes in the constant
  /// graph can be resolved.
  llvm::DenseMap<Value, std::pair<ElementsAttr, TensorShape>> constants;
  /// Mapping from parameter name in subgraph to the value that it should be
  /// defined as.
  llvm::DenseMap<StringAttr, int64_t> parameters;
};

/// Run a Model to get the constant outputs to a subgraph. Returns a mapping
/// from output Vale to ElementsAttr containing the constant resulting from
/// folding.
ErrorOr<ModelResult> executeModel(MLIRContext *ctx, const Model *model,
                                  SubgraphInfo subgraphInfo,
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

  const AnyAsyncValueRef &ch = resultCopy.front();
  assert(ch.isReady() && isa<Chain>(ch) &&
         "first result must be a ready chain");
  resultCopy = resultCopy.drop_front();
  llvm::SmallVector<AnyAsyncValueRef> outputTensorRefs;
  outputTensorRefs.reserve(results.size());
  for (auto &result : resultCopy) {
    outputTensorRefs.push_back(
        executableModel->convertToGenericTensor(std::move(result)));
  }
  LLCL::await(outputTensorRefs);

  // Get the constant values.
  llvm::DenseMap<Value, std::pair<ElementsAttr, TensorShape>> valuesToTensors;
  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);
  size_t outIdx = 0;
  for (auto op : subgraphInfo.subgraph.outputs) {
    for (auto result : op->getResults()) {
      const auto &tensor = outputTensorRefs[outIdx];
      if (tensor.isError())
        return Error(tensor.getDiagnostic().getMessage().get());

      // Copy output tensor into storage.
      assert(tensor.isReady() && "We already awaited, tensor should be ready");
      auto *t = dyn_cast<Tensor>(tensor);
      assert(t && "Output from subgraph execution should be a tensor");
      // Output shape has runtime-dependent size so we need to read out what
      // the size is.
      TensorShape shape = t->getShape();
      int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                     std::multiplies<int64_t>());
      ArrayType arrayType = ArrayType::get(
          op->getContext(), size, toMLIRType(ctx, t->getEltType()).get());
      valuesToTensors[result] = {
          getAttrForTensorDataCopy(
              arrayType,
              subgraphInfo.constantSubgraphName + "_" + std::to_string(outIdx),
              {reinterpret_cast<const char *>(t->getBuffer()),
               t->getSizeInBytes()},
              resourceManager),
          shape};
      outIdx++;
    }
  }

  // Get parameters now. They come after the constants.
  llvm::DenseMap<StringAttr, int64_t> params;
  for (auto paramName : subgraphInfo.subgraph.paramDeclarations()) {
    const auto &tensor = outputTensorRefs[outIdx];
    if (tensor.isError())
      return Error(tensor.getDiagnostic().getMessage().get());
    assert(tensor.isReady() && "We already awaited, tensor should be ready");
    auto *t = dyn_cast<Tensor>(tensor);
    assert(t && "Output from subgraph execution should be a tensor");
    params[paramName] = reinterpret_cast<const int64_t *>(t->getBuffer())[0];
    outIdx++;
  }
  return ModelResult{valuesToTensors, params};
}

/// The use-def graph has both declarations (no associated value) and
/// definitions (x = y). We just need any parameter that is created so we search
/// through both for the defining operation.
Operation *lookupParamDef(const KGEN::ParameterUseDefGraph &paramUseDef,
                          StringAttr name) {
  Operation *def;
  auto defIt = paramUseDef.defs.find(name);
  auto declIt = paramUseDef.decls.find(name);
  if (defIt != paramUseDef.defs.end())
    def = defIt->second.defOp;
  else if (declIt != paramUseDef.decls.end())
    def = declIt->second.declOp;
  else
    assert(false && "Could not find def!");
  return def;
}

SmallVector<Subgraph>
constantSubgraphs(MO::GraphOp graph,
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

    // Skip yield, return, etc.
    if (dyn_cast<MOControlOpInterface>(op)) {
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
          [&](KGEN::ParamDeclAttr param, ParamDefValue val) {
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
  assert(iter < bound_for_maximum_iterations &&
         "Hit bound on maximum number of iterations for detecting constant "
         "subgraphs. Either this graph op is VERY large (in which case this "
         "bound should be increased), or else somehow there is infinite "
         "recursion (which is a bug)");

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
SubgraphInfo cloneSubgraph(OpBuilder builder, GraphOp orig, ModuleOp module,
                           Subgraph subgraph, int64_t &name_counter) {
  llvm::SmallVector<Type> resultTypes;
  for (auto op : subgraph.outputs)
    for (auto result : op->getResults())
      resultTypes.push_back(result.getType());
  int64_t counter = orig.getProperties().getCounter();
  // This name cannot be too long otherwise something breaks :(.
  auto graphop = builder.create<GraphOp>(
      module.getLoc(),
      (orig.getName() + "_subgraph_" + std::to_string(name_counter)).str(),
      SmallVector<Type>{}, resultTypes);
  name_counter++;
  orig.getProperties().setCounter(counter + 1);
  builder.setInsertionPointToEnd(graphop.getBodyBlock());

  SmallVector<Operation *, 8> opsSorted(subgraph.ops.begin(),
                                        subgraph.ops.end());
  // Topologically sort so definitions come before uses.
  // Sort param first, doesn't necessarily maintain topological order.
  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();
  opsSorted = sortParams(opsSorted);
  mlir::computeTopologicalSorting(opsSorted);
  IRMapping oldToNew;
  for (auto old_op : opsSorted)
    builder.clone(*old_op, oldToNew);
  llvm::SmallVector<Value> newOutputs;
  llvm::SetVector<KGEN::ParamDeclAttr> outputParams;
  for (auto op : subgraph.outputs) {
    for (auto result : op->getResults()) {
      Value newOutput = oldToNew.lookup<Value>(result);
      if (newOutput.getType().isa<MOSH::ShapeType>()) {
        // Can't output shapes so we convert it to a tensor of unknown size.
        newOutput = builder.create<Shape::ToTensorOp>(
            loc,
            TensorType::get({KGEN::UnknownAttr::get(newOutput.getType())},
                            DType::si64),
            newOutput);
      }
      newOutput.getType().walk([&](KGEN::ParamDeclRefAttr ref) {
        outputParams.insert(KGEN::ParamDeclAttr::get(
            ref.getContext(), ref.getName(), ref.getType()));
      });
      newOutputs.push_back(newOutput);
    }
  }

  // Add parameter declaration/definitions that are used in the subgraph. They
  // are converted to indexes and then to tensors so that we can output them.
  Type indexType = IndexType::get(ctx);
  for (auto param : subgraph.paramDeclarations()) {
    Value index = builder.create<KGEN::ParamConstantOp>(
        loc, indexType, KGEN::ParamDeclRefAttr::get(ctx, param, indexType));
    newOutputs.push_back(builder.create<IndexToTensorOp>(
        loc,
        TensorType::get(ArrayRef<int64_t>({}),
                        IntegerType::get(ctx, 64, IntegerType::Unsigned)),
        index));
  }
  // No output parameters because we bind them all locally.
  auto out =
      builder.create<OutputOp>(loc, newOutputs, KGEN::ParameterExprArrayAttr{});
  // Have to update the GraphOp's signature.
  graphop.setSignature(SignatureType::get(ctx, {}, out.getOperandTypes()));
  graphop.setFunctionType(
      mlir::FunctionType::get(ctx, {}, out.getOperandTypes()));
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
  // included in the output will not be constant folded at all. I.e. only
  // the returned subgraphs will be constant folded.
  virtual SmallVector<Subgraph>
  shouldFold(SmallVector<Subgraph> subgraphs) const = 0;
  virtual ~ConstantFoldPolicy(){};
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
      if (!llvm::all_of(subgraph.outputs,
                        [](Operation *op) { return dyn_cast<ConstantOp>(op); }))
        foldThese.push_back(subgraph);
    }
    return foldThese;
  }
  ~FactorMemIncrease() override {}

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
  static int64_t outputMemUsage(Subgraph subgraph) {
    int64_t outputSize = 0;
    for (auto output : subgraph.outputs)
      outputSize += valueSize(subgraph, output);
    return outputSize;
  }

  // Calculate the size in bytes of memory used by all constant ops in the
  // subgraph.
  static int64_t constantMemUsage(Subgraph subgraph) {
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
    return Error("Failed to run MOToMGP pipeline");

  // Convert to MEF. We write it to a file and then read it back in. LLVM
  // doesn't seem to provide a way to write and read it directly from a
  // buffer.
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
  auto mefFileOr = MEFFile::open(file->getBuffer());
  if (mefFileOr.isError())
    return mefFileOr.takeError();
  auto modelsOr = loadMGPMEFFile(std::move(*mefFileOr));
  if (modelsOr.isError())
    return modelsOr.takeError();
  return {ModelsWithBuffer{std::move(*modelsOr), std::move(file)}};
}

void ConstantFoldPass::runOnOperation() {
  // At a high level, this pass:
  // 1. Extracts all constant subgraphs.
  // 2. Clones each subgraph into its own GraphOp.
  // 3. Lowers and executes each GraphOp from the previous step.
  // 4. Replaces subgraph outputs with outputs from the previous step.

  MLIRContext *ctx = &getContext();
  // Used for param use-def graph.
  auto &paramCache = getAnalysis<KGEN::ParameterCollector::Analysis>();

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
    SmallVector<SubgraphInfo> subgraphReplacements;
    int64_t name_counter = 0;
    getOperation().walk([&](GraphOp graph) {
      auto subgraphs = constantSubgraphs(graph, paramCache);
      if (subgraphs.empty())
        return mlir::WalkResult::advance();
      subgraphs = policy->shouldFold(subgraphs);
      for (auto subgraph : subgraphs) {
        builder.setInsertionPointToEnd(module.getBody());
        subgraphReplacements.push_back(
            cloneSubgraph(builder, graph, module, subgraph, name_counter));
      }
      return WalkResult::advance();
    });

    if (module.getBody()->getOperations().empty()) {
      // Remove module to not clutter the output.
      module.erase();
      return;
    }

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

    // We have to consolidate rewrites to the same graphop as the rewrites of
    // one subgraph may invalidate ops in another subgraph (due to fold hooks).
    llvm::DenseMap<GraphOp,
                   llvm::SmallVector<std::pair<ModelResult, SubgraphInfo>>>
        executionResults;
    for (size_t i = 0; i < subgraphReplacements.size(); i++) {
      SubgraphInfo subgraphInfo = subgraphReplacements[i];
      const Model *model = modelsOr.get().models->getModelNamed(
          subgraphInfo.constantSubgraphName);
      assert(model != nullptr &&
             "Model should exist, make sure names above match.");
      ErrorOr<ModelResult> valuesToTensorsOr =
          executeModel(ctx, model, subgraphInfo, runtime);
      if (failed(valuesToTensorsOr)) {
        mlir::emitError(module.getLoc(),
                        Twine(valuesToTensorsOr.takeError().get()));
        return signalPassFailure();
      }
      auto it = executionResults.find(subgraphInfo.orig);
      if (it != executionResults.end()) {
        it->second.push_back({valuesToTensorsOr.takeValue(), subgraphInfo});
      } else {
        executionResults[subgraphInfo.orig] = {
            {valuesToTensorsOr.takeValue(), subgraphInfo}};
      }
    }

    // Replace output expressions with constants. Also handles DCE.
    for (auto [graph, results] : executionResults) {
      llvm::SetVector<Operation *> internalOps;
      for (auto r : results) {
        SubgraphInfo &subgraphInfo = r.second;
        auto filtered = llvm::make_filter_range(
            subgraphInfo.subgraph.ops, [&](Operation *op) {
              return llvm::all_of(op->getResults(), [&](OpResult r) {
                return subgraphInfo.subgraph.fullyContains(r);
              });
            });
        internalOps.insert(filtered.begin(), filtered.end());
      }

      llvm::DenseMap<Value, std::pair<ElementsAttr, TensorShape>> constants;
      llvm::DenseMap<StringAttr, int64_t> params;
      for (auto r : results) {
        constants.insert(r.first.constants.begin(), r.first.constants.end());
        params.insert(r.first.parameters.begin(), r.first.parameters.end());
      }

      RewritePatternSet patterns{ctx};
      patterns.add<RewriteConstantOps>(ctx, constants, internalOps);
      if (failed(applyPatternsAndFoldGreedily(graph, std::move(patterns)))) {
        mlir::emitError(module.getLoc(),
                        "Failed to rewrite ops into their constant values.");
        return signalPassFailure();
      }

      // Add parameters that were defined in the subgraph.
      OpBuilder builder(ctx);
      builder.setInsertionPointToStart(graph.getBodyBlock());
      for (auto [name, val] : params) {
        builder.create<KGEN::ParamDeclareOp>(
            graph.getLoc(),
            KGEN::ParamDeclAttr::get(ctx, name, IndexType::get(ctx)),
            builder.getIndexAttr(val));
      }
    }

    // Remove module to not clutter the output.
    module.erase();
  }
}

} // namespace M::MO
