//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace M::MO {

#define GEN_PASS_DEF_FLATTENELEMENTWISE
#include "GenericML/GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

template <typename ElemwiseInterfaceT>
bool isElemwiseOpSimple(ElemwiseInterfaceT elemwise) {
  // abort if the elemwise op cannot have its resulting tensor type changed
  // trivially.
  if (isa<MO::RangeOp, MO::UnsqueezeShapeOp>(elemwise))
    return false;

  return true;
}

bool isSimpleOp(Operation *op) {
  if (auto unary = dyn_cast<ElementWiseUnary>(op))
    return isElemwiseOpSimple(unary);
  else if (auto binary = dyn_cast<ElementWiseBinary>(op))
    return isElemwiseOpSimple(binary);
  else if (isa<MO::Static::ReshapeOp>(op))
    return true;
  return false;
}

template <typename ElemwiseInterfaceT>
void flattenElemwiseOp(ElemwiseInterfaceT elemwise, OpBuilder &builder) {
  TypedValue<TensorType> res = elemwise.getResult();
  TensorType resTt = res.getType();

  // abort if this elementwise op is already flattened
  if (resTt.getRank() <= 1)
    return;

  // compute shape of new flattened tensor
  MOSH::ShapeAttr shape = resTt.getTypedShapeAttr();
  TypedAttr flattenedSize = shape.getSymbolicNumElements();
  MOSH::ShapeAttr newShape =
      MOSH::ShapeAttr::get({flattenedSize}, shape.getType());

  // add reshapes for each operand. unflattened -> flattened
  for (auto [i, operand] : llvm::enumerate(elemwise->getOperands())) {
    // this may differ from resTt if the elementwise op is a cast / pow
    // furthermore, the dtype of each operand may be different for pow
    TensorType inpTt = cast<TensorType>(operand.getType());
    TensorType newInpTt = TensorType::get(
        newShape, inpTt.getDtype(), inpTt.getDeviceRef(), inpTt.getMetadata());

    builder.setInsertionPoint(elemwise);
    auto newReshapeOp = MO::Static::ReshapeOp::create(
        builder, elemwise.getLoc(), newInpTt, operand);

    elemwise->setOperand(i, newReshapeOp.getResult());
  }

  // add reshape for result. flattened -> unflattened
  builder.setInsertionPointAfter(elemwise);
  auto newReshapeOp =
      MO::Static::ReshapeOp::create(builder, elemwise.getLoc(), resTt, res);
  res.replaceAllUsesExcept(newReshapeOp.getResult(), {newReshapeOp});

  // update elementwise op to use flattened shape
  TensorType newResTt = TensorType::get(
      newShape, resTt.getDtype(), resTt.getDeviceRef(), resTt.getMetadata());
  res.setType(newResTt);
}

/// Collect all operands and users of a given op that reside in same block
void getNeighbors(SmallVector<Operation *> &neighbors, Operation *u) {
  neighbors.clear();
  for (auto operands : u->getOperands())
    if (auto v = operands.getDefiningOp())
      if (v->getBlock() == u->getBlock())
        neighbors.push_back(v);
  for (auto v : u->getUsers())
    if (v->getBlock() == u->getBlock())
      neighbors.push_back(v);
}

/// Collect all simpleOps reachable from a given simpleOp
llvm::SmallDenseSet<Operation *> collectSubgraph(Operation *simpleOp,
                                                 Block *block) {

  llvm::SmallDenseSet<Operation *> subgraph;
  SmallVector<Operation *> stack{simpleOp};
  llvm::SmallDenseSet<Operation *> previouslyAddedToStack{simpleOp};
  SmallVector<Operation *> neighbors;

  // DFS traversal
  while (!stack.empty()) {
    Operation *u = stack.pop_back_val();

    if (!subgraph.contains(u) && isSimpleOp(u)) {
      subgraph.insert(u);

      // add unexplored neighbors v to frontier
      getNeighbors(neighbors, u);
      for (auto v : neighbors) {
        if (!previouslyAddedToStack.contains(v)) {
          previouslyAddedToStack.insert(v);
          stack.push_back(v);
        }
      }
    }
  }

  return subgraph;
}

template <typename ElemwiseInterfaceT>
void handleElemwiseOp(ElemwiseInterfaceT elemwise, OpBuilder &builder,
                      llvm::SmallDenseSet<Operation *> &visitedOps) {
  // if we've already visited this op, skip it
  if (visitedOps.contains(elemwise))
    return;

  // collect all ops in subgraph
  llvm::SmallDenseSet<Operation *> subgraph =
      collectSubgraph(elemwise, elemwise->getBlock());

  // count number of reshapes in subgraph
  auto numReshapes = 0;
  for (auto op : subgraph)
    if (isa<MO::Static::ReshapeOp>(op))
      ++numReshapes;

  // if there are no reshapes in the subgraph, flattening the elementwise ops
  // is unnecessary and will not unblock additional elementwise fusions
  if (numReshapes == 0)
    return;

  // flatten all elementwise ops in subgraph
  for (auto op : subgraph)
    if (auto unary = dyn_cast<ElementWiseUnary>(op))
      flattenElemwiseOp(unary, builder);
    else if (auto binary = dyn_cast<ElementWiseBinary>(op))
      flattenElemwiseOp(binary, builder);

  // mark all ops in subgraph as visited
  for (auto op : subgraph)
    visitedOps.insert(op);
}

/// This pass finds all elementwise ops and optionally flattens them if it could
/// possibly yield additional elementwise fusions. This is because reshapes
/// between elementwise ops block elementwise fusion. By making all of the
/// elementwise ops in a connected subgraph of elementwise + reshape ops the
/// same shape, we will eliminate all reshapes internal to that subgraph.
///
/// To identify subgraphs, we run a DFS along users and operands. For subgraphs
/// with only elementwise ops and no reshape ops, we skip flattening as there
/// will be no benefit since no reshapes are blocking fusion.
///
/// Note that this pass may insert additional reshapes on the boundaries between
/// elementwise and non-elementwise ops. In most cases, this does not block
/// fusion between the elementwise + non-elementwise op. This fusion is achieved
/// via view decomposition + affine view fusion.
///
/// Additionally, this pass may insert many extraneous reshape ops that will
/// need to be removed via canonicalization + cse passes.
template <typename ElemwiseInterfaceT>
void maybeFlattenElemwiseOp(ElemwiseInterfaceT elemwise, OpBuilder &builder,
                            llvm::SmallDenseSet<Operation *> &visitedOps) {}
struct FlattenElementwise
    : public impl::FlattenElementwiseBase<FlattenElementwise> {
  using FlattenElementwiseBase::FlattenElementwiseBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module->getContext();

    OpBuilder builder{ctx};
    llvm::SmallDenseSet<Operation *> visitedOps;
    module->walk([&](ElementWiseUnary elemwise) {
      handleElemwiseOp(elemwise, builder, visitedOps);
    });
    module->walk([&](ElementWiseBinary elemwise) {
      handleElemwiseOp(elemwise, builder, visitedOps);
    });
  }
};

} // namespace M::MO
