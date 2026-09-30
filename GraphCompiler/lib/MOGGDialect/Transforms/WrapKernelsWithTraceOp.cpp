//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_WRAPKERNELSWITHTRACEOP
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

struct WrapKernelsWithTraceOp
    : public MOGG::impl::WrapKernelsWithTraceOpBase<WrapKernelsWithTraceOp> {
  using WrapKernelsWithTraceOpBase::WrapKernelsWithTraceOpBase;

  void runOnOperation() override {
    MOGG::KernelStubOp stub = getOperation();
    // A MOGG-ABI stub is wrapped by `createKernelStub`, an MGP-ABI one by
    // `AddProfileScopeTracing`.
    if (stub.getAbi() != MOGG::ABIKind::MOGG)
      return;

    Block &body = stub.getBodyBlock();
    Operation *terminator = body.getTerminator();

    // Only the dispatch goes inside, as on the MOGG ABI: setup ops stay out so
    // passes that rebuild the dispatch from outside -- device-graph synthesis
    // erases the whole trace -- still reach the values they feed it.
    // Collect first, or the walk would move the new op into its own region.
    SmallVector<Operation *> traced;
    bool seenCall = false;
    for (Operation &op : body) {
      if (&op == terminator || isa<MOGG::WithTraceOp>(op))
        continue;
      seenCall |=
          isa<MOGG::CallOpaqueOp, MOGG::CallForeachOp, MOGG::CallShapeOp>(op);
      if (seenCall)
        traced.push_back(&op);
    }
    if (traced.empty())
      return;

    OpBuilder builder(terminator);
    Location loc = stub.getLoc();
    // The terminator stays outside and reads the results, so a shape function
    // keeps its `return`. `AddTryExceptBlocks` forwards through `mogg.try` the
    // same way, which is what lets the two nest.
    auto withTrace =
        MOGG::WithTraceOp::create(builder, loc, terminator->getOperandTypes());
    Block &tracedBlock = withTrace.getBody().emplaceBlock();
    for (Operation *op : traced)
      op->moveBefore(&tracedBlock, tracedBlock.end());

    builder.setInsertionPointToEnd(&tracedBlock);
    MOGG::YieldOp::create(builder, loc, terminator->getOperands());
    terminator->setOperands(withTrace.getOutputs());
  }
};

} // namespace

} // namespace M::MOGG
