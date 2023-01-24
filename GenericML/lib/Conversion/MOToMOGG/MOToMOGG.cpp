//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Converts MO ops to MGP and BEF ops.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Conversion/MOToMOGG.h"
#include "GenericML/MODialect/MOOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMOGG
#include "GenericML/Conversion/MOToMOGG.h.inc"

} // namespace M::MO

namespace {

enum class OpType {
  /* class of MOGG op this kernel represents */
  kExternal,
  kKGEN,
};

struct KernelRecord {
  // TODO: Figure out what fields this needs and auto populate
  // from lit.
  OpType type;
  StringLiteral opName;
};

static ErrorOr<ValueRange> selectMOGGOp(Operation *op) {
  // TODO: Turn this into a registry we can access from lit
  const static llvm::StringMap<SmallVector<KernelRecord, 5>> kernels = {
      {"abs", {{OpType::kExternal, "mop.kernels.abs"}}},
      {"add", {{OpType::kExternal, "mop.kernels.add"}}},
      {"broadcast_shape", {{OpType::kExternal, "mop.kernels.broadcast_shape"}}},
      {"broadcast_to", {{OpType::kExternal, "mop.kernels.broadcast_to"}}},
      {"div", {{OpType::kExternal, "mop.kernels.div"}}},
      {"erf", {{OpType::kExternal, "mop.kernels.erf"}}},
      {"exp", {{OpType::kExternal, "mop.kernels.exp"}}},
      {"gather", {{OpType::kExternal, "mop.kernels.gather"}}},
      {"matmul", {{OpType::kExternal, "mop.kernels.matmul"}}},
      {"mul", {{OpType::kExternal, "mop.kernels.mul"}}},
      {"relu", {{OpType::kExternal, "mop.kernels.relu"}}},
      {"sigmoid", {{OpType::kExternal, "mop.kernels.sigmoid"}}},
      {"shape_of", {{OpType::kExternal, "mop.kernels.shape_of"}}},
      {"softmax", {{OpType::kExternal, "mop.kernels.softmax"}}},
      {"sub", {{OpType::kExternal, "mop.kernels.sub"}}},
      {"tanh", {{OpType::kExternal, "mop.kernels.tanh"}}},
  };

  std::string opName = op->getName().stripDialect().str();

  OpBuilder b(op);
  if (kernels.count(opName)) {
    for (auto kernel : kernels.lookup(opName)) {
      // TODO: Put kernel selection metrics in here.
      if (kernel.type == OpType::kExternal) {
        // Create the mogg_ext_op
        b.setInsertionPoint(op);
        return b
            .create<MOGG::ExtOp>(op->getLoc(),
                                 /*outputTypes=*/op->getResultTypes(),
                                 /*op_name=*/kernel.opName,
                                 /*inputOperands=*/op->getOperands())
            .getResults();
      }
    }
  }
  return Error("Failed to find a kernel for " + opName);
}

struct MOOpPattern : public RewritePattern {
  MOOpPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // skip non-mo ops and container/constant/output mo ops
    if (!isa<MO::MODialect>(op->getDialect()) || isa<MO::GraphOp>(op) ||
        isa<MO::ConstantOp>(op) || isa<MO::OutputOp>(op))
      return failure();

    auto results = selectMOGGOp(op);
    if (results.isError()) {
      return failure();
    } else {
      rewriter.replaceOp(op, results.takeValue());
      return success();
    }
  }
};

struct MOToMOGGPass : public M::MO::impl::MOToMOGGBase<MOToMOGGPass> {
  using MOToMOGGBase::MOToMOGGBase;

  void runOnOperation() override;
};

} // namespace

void MOToMOGGPass::runOnOperation() {
  auto *ctx = &getContext();

  RewritePatternSet patterns{ctx};
  patterns.add<MOOpPattern>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}
