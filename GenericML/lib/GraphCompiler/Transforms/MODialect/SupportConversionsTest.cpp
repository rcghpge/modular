//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOSupport/Activations.h"
#include "GenericML/GraphCompiler/Conversion/MOTarget/Conversion.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;

class ConvertPreluPattern : public ConversionPattern {
public:
  ConvertPreluPattern(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, "prelu", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto x = cast<MO::TensorValue>(operands[0]);
    auto alpha = cast<MO::TensorValue>(operands[1]);
    auto result = convertPRelu(rewriter, op->getLoc(), x, alpha);
    rewriter.replaceOp(op, result);
    return success();
  }
};

namespace M::MO {

#define GEN_PASS_DEF_SUPPORTCONVERSIONSTEST
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

class SupportConversionTestPass
    : public impl::SupportConversionsTestBase<SupportConversionTestPass> {
  using SupportConversionsTestBase::SupportConversionsTestBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    TypeConverter converter;
    converter.addConversion([](TensorType t) { return t; });
    RewritePatternSet patterns{ctx};
    patterns.add<ConvertPreluPattern>(converter, ctx);
    MO::MOConversionTarget target(*ctx);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  };
};

} // namespace M::MO
