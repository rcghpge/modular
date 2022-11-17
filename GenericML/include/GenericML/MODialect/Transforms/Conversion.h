//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_TRANSFORMS_CONVERSION_H
#define GENERICML_MODIALECT_TRANSFORMS_CONVERSION_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M::MO {

/// Conversion pattern to use when converting containers for graph-like models.
template <typename SourceOp, typename TargetOp>
class GenericGraphOpConversion final : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fnType = op.getFunctionType();

    TypeConverter::SignatureConversion signatureConverter(
        fnType.getNumInputs());
    auto *typeConverter = this->getTypeConverter();
    for (const auto &argType : enumerate(fnType.getInputs())) {
      auto convertedType = typeConverter->convertType(argType.value());
      if (!convertedType) {
        return rewriter.notifyMatchFailure(op,
                                           "argument type cannot be converted");
      }
      signatureConverter.addInputs(argType.index(), convertedType);
    }

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(fnType.getResults(), resultTypes))) {
      return rewriter.notifyMatchFailure(op,
                                         "result types cannot be converted");
    }

    auto newOp = rewriter.create<TargetOp>(
        op.getLoc(), op.getName(), signatureConverter.getConvertedTypes(),
        resultTypes);

    // The builder should automatically add a body, so we remove it, and replace
    // it with the body of the source op.
    newOp.eraseBody();
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());

    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *typeConverter,
                                           &signatureConverter))) {
      return rewriter.notifyMatchFailure(
          op, "block argument types cannot be converted");
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion pattern to use when converting terminators of graph-like
/// containers.
template <typename SourceOp, typename TargetOp>
class GenericOutputOpConversion final : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getOperands());
    return success();
  }
};

/// ConversionTarget for converting to the MO dialect.
class MOConversionTarget : public ConversionTarget {
public:
  explicit MOConversionTarget(MLIRContext &ctx);
};

} // namespace M::MO

#endif // GENERICML_MODIALECT_TRANSFORMS_CONVERSION_H
