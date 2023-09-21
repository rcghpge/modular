//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Support for working with device-dependent buffer and tensor values and
// types.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_BUFFERBUILDER_H
#define GENERICML_MGPDIALECT_BUFFERBUILDER_H

#include "GenericML/MGPDialect/MGPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// Generic parsing/printing helpers
//===----------------------------------------------------------------------===//

/// Parses one off:
///   ':' resultRawType
///   bufferRawType ':' resultRawType
/// In the first form the bufferRawType is derived from the resultRawType.
template <typename BufferType, typename TensorType>
ParseResult genericParseMayBeDynamicBuffer(AsmParser &parser,
                                           Type &bufferRawType,
                                           Type &resultRawType) {
  mlir::SMLoc loc;
  if (parser.getCurrentLocation(&loc))
    return failure();
  OptionalParseResult parseResult = parser.parseOptionalType(bufferRawType);
  bool hasBufferTypeAnnotation = parseResult.has_value();
  if (hasBufferTypeAnnotation) {
    if (failed(parseResult.value()))
      return failure();
    if (!isa<BufferType>(bufferRawType))
      return parser.emitError(loc)
             << "expecting !" << BufferType::getMnemonic() << " type";
  }

  if (parser.parseColon() || parser.parseType(resultRawType))
    return failure();
  auto resultTensorType = dyn_cast<TensorType>(resultRawType);
  if (resultTensorType == nullptr)
    return parser.emitError(loc)
           << "expecting !" << TensorType::getMnemonic() << " type";
  if (!hasBufferTypeAnnotation) {
    // No buffer type annotation -- buffer operand's type is uniquely determined
    // from the result tensor type.
    bufferRawType = BufferType::get(resultTensorType);
  }
  return success();
}

/// Prints one off:
///   ':' resultType
///   bufferType ':' resultType
/// The second form is used only if the bufferType does not match the result
/// type.
template <typename BufferType, typename TensorType>
void genericPrintMayBeDynamicBuffer(AsmPrinter &printer, const Operation *op,
                                    BufferType bufferType,
                                    TensorType resultType) {
  if (BufferType::get(resultType) != bufferType) {
    printer.printType(bufferType);
    printer.getStream() << ' ';
  }
  printer.getStream() << ": ";
  printer.printType(resultType);
}

//===----------------------------------------------------------------------===//
// Generic verification helpers
//===----------------------------------------------------------------------===//

/// Checks the tensor spec, buffer type and result tensor types are consistent.
template <typename BufferType, typename TensorType>
LogicalResult genericTensorCreateVerify(Operation *op, BufferType bufferType,
                                        CpuTensorSpecType tensorSpecType,
                                        TensorType tensorType) {
  if (tensorSpecType != tensorType.getTensorSpec())
    return op->emitOpError() << "expected spec to match result tensor spec"
                             << " (" << tensorSpecType << " vs "
                             << tensorType.getTensorSpec() << ")";

  BufferType expectedBufferType = BufferType::get(tensorType);
  bool bufferIsStatic = bufferType.getOptSize().has_value();
  bool expectedBufferIsStatic = expectedBufferType.getOptSize().has_value();
  if (bufferIsStatic != expectedBufferIsStatic) {
    // The buffer type and tensor spec types don't agree on whether they
    // are static. This can arise when lowering value-semantic reshapes
    // into buffer-semantic form. For example:
    //   %z = reshape %x, %y : (tensor<?x?xf32>, tensor<2xui32>) ->
    //   tensor<8x16xf32>
    // Here %z will be created from the same buffer as %x, but will now have
    // a known size. Conversely, in:
    //   %z = reshape %x, %y : (tensor<8x4xf32>, tensor<2xui32>) ->
    //   tensor<?x?xf32>
    // the buffer for %z will now have an unknown size.
    // In this case there's nothing to check.
    // no-op
  } else if (bufferType != expectedBufferType) {
    return op->emitOpError()
           << "expected buffer type to match type implied by tensor spec ("
           << bufferType << " vs " << expectedBufferType << ").";
  }
  return success();
}

/// Checks the buffer type and tensor types are consistent.
template <typename BufferType, typename TensorType>
LogicalResult
genericTensorExtractVerify(Operation *op, TensorType tensorType,
                           BufferType resultBufferType,
                           CpuTensorSpecType resultTensorSpecType) {
  if (tensorType.getTensorSpec() != resultTensorSpecType)
    return op->emitOpError()
           << "expected the result spec to match tensor's spec"
           << " (" << resultTensorSpecType << " vs "
           << tensorType.getTensorSpec() << ")";
  BufferType expectedBufferType = BufferType::get(tensorType);
  if (expectedBufferType != resultBufferType)
    return op->emitOpError()
           << "expected the result buffer to match the type "
              "implied by the tensor type ("
           << resultBufferType << " vs " << expectedBufferType << ").";
  return success();
}

//===----------------------------------------------------------------------===//
// TypeBuilder
//===----------------------------------------------------------------------===//

/// A TypeBuilder provides methods for building device specific tensor types.
/// It is only needed during lowering when lowered types must be provided
/// ahead of lowering itself.
class TypeBuilder {
public:
  virtual ~TypeBuilder() = default;

  /// Returns Type representing device specific tensor with given tensor spec
  /// type.
  virtual Type tensorType(MLIRContext *ctx,
                          CpuTensorSpecType tensorSpecType) = 0;
};

//===----------------------------------------------------------------------===//
// CpuTypeBuilder
//===----------------------------------------------------------------------===//

class CpuTypeBuilder : public TypeBuilder {
public:
  using TensorType = CpuTensorType;

  ~CpuTypeBuilder() override = default;

  Type tensorType(MLIRContext *ctx, CpuTensorSpecType tensorSpecType) override {
    return CpuTensorType::get(ctx, tensorSpecType);
  }
};

//===----------------------------------------------------------------------===//
// BufferBuilder
//===----------------------------------------------------------------------===//

/// A BufferBuilder provides methods for building device-dependent buffer and
/// tensor operations. Note that all chain (input and output) must be
/// !mef.chains.
class BufferBuilder {
public:
  virtual ~BufferBuilder() = default;

  /// Returns Value representing the constant buffer with given attribute as
  /// its contents.
  virtual Value bufferConstant(OpBuilder &builder, Location loc,
                               ElementsAttr value) = 0;

  /// Returns the Values representing the buffer and tensor spec of the tensor
  /// Value.
  virtual std::pair<Value, TypedValue<CpuTensorSpecType>>
  tensorExtract(OpBuilder &builder, Location loc, Value tensor) = 0;

  /// Returns the Value representing the shape of the tensor Value, as
  /// an !mgp.cpu.tensor_spec with unknown dtype.
  virtual TypedValue<CpuTensorSpecType>
  tensorExtractShape(OpBuilder &builder, Location loc, Value tensor) = 0;

  /// Return the Value representing a tensor constructed from the Values
  /// representing its buffer and tensor spec.
  virtual Value tensorCreate(OpBuilder &builder, Location loc, Value inChain,
                             Value buffer,
                             TypedValue<CpuTensorSpecType> tensorSpec) = 0;

  /// Returns the Value representing the buffer of given statically known size
  /// and alignment. The alignment may be ignored for some devices.
  virtual Value bufferAllocStatic(OpBuilder &builder, Location loc,
                                  Value inChain, uint64_t size,
                                  uint64_t alignment) = 0;

  /// Returns the Value representing the buffer of size matching the given
  /// tensor spec and alignment. The alignment may be ignored for some devices.
  virtual Value bufferAllocSpec(OpBuilder &builder, Location loc, Value inChain,
                                TypedValue<MGP::CpuTensorSpecType> tensorSpec,
                                uint64_t alignment) = 0;

  /// Returns the chain indicating when srcBuffer on the host device has been
  /// transferred to dstBuffer on this builder's device.
  virtual Value bufferTransferFromHost(OpBuilder &builder, Location loc,
                                       Value inChain,
                                       TypedValue<MGP::CpuBufferType> srcBuffer,
                                       Value dstBuffer) = 0;

  /// Returns the chain indicating when srcBuffer on this builder's device has
  /// been transferred to dstBuffer on the dstDevice.
  virtual Value
  bufferTransferToHost(OpBuilder &builder, Location loc, Value inChain,
                       Value srcBuffer,
                       TypedValue<MGP::CpuBufferType> dstBuffer) = 0;

  /// Return the chain indicating tensor has been printed.
  virtual Value debugTensorPrint(OpBuilder &builder, Location loc,
                                 Value inChain, Value tensor,
                                 StringAttr label) = 0;
};

//===----------------------------------------------------------------------===//
// CpuBufferBuilder
//===----------------------------------------------------------------------===//

class CpuBufferBuilder : public BufferBuilder {
public:
  using BufferType = CpuBufferType;
  using TensorType = CpuTensorType;

  ~CpuBufferBuilder() override = default;

  Value bufferConstant(OpBuilder &builder, Location loc,
                       ElementsAttr data) override;

  std::pair<Value, TypedValue<CpuTensorSpecType>>
  tensorExtract(OpBuilder &builder, Location loc, Value tensor) override;

  TypedValue<CpuTensorSpecType>
  tensorExtractShape(OpBuilder &builder, Location loc, Value tensor) override;

  Value tensorCreate(OpBuilder &builder, Location loc, Value inChain,
                     Value buffer,
                     TypedValue<CpuTensorSpecType> tensorSpec) override;

  Value bufferAllocStatic(OpBuilder &builder, Location loc, Value inChain,
                          uint64_t size, uint64_t alignment) override;

  Value bufferAllocSpec(OpBuilder &builder, Location loc, Value inChain,
                        TypedValue<MGP::CpuTensorSpecType> tensorSpec,
                        uint64_t alignment) override;

  Value bufferTransferFromHost(OpBuilder &builder, Location loc, Value inChain,
                               TypedValue<MGP::CpuBufferType> srcBuffer,
                               Value dstBuffer) override;

  Value bufferTransferToHost(OpBuilder &builder, Location loc, Value inChain,
                             Value srcBuffer,
                             TypedValue<MGP::CpuBufferType> dstBuffer) override;

  Value debugTensorPrint(OpBuilder &builder, Location loc, Value inChain,
                         Value tensor, StringAttr label) override;
};

//===----------------------------------------------------------------------===//
// BufferizedTensor
//===----------------------------------------------------------------------===//

/// A representation of tensor as a pair of a buffer and its tensor spec.
/// The tensor spec is always an MGP::CpuTensorSpecType, but the buffer type
/// depends on the device being lowered to.
struct BufferizedTensor {
  /// The buffer holding the lowered tensor's contents. Will be written into
  /// by the producing op, then read by all consuming ops. The actual type
  /// representing the buffer (mgp.cpu.buffer, cuda.buffer, etc) depends
  /// on the device it is to be stored on.
  Value buffer;

  /// The mgp.cpu.tensor_spec describing the shape and dtype of tensor
  /// at runtime.
  ///
  /// TODO(|#17379): Avoid committing to a tensor spec, and instead just
  /// capture the index-typed values holding dynamic dimensions, or the
  /// 'shape' (ie a spec with unknown dtype) if the tensor has dynamic rank.
  /// I think this change is best done when finishing supprot for dynamic
  /// ranked MO tensors.
  TypedValue<CpuTensorSpecType> tensorSpec;

  BufferizedTensor() = default;

  BufferizedTensor(Value buffer, TypedValue<MGP::CpuTensorSpecType> tensorSpec)
      : buffer(buffer), tensorSpec(tensorSpec) {}
};

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_BUFFERBUILDER_H
