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

#include "GenericML/MGPDialect/MGPOps.h"
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

  virtual void vtableAnchor();
};

//===----------------------------------------------------------------------===//
// CpuTypeBuilder
//===----------------------------------------------------------------------===//

class CpuTypeBuilder : public TypeBuilder {
public:
  ~CpuTypeBuilder() override = default;

  Type tensorType(MLIRContext *ctx, CpuTensorSpecType tensorSpecType) override {
    return CpuTensorType::get(ctx, tensorSpecType);
  }
};

//===----------------------------------------------------------------------===//
// BufferBuilder
//===----------------------------------------------------------------------===//

/// A BufferBuilder provides methods for building device-dependent buffer,
/// tensor and kernel ops. All chains (input and output) must be !mef.chains,
/// even if the device has a device-specific 'chain' like type. Since an
/// instance of this class is created for each device in the model, derived
/// classes may accumulate device-specific lowering state therein.
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

  /// Returns the chain indicating the given archive with given kernel and
  /// constant names has been loaded into the runtime.
  virtual Value kernelLoadArchive(OpBuilder &builder, Location loc,
                                  Value inChain,
                                  TypedValue<MGP::ContextType> initContext,
                                  ElementsAttr archive, ArrayAttr kernelNames,
                                  ArrayAttr constantNames) = 0;

  /// Returns the chain indicating the given kernel has completed execution
  /// on the given operands. Type parameters from the operand !mo.tensor types
  /// must have already been lowered to their index typed dimensios. It is
  /// possible for the kernel to return output dimensions, however this is
  /// only used by synchronous shape functions.
  ///
  /// If isViaHost is true, the kernel is implemented in two parts:
  ///  - the host side kernel, which is given arguments so as to be able
  ///    to call/launch...
  ///  - the device side kernel
  /// The kernelExecute will just setup the correct execution op for the
  /// host side kernel.
  virtual Value
  kernelExecute(OpBuilder &builder, Location loc, StringRef kernelName,
                bool isViaHost, Value inChain,
                TypedValue<MGP::ContextType> execContext,
                ArrayRef<Value> buffers,
                ArrayRef<InOutSignatureAttr::InOutSemantics> bufferSemantics,
                ArrayRef<Value> inDimensions, uint64_t numOutDimensions,
                SmallVectorImpl<Value> &outDimensions) = 0;

  /// Places the given buffer into a freshly allocated slot within the
  /// given !mgp.context. Returns the chain which indicates when the buffer
  /// has been placed, and the index of the slot from which the buffer may
  /// later be retrieved. Note that the input and output chains are always
  /// !mef.chain, irrespective of the native chain type for this builder.
  ///
  /// Generally this is only appropriate within the 'init' block of an
  /// mgp.model op.
  virtual std::pair<TypedValue<MEF::ChainType>, size_t>
  cache(OpBuilder &initBuilder, Location loc,
        TypedValue<MEF::ChainType> inChain, ContextCreateOp contextCreateOp,
        Value buffer) = 0;

  /// Retrieves a buffer of the expected type from the given slot of the given
  /// !mgp.context. Note that the input chain is always a !mef.chain,
  /// irrespective of the native chain type for this builder.
  virtual Value getCached(OpBuilder &builder, Location loc,
                          TypedValue<MEF::ChainType> inChain,
                          TypedValue<MGP::ContextType> execContext,
                          size_t bufferSlot, Type expectedBufferType) = 0;

  /// Returns the join of all the given chain values, which must be all native
  /// to this builder.
  virtual Value chainCreate(OpBuilder &builder, Location loc,
                            ArrayRef<Value> inChains) = 0;

  /// Returns the given chain, in builder native form, as a !mef.chain.
  virtual TypedValue<MEF::ChainType>
  toMEFChain(OpBuilder &builder, Location loc, Value inChain) = 0;

  /// Returns the given !mef.chain as a chain in builder native form.
  virtual Value fromMEFChain(OpBuilder &builder, Location loc,
                             TypedValue<MEF::ChainType> inChain) = 0;

  virtual void vtableAnchor();
};

//===----------------------------------------------------------------------===//
// CpuBufferBuilder
//===----------------------------------------------------------------------===//

class CpuBufferBuilder : public BufferBuilder {
public:
  CpuBufferBuilder(uint64_t slot) : slot(slot) {}
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

  Value kernelLoadArchive(OpBuilder &builder, Location loc, Value inChain,
                          TypedValue<MGP::ContextType> initContext,
                          ElementsAttr archive, ArrayAttr kernelNames,
                          ArrayAttr constantNames) override;

  Value
  kernelExecute(OpBuilder &builder, Location loc, StringRef kernelName,
                bool isViaHost, Value inChain,
                TypedValue<MGP::ContextType> execContext,
                ArrayRef<Value> buffers,
                ArrayRef<InOutSignatureAttr::InOutSemantics> bufferSemantics,
                ArrayRef<Value> inDimensions, uint64_t numOutDimensions,
                SmallVectorImpl<Value> &outDimensions) override;

  Value chainCreate(OpBuilder &builder, Location loc,
                    ArrayRef<Value> inChains) override;

  std::pair<TypedValue<MEF::ChainType>, size_t>
  cache(OpBuilder &builder, Location loc, TypedValue<MEF::ChainType> inChain,
        ContextCreateOp contextCreateOp, Value buffer) override;

  Value getCached(OpBuilder &builder, Location loc,
                  TypedValue<MEF::ChainType> inChain,
                  TypedValue<MGP::ContextType> execContext, size_t slot,
                  Type expectedBufferType) override;

  TypedValue<MEF::ChainType> toMEFChain(OpBuilder &builder, Location loc,
                                        Value inChain) override;

  Value fromMEFChain(OpBuilder &builder, Location loc,
                     TypedValue<MEF::ChainType> inChain) override;

  /// Track this compiled kernel as corresponding to this index in the
  /// compiled kernels context.
  void trackCompiledKernel(StringRef name, size_t index) {
    compiledKernelsToIndex[name] = index;
  }

  /// Get the index of the kernel in the compiled kernels context.
  size_t getCompiledKernelIndex(StringRef name) {
    auto itr = compiledKernelsToIndex.find(name);
    assert(itr != compiledKernelsToIndex.end() &&
           "All kernels must be accounted for through lowering the load "
           "kernels.");
    return itr->second;
  }

  /// Returns the slot holding the compiled kernels context in the !mgp.context.
  uint64_t getSlot() const { return slot; }

private:
  /// Maps the name of each compiled kernel to its index in the compiled kernel
  /// context.
  llvm::StringMap<size_t> compiledKernelsToIndex;
  /// The slot holding the compiled kernels context in the !mgp.context.
  uint64_t slot;
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

//===----------------------------------------------------------------------===//
// BufferBuilderContext
//===----------------------------------------------------------------------===//

/// Collects the type and buffer builders for every device in a model.
class BufferBuilderContext {
public:
  BufferBuilderContext(DeviceRefAttr hostDevice) : hostDevice(hostDevice) {}

  /// Register the given type builder for the given device.
  void registerTypeBuilder(DeviceRefAttr device,
                           std::unique_ptr<TypeBuilder> typeBuilder) {
    bool inserted =
        typeBuilderMap.insert({device, std::move(typeBuilder)}).second;
    assert(inserted && "a type builder has already been registered for device");
  }

  /// Returns the type builder to use for device. The builder is owned by
  /// this object.
  TypeBuilder *getTypeBuilder(DeviceRefAttr device) const {
    auto itr = typeBuilderMap.find(device);
    assert(itr != typeBuilderMap.end() &&
           "no type builder has been registered for device");
    return itr->second.get();
  }

  /// Register the given buffer builder for the given device.
  void registerBufferBuilder(DeviceRefAttr device,
                             std::unique_ptr<BufferBuilder> bufferBuilder) {
    bool inserted =
        bufferBuilderMap.insert({device, std::move(bufferBuilder)}).second;
    assert(inserted &&
           "a buffer builder has already been registered for device");
  }

  /// Returns the buffer builder to use for device. The builder is owned by
  /// this object.
  BufferBuilder *getBufferBuilder(DeviceRefAttr device) const {
    assert(device);
    auto itr = bufferBuilderMap.find(device);
#if MODULAR_DEBUG
    if (itr == bufferBuilderMap.end()) {
      llvm::errs() << "no buffer builder has been registered for device "
                   << device << "\n";
    }
#endif
    assert(itr != bufferBuilderMap.end() &&
           "no buffer builder has been registered for device");
    return itr->second.get();
  }

  /// Return the 'host' device.
  DeviceRefAttr getHostDevice() const { return hostDevice; }

  /// Returns the buffer builder to use for the 'host' device.
  BufferBuilder *getHostBufferBuilder() const {
    return getBufferBuilder(hostDevice);
  }

private:
  /// The distinguished 'host' device.
  DeviceRefAttr hostDevice;

  /// The type builder to use for each device.
  llvm::DenseMap<DeviceRefAttr, std::unique_ptr<TypeBuilder>> typeBuilderMap;

  /// The buffer builder to use for each device.
  llvm::DenseMap<DeviceRefAttr, std::unique_ptr<BufferBuilder>>
      bufferBuilderMap;
};

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_BUFFERBUILDER_H
