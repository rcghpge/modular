//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"

#include "KGEN/KGENDialect/KGENUtils.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/TargetParser/Host.h"

namespace M::MOGG {

ErrorOr<TargetInfoAttr> getTargetInfo(MLIRContext *ctx) {
  ErrorOr<TargetInfoAttr> targetOr = getTargetInfoFor(
      ctx, llvm::sys::getDefaultTargetTriple(), llvm::sys::getHostCPUName(),
      getHostCPUFeatures(), /*tuneCpu=*/"",
      /*acceleratorArch=*/M::Driver::Device::getAcceleratorArchOrEmpty());
  if (targetOr.isError())
    return targetOr;

  TargetInfoAttr target = targetOr.takeValue();

  // Retrieve the target info from the device spec, however 'complete' it with
  // additional fields derived from the 'core' fields.
  ErrorOr<TargetInfoAttr> targetInfoOr = getTargetInfoFor(
      ctx, target.getTripleStr(), target.getArch(), target.getFeatures(),
      target.getTuneCpu(), M::Driver::Device::getAcceleratorArchOrEmpty());
  return targetInfoOr;
}

DenseResourceElementsAttr createArchiveAttr(MLIRContext *ctx, BufferRef buffer,
                                            StringRef name) {
  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);

  // Pretend this is a "tensor" of data.
  auto attrType =
      RankedTensorType::get({(int64_t)buffer->getBufferSize()},
                            IntegerType::get(ctx, 8, IntegerType::Unsigned));
  std::string attrName = "archive_" + name.str();
  mlir::AsmResourceBlob blob(
      {buffer->getBufferStart(), buffer->getBufferSize()}, /*dataAlignment=*/8,
      /*deleter=*/
      [buffer = std::move(buffer)](void *, size_t, size_t) mutable {
        buffer.reset(); // Shouldn't need to do this, but just to be explicit.
      },
      /*isMutable=*/false);
  return DenseResourceElementsAttr::get(
      attrType, resourceManager.insert(attrName, std::move(blob)));
}

} // namespace M::MOGG
