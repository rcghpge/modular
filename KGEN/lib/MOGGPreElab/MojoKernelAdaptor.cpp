//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/StringSwitch.h"

namespace M::KGEN::MOGGPreElab {

MojoKernelOperandAdaptor::MojoKernelOperandAdaptor(
    std::optional<uint64_t> positionInFunction, StringRef typeName,
    ArrayAttr argumentSourceNames, ArrayAttr argsIoSpecs, uint64_t offset) {
  if (positionInFunction.has_value()) {
    sourceDescriptor = MojoKernelOperandSourceDescriptor{
        cast<StringAttr>(argumentSourceNames[*positionInFunction]).strref(),
        *positionInFunction};
  }

  auto getIOSpec = [&]() -> KGEN::MOGGPreElab::IOSpec {
    // Default to InputTensor if no spec info available.
    auto defaultSpec = KGEN::MOGGPreElab::IOSpec::InputTensor;
    if (!positionInFunction.has_value() || argsIoSpecs == nullptr ||
        *positionInFunction >= argsIoSpecs.size())
      return defaultSpec;

    auto specAttr = dyn_cast<StringAttr>(argsIoSpecs[*positionInFunction]);
    if (!specAttr)
      return defaultSpec;
    return *KGEN::MOGGPreElab::toIOSpec(specAttr);
  };

  llvm::StringSwitch<std::function<void(void)>>(typeName)
      .Case(TensorOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<TensorOperandAdaptor>(
                  TensorOperandAdaptor{getIOSpec()});
            })
      .Case(VariadicTensorOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<VariadicTensorOperandAdaptor>(
                  VariadicTensorOperandAdaptor{getIOSpec()});
            })
      .Case(ListOfTensorOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<ListOfTensorOperandAdaptor>(
                  ListOfTensorOperandAdaptor{getIOSpec()});
            })
      .Case(ScalarOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<ScalarOperandAdaptor>(
                  ScalarOperandAdaptor{});
            })
      .Case(DevicesContextPtrOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<DevicesContextPtrOperandAdaptor>(
                  DevicesContextPtrOperandAdaptor{});
            })
      .Case(DevicesContextPtrListOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<DevicesContextPtrListOperandAdaptor>(
                  DevicesContextPtrListOperandAdaptor{});
            })
      .Default([&]() {
        underlyingType.emplace<OpaqueOperandAdaptor>(
            OpaqueOperandAdaptor{typeName});
      })();
}

} // namespace M::KGEN::MOGGPreElab
