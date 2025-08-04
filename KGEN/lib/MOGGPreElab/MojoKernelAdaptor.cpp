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
    std::optional<uint64_t> positionInFunction, StringAttr typeName,
    ArrayAttr argumentSourceNames, ArrayAttr argsIoSpecs, bool isByRefResult,
    bool promoteSIMDToFusedTensor) {
  if (positionInFunction.has_value()) {
    sourceDescriptor = MojoKernelOperandSourceDescriptor{
        cast<StringAttr>(argumentSourceNames[*positionInFunction]).strref(),
        *positionInFunction, isByRefResult};
  }

  auto getIOSpec = [&]() -> KGEN::MOGGPreElab::IOSpec {
    // Default to InputTensor if no spec info available.
    auto defaultSpec = promoteSIMDToFusedTensor
                           ? KGEN::MOGGPreElab::IOSpec::FusedInputTensor
                           : KGEN::MOGGPreElab::IOSpec::InputTensor;
    if (!positionInFunction.has_value() || argsIoSpecs == nullptr ||
        *positionInFunction >= argsIoSpecs.size())
      return defaultSpec;

    auto specAttr = dyn_cast<StringAttr>(argsIoSpecs[*positionInFunction]);
    if (!specAttr)
      return defaultSpec;
    return *KGEN::MOGGPreElab::toIOSpec(specAttr);
  };

  if (!typeName) {
    underlyingType.emplace<UnsupportedOperandAdaptor>();
    return;
  }

  llvm::StringSwitch<std::function<void(void)>>(typeName.strref())
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
              if (promoteSIMDToFusedTensor) {
                underlyingType.emplace<TensorOperandAdaptor>(
                    TensorOperandAdaptor{getIOSpec()});
              } else {
                underlyingType.emplace<ScalarOperandAdaptor>(
                    ScalarOperandAdaptor{});
              }
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

MojoKernelOperandAdaptor
MojoKernelOperandAdaptor::buildElementwiseOutputOperand() {
  MojoKernelOperandAdaptor res;
  res.sourceDescriptor = MojoKernelOperandSourceDescriptor{
      "out", /*position=*/0, /*isByRefResult=*/false};
  res.underlyingType.emplace<TensorOperandAdaptor>(
      TensorOperandAdaptor{KGEN::MOGGPreElab::IOSpec::FusedOutputTensor});
  return res;
}

} // namespace M::KGEN::MOGGPreElab
