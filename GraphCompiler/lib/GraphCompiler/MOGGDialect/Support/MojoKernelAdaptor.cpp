//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/StringSwitch.h"

namespace M::MOGG {

MojoKernelOperandAdaptor::MojoKernelOperandAdaptor(
    std::optional<uint64_t> positionInFunction, StringAttr typeName,
    ArrayAttr argumentSourceNames, ArrayAttr argsIoSpecs, bool isByRefResult,
    bool promoteSIMDToFusedTensor) {
  if (positionInFunction.has_value()) {
    sourceDescriptor = MojoKernelOperandSourceDescriptor{
        cast<StringAttr>(argumentSourceNames[*positionInFunction]).strref(),
        *positionInFunction, isByRefResult};
  }

  auto getIOSpec = [&]() -> MOGG::IOSpec {
    // Default to InputTensor if no spec info available.
    auto defaultSpec = promoteSIMDToFusedTensor ? MOGG::IOSpec::FusedInputTensor
                                                : MOGG::IOSpec::InputTensor;
    if (!positionInFunction.has_value() || argsIoSpecs == nullptr ||
        *positionInFunction >= argsIoSpecs.size())
      return defaultSpec;

    auto specAttr = dyn_cast<StringAttr>(argsIoSpecs[*positionInFunction]);
    if (!specAttr)
      return defaultSpec;
    return *MOGG::toIOSpec(specAttr);
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
      .Case(MOJO_FUSED_INPUT_VARIADIC_TENSORS_NAME,
            [&]() {
              underlyingType.emplace<VariadicTensorOperandAdaptor>(
                  VariadicTensorOperandAdaptor{getIOSpec()});
            })
      .Case(MOJO_FUSED_OUTPUT_VARIADIC_TENSORS_NAME,
            [&]() {
              underlyingType.emplace<VariadicTensorOperandAdaptor>(
                  VariadicTensorOperandAdaptor{getIOSpec()});
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
      .Case(DeviceContextOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<DeviceContextOperandAdaptor>(
                  DeviceContextOperandAdaptor{});
            })
      .Case(DeviceContextListOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<DeviceContextListOperandAdaptor>(
                  DeviceContextListOperandAdaptor{});
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
      TensorOperandAdaptor{MOGG::IOSpec::FusedOutputTensor});
  return res;
}

} // namespace M::MOGG
