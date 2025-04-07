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
    ArrayAttr positionOfMutableTensors, ArrayAttr positionOfFusedTensors,
    ArrayAttr argumentSourceNames, uint64_t offset) {
  if (positionInFunction.has_value()) {
    sourceDescriptor = MojoKernelOperandSourceDescriptor{
        cast<StringAttr>(argumentSourceNames[*positionInFunction]).strref(),
        *positionInFunction};
  }

  auto isMutable = [&]() {
    if (!positionOfMutableTensors)
      return false;
    return llvm::any_of(positionOfMutableTensors, [&](Attribute indexAttr) {
      return (*positionInFunction - offset) ==
             uint64_t(cast<IntegerAttr>(indexAttr).getInt());
    });
  };

  auto isFused = [&]() {
    if (!positionOfFusedTensors)
      return false;
    return llvm::any_of(positionOfFusedTensors, [&](Attribute indexAttr) {
      return *positionInFunction ==
             uint64_t(cast<IntegerAttr>(indexAttr).getInt());
    });
  };

  llvm::StringSwitch<std::function<void(void)>>(typeName)
      .Case(TensorOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<TensorOperandAdaptor>(
                  TensorOperandAdaptor{isMutable(), isFused()});
            })
      .Case(VariadicTensorOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<VariadicTensorOperandAdaptor>(
                  VariadicTensorOperandAdaptor{isMutable(), isFused()});
            })
      .Case(ListOfTensorOperandAdaptor::typeName,
            [&]() {
              underlyingType.emplace<ListOfTensorOperandAdaptor>(
                  ListOfTensorOperandAdaptor{isMutable(), isFused()});
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
