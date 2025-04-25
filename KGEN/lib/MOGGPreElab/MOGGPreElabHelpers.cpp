//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"

namespace M::KGEN::MOGGPreElab {

StringRef toString(IOSpec spec) {
  switch (spec) {
  case IOSpec::InputTensor:
    return kInputTensor;
  case IOSpec::OutputTensor:
    return kOutputTensor;
  case IOSpec::MutableInputTensor:
    return kMutableInputTensor;
  case IOSpec::FusedInputTensor:
    return kFusedInputTensor;
  case IOSpec::FusedOutputTensor:
    return kFusedOutputTensor;
  case IOSpec::FusedComputeOutputTensor:
    return kFusedComputeOutputTensor;
  }
}

std::optional<IOSpec> toIOSpec(StringRef str) {
  if (str == kInputTensor)
    return IOSpec::InputTensor;
  if (str == kOutputTensor)
    return IOSpec::OutputTensor;
  if (str == kMutableInputTensor)
    return IOSpec::MutableInputTensor;
  if (str == kFusedInputTensor)
    return IOSpec::FusedInputTensor;
  if (str == kFusedOutputTensor)
    return IOSpec::FusedOutputTensor;
  if (str == kFusedComputeOutputTensor)
    return IOSpec::FusedComputeOutputTensor;
  return std::nullopt;
}

std::pair<TypedAttr, TypedAttr> getParams(KGEN::MOGGPreElab::IOSpec ioSpec,
                                          Builder &builder) {
  switch (ioSpec) {
  case KGEN::MOGGPreElab::IOSpec::InputTensor:
    return {builder.getBoolAttr(KGEN::MOGGPreElab::kIOSpecImmutable),
            builder.getIndexAttr(KGEN::MOGGPreElab::kIOSpecIOInput)};

  case KGEN::MOGGPreElab::IOSpec::OutputTensor:
    return {builder.getBoolAttr(KGEN::MOGGPreElab::kIOSpecMutable),
            builder.getIndexAttr(KGEN::MOGGPreElab::kIOSpecIOOutput)};

  case KGEN::MOGGPreElab::IOSpec::MutableInputTensor:
    return {builder.getBoolAttr(KGEN::MOGGPreElab::kIOSpecMutable),
            builder.getIndexAttr(KGEN::MOGGPreElab::kIOSpecIOInput)};

  case KGEN::MOGGPreElab::IOSpec::FusedInputTensor:
    return {builder.getBoolAttr(KGEN::MOGGPreElab::kIOSpecImmutable),
            builder.getIndexAttr(KGEN::MOGGPreElab::kIOSpecIOFusedInput)};

  case KGEN::MOGGPreElab::IOSpec::FusedOutputTensor:
    return {builder.getBoolAttr(KGEN::MOGGPreElab::kIOSpecMutable),
            builder.getIndexAttr(KGEN::MOGGPreElab::kIOSpecIOFusedOutput)};

  case KGEN::MOGGPreElab::IOSpec::FusedComputeOutputTensor:
    return {
        builder.getBoolAttr(KGEN::MOGGPreElab::kIOSpecMutable),
        builder.getIndexAttr(KGEN::MOGGPreElab::kIOSpecIOFusedComputeOutput)};
  }
}

bool isOutputIOSpec(IOSpec spec) {
  return spec == IOSpec::OutputTensor || spec == IOSpec::FusedOutputTensor ||
         spec == IOSpec::FusedComputeOutputTensor;
}

bool isFusableIOSpec(IOSpec spec) {
  return spec == IOSpec::FusedInputTensor ||
         spec == IOSpec::FusedOutputTensor ||
         spec == IOSpec::FusedComputeOutputTensor;
}

bool isMutableIOSpec(IOSpec spec) { return spec == IOSpec::MutableInputTensor; }

} // namespace M::KGEN::MOGGPreElab
