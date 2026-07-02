//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"

#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

namespace M::MOGG {

// Builds the canonical vendor-prefixed accelerator string ("nvidia:sm_90a",
// "amdgpu:gfx942") from an explicit {arch, triple} override, or empty if the
// override does not describe a GPU target. The override carries the graph's
// default-device arch/triple, which may describe a *CPU* (e.g. "graniterapids"
// on an x86_64 host when the default device is the host) — that must not be
// used as an accelerator, so a non-GPU triple maps to empty and the caller
// falls back to probing the host GPU. GPU classification and the vendor prefix
// come from the triple (authoritative), not from guessing the arch mnemonic.
// The prefix matches what downstream consumers (e.g. the NVPTX backend's ptxas
// invocation) parse by splitting on ':'.
static std::string overrideAcceleratorArch(StringRef arch,
                                           StringRef tripleStr) {
  if (arch.empty() || tripleStr.empty())
    return "";
  llvm::Triple triple(tripleStr);
  if (!KGEN::isGPUTriple(triple))
    return ""; // non-GPU target (e.g. a CPU default device)
  if (arch.contains(':'))
    return arch.str(); // already vendor-qualified (e.g. Metal's "metal:3")
  if (triple.isNVPTX())
    return ("nvidia:" + arch).str();
  if (triple.isAMDGCN())
    return ("amdgpu:" + arch).str();
  if (KGEN::isMetalTriple(triple))
    return ("metal:" + arch).str();
  return arch.str(); // GPU triple, unrecognized vendor: pass the arch through
}

static ErrorOr<std::string> resolveVirtualCpuArch(StringRef requested,
                                                  StringRef triple) {
  if (requested.empty())
    return std::string();
  if (requested == "native")
    return Error("virtual CPU target 'native' re-leaks the host CPU; pass an "
                 "explicit arch (e.g. 'x86-64-v3') or 'generic'.");
  if (requested != "generic")
    return requested.str();

  llvm::Triple t(triple);
  if (t.getArch() == llvm::Triple::x86_64)
    return std::string("x86-64");
  if (t.isAArch64())
    return std::string("generic"); // LLVM's AArch64 baseline CPU name
  return Error("virtual CPU target 'generic' is only supported for x86_64 and "
               "aarch64 hosts.");
}

ErrorOr<VirtualCpuTargetSpec> parseVirtualCpuTarget(StringRef knob,
                                                    StringRef hostTriple) {
  // No '=' => legacy bare-arch form.
  if (!knob.contains('=')) {
    ErrorOr<std::string> archOr = resolveVirtualCpuArch(knob, hostTriple);
    if (archOr.isError())
      return archOr.takeError();
    VirtualCpuTargetSpec spec;
    spec.cpu = archOr.takeValue();
    return spec;
  }

  VirtualCpuTargetSpec spec;
  bool haveCpu = false;
  llvm::SmallVector<StringRef, 4> fields;
  knob.split(fields, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef field : fields) {
    auto [key, value] = field.split('=');
    key = key.trim();
    value = value.trim();
    if (key == "triple") {
      spec.triple = value.str();
    } else if (key == "cpu") {
      spec.cpu = value.str();
      haveCpu = !value.empty();
    } else if (key == "features") {
      spec.features = value.str();
      spec.hasExplicitFeatures = true;
    } else {
      return Error("unknown virtual CPU target field '" + key.str() +
                   "'; expected 'triple', 'cpu', or 'features'.");
    }
  }
  if (!haveCpu)
    return Error("virtual CPU target descriptor must specify a non-empty "
                 "'cpu=' field.");

  // De-sugar "generic" against the descriptor's triple (else host) so it
  // resolves to the target arch's baseline, not the host's.
  StringRef effTriple =
      spec.triple.empty() ? hostTriple : StringRef(spec.triple);
  ErrorOr<std::string> archOr = resolveVirtualCpuArch(spec.cpu, effTriple);
  if (archOr.isError())
    return archOr.takeError();
  spec.cpu = archOr.takeValue();
  return spec;
}

ErrorOr<TargetInfoAttr> getTargetInfo(MLIRContext *ctx,
                                      StringRef acceleratorArchOverride,
                                      StringRef acceleratorTripleOverride) {
  std::string triple = llvm::sys::getDefaultTargetTriple();
  std::string cpu = llvm::sys::getHostCPUName().str();
  std::string features = getHostCPUFeatures();

  // Cross-compilation: when the override describes a GPU target, honor it
  // instead of probing the host GPU, so a target GPU can be selected on a
  // machine that lacks it (or has no GPU at all). A non-GPU override (e.g. a
  // CPU default device) falls back to host detection so GPU kernels still
  // target the real GPU. Only the accelerator is overridden; the host CPU
  // triple/CPU/features still describe the machine running codegen.
  auto acceleratorArch = [&]() -> std::string {
    if (std::string gpu = overrideAcceleratorArch(acceleratorArchOverride,
                                                  acceleratorTripleOverride);
        !gpu.empty())
      return gpu;
    return M::Driver::Device::getAcceleratorArchOrEmpty();
  };

  std::string virtualCpu = M::Driver::getVirtualCpuTarget();
  if (!virtualCpu.empty()) {
    ErrorOr<VirtualCpuTargetSpec> specOr =
        parseVirtualCpuTarget(virtualCpu, triple);
    if (specOr.isError())
      return specOr.takeError();
    VirtualCpuTargetSpec spec = specOr.takeValue();
    if (!spec.triple.empty())
      triple = spec.triple;
    cpu = spec.cpu;
    // Name-only (empty) features unless the descriptor pins them verbatim;
    // name-only derives from the CPU model with no host CPUID leak.
    features = spec.hasExplicitFeatures ? spec.features : "";
  }

  ErrorOr<TargetInfoAttr> targetOr =
      getTargetInfoFor(ctx, triple, cpu, features, /*tuneCpu=*/"",
                       /*acceleratorArch=*/acceleratorArch());
  if (targetOr.isError())
    return targetOr;

  TargetInfoAttr target = targetOr.takeValue();

  // Retrieve the target info from the device spec, however 'complete' it with
  // additional fields derived from the 'core' fields.
  ErrorOr<TargetInfoAttr> targetInfoOr = getTargetInfoFor(
      ctx, target.getTripleStr(), target.getArch(), target.getFeatures(),
      target.getTuneCpu(), acceleratorArch());
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
