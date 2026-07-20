//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MOToMOGGLowering.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/OperandMapper.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/AssertStream.h"
#include "Support/Filesystem/Paths.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::MOGG;

MojoRegistry::MojoRegistry(mlir::ModuleOp module) : deviceInfoMapping(module) {
  for (auto decl : module.getOps<KernelDeclOp>()) {
    kernelByName[decl.getRegistration()].push_back(decl);
    kernelBySymName[decl.getSymName()] = decl;
  }
  for (auto decl : module.getOps<PrimitiveDeclOp>()) {
    llvm::StringRef reg = decl.getRegistration();
    if (!primitiveByName.contains(reg)) {
      primitiveByName[reg] = decl;
      primitiveBySymName[decl.getSymName()] = decl;
    }
  }
  for (auto decl : module.getOps<ShapeFunctionDeclOp>()) {
    shapeFuncByName[decl.getRegistration()] = decl;
    shapeFuncBySymName[decl.getSymName()] = decl;
  }
}

llvm::ArrayRef<KernelDeclOp>
MojoRegistry::lookupKernel(llvm::StringRef name) const {
  auto it = kernelByName.find(name);
  if (it == kernelByName.end())
    return {};
  return it->second;
}

bool MojoRegistry::containsKernel(llvm::StringRef name) const {
  return kernelByName.contains(name);
}

KernelDeclOp
MojoRegistry::lookupKernelBySymName(llvm::StringRef symName) const {
  auto it = kernelBySymName.find(symName);
  ASSERT_STREAM(it != kernelBySymName.end(),
                "no kernel registered for '" << symName << "'");
  return it->second;
}

PrimitiveDeclOp MojoRegistry::lookupPrimitive(llvm::StringRef name) const {
  auto it = primitiveByName.find(name);
  ASSERT_STREAM(it != primitiveByName.end(),
                "no primitive registered for '" << name << "'");
  return it->second;
}

PrimitiveDeclOp
MojoRegistry::lookupPrimitiveBySymName(llvm::StringRef symName) const {
  auto it = primitiveBySymName.find(symName);
  ASSERT_STREAM(it != primitiveBySymName.end(),
                "no primitive registered for '" << symName << "'");
  return it->second;
}

bool MojoRegistry::containsPrimitive(llvm::StringRef name) const {
  return primitiveByName.contains(name);
}

ShapeFunctionDeclOp
MojoRegistry::lookupShapeFunction(llvm::StringRef name) const {
  auto it = shapeFuncByName.find(name);
  ASSERT_STREAM(it != shapeFuncByName.end(),
                "no shape function registered for '" << name << "'");
  return it->second;
}

ShapeFunctionDeclOp
MojoRegistry::lookupShapeFunctionBySymName(llvm::StringRef symName) const {
  auto it = shapeFuncBySymName.find(symName);
  ASSERT_STREAM(it != shapeFuncBySymName.end(),
                "no shape function registered for sym_name '" << symName
                                                              << "'");
  return it->second;
}

bool MojoRegistry::containsShapeFunction(llvm::StringRef name) const {
  return shapeFuncByName.contains(name);
}

static bool deviceInfoMatches(DeviceInfoAttr kernelInfo,
                              DeviceInfoAttr deviceInfo) {
  auto matches = [](llvm::StringRef kernelField, llvm::StringRef runtimeField) {
    return kernelField.empty() || kernelField == runtimeField;
  };
  return matches(kernelInfo.getLabel(), deviceInfo.getLabel()) &&
         matches(kernelInfo.getApi(), deviceInfo.getApi()) &&
         matches(kernelInfo.getArch(), deviceInfo.getArch()) &&
         matches(kernelInfo.getModel(), deviceInfo.getModel());
}

static int deviceInfoSpecificity(DeviceInfoAttr infos) {
  return !infos.getLabel().empty() + !infos.getApi().empty() +
         !infos.getArch().empty() + !infos.getModel().empty();
}

// Transitional tile-codegen gate (remove per GEX-3941): returns true when the
// kernel's `execute` declares a tile-based fused output
// (_FusedComputeOutputTileTensor). Such a kernel is only lowerable on a device
// configured for tile codegen (tile_based_fusion), so `getResolvedKernel`
// discounts it as a candidate on non-tile devices.
static bool kernelDeclaresTileOutput(KernelDeclOp decl) {
  KernelDeclAdaptor adaptor(decl);
  if (!adaptor.executeFunction.has_value())
    return false;
  for (const auto &arg : adaptor.executeFunction->outputArguments) {
    const auto *tensorArg =
        std::get_if<KernelDeclTensorAdaptor>(&arg.underlyingType);
    if (tensorArg && tensorArg->ioSpec == IOSpec::FusedComputeOutputTileTensor)
      return true;
  }
  return false;
}

ErrorOr<KernelDeclOp>
MojoRegistry::getResolvedKernel(llvm::StringRef name,
                                DeviceRefAttr device) const {
  DeviceInfoAttr deviceInfo = deviceInfoMapping.get(device);
  auto it = kernelByName.find(name);
  ASSERT_STREAM(it != kernelByName.end(),
                "no kernel registered for '" << name << "'");

  KernelDeclOp bestBuiltin;
  KernelDeclOp bestUser;
  int bestBuiltinSpecificity = -1;
  int bestUserSpecificity = -1;

  for (KernelDeclOp decl : it->second) {
    DeviceInfoAttr kernelInfo = decl.getRegistrationDeviceInfo();
    if (!deviceInfoMatches(kernelInfo, deviceInfo))
      continue;
    // TODO(GEX-3941): transitional — discount any candidate whose fused output
    // is tile-based on a device not configured for tile_based_fusion, so the
    // generic (SIMD) registration is selected instead. Remove once the
    // tile-codegen transition completes.
    if (!deviceInfo.getTileBasedFusion() && kernelDeclaresTileOutput(decl))
      continue;
    int specificity = deviceInfoSpecificity(kernelInfo);
    if (decl.getIsUserDefined()) {
      if (specificity > bestUserSpecificity) {
        bestUser = decl;
        bestUserSpecificity = specificity;
      } else if (specificity == bestUserSpecificity && bestUser) {
        ASSERT_STREAM(false, "ambiguous kernel resolution for '"
                                 << name
                                 << "': multiple user-defined registrations "
                                    "match the runtime device with equal "
                                    "specificity");
      }
    } else {
      if (specificity > bestBuiltinSpecificity) {
        bestBuiltin = decl;
        bestBuiltinSpecificity = specificity;
      } else if (specificity == bestBuiltinSpecificity && bestBuiltin) {
        ASSERT_STREAM(false, "ambiguous kernel resolution for '"
                                 << name
                                 << "': multiple built-in registrations "
                                    "match the runtime device with equal "
                                    "specificity");
      }
    }
  }

  auto best = bestUser ? bestUser : bestBuiltin;
  if (!best) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "no kernel matching device info for '" << name << "' on device "
       << mlir::Attribute(deviceInfo) << "; candidates:";
    for (KernelDeclOp decl : it->second)
      os << "\n  - " << mlir::Attribute(decl.getRegistrationDeviceInfo())
         << " (" << decl.getSymName() << ")";
    return Error(os.str());
  }
  return best;
}

DeviceRefAttr MojoRegistry::getDeviceRefForOp(mlir::Operation *op) {
  if (auto hasDevice = mlir::dyn_cast<MO::MOHasDeviceInterface>(op)) {
    auto devices = hasDevice.getExecutionDevices();
    if (!devices.empty())
      return devices[0];
  }
  return getHostDevice(op->getContext());
}

void M::MOGG::copyKernelDecls(mlir::ModuleOp src, mlir::ModuleOp dst) {
  // Idempotent: if `dst` already carries the opaque-type mapping attribute
  // it has already been seeded (e.g. from a prior compile of the same graph
  // module, or from textual MLIR that was exported after a previous seed).
  // Re-cloning the decls would redefine their symbols.
  if (dst->hasAttr(kOpaqueTypeMappingAttrName))
    return;

  mlir::OpBuilder builder(dst.getContext());
  builder.setInsertionPointToEnd(dst.getBody());
  for (auto decl : src.getOps<KernelDeclOp>())
    builder.clone(*decl.getOperation());
  for (auto decl : src.getOps<PrimitiveDeclOp>())
    builder.clone(*decl.getOperation());
  for (auto decl : src.getOps<ShapeFunctionDeclOp>())
    builder.clone(*decl.getOperation());

  // Also carry over the opaque-type mapping so `mogg-rewrite-opaque-types`
  // can run on `dst` without re-loading any packages.
  if (auto mapping = src->getAttrOfType<mlir::DictionaryAttr>(
          kOpaqueTypeMappingAttrName)) {
    dst->setAttr(kOpaqueTypeMappingAttrName, mapping);
  }
}

namespace {

/// Owns the parser state needed to resolve module names and walk the module
/// import directories. One instance serves any number of resolutions, so
/// callers pay the `SharedState` construction cost (config read, dialect
/// preloading) once.
class LITModuleResolver {
public:
  explicit LITModuleResolver(mlir::MLIRContext *ctx)
      : config(ctx, options), sharedState(sourceMgr, config) {}

  /// Resolve `moduleName` to the path of its precompiled binary package.
  ErrorOr<std::string> resolveBinaryPackagePath(llvm::StringRef moduleName) {
    std::optional<KGEN::LIT::SharedState::ModuleSpec> modulePath =
        sharedState.resolveModulePath(moduleName, llvm::SMLoc());
    if (!modulePath)
      return Error(
          llvm::formatv("Failed to resolve module path for `{}`", moduleName)
              .str());
    if (!modulePath->isPrecompiled()) {
      return Error(
          llvm::formatv("Resolved module path for `{}` to source directory "
                        "`{}` but a .mojoc precompiled file was expected "
                        "instead",
                        moduleName, modulePath->path)
              .str());
    }
    return modulePath->path;
  }

  KGEN::LIT::SharedState &getSharedState() { return sharedState; }

private:
  // Order matters: `config` refers to `options`, `sharedState` to both.
  llvm::SourceMgr sourceMgr;
  KGEN::CompilationOptions options;
  KGEN::LIT::ParserConfig config;
  KGEN::LIT::SharedState sharedState;
};

ErrorOr<llvm::SmallVector<std::string>>
resolveBuiltInKernelPackagePaths(LITModuleResolver &resolver) {
  llvm::SmallVector<std::string> paths;
  for (llvm::StringRef moduleName :
       {kMoggKernelsModuleName, kMoggPrimitivesModuleName}) {
    auto pathOr = resolver.resolveBinaryPackagePath(moduleName);
    if (pathOr.isError())
      return pathOr.takeError();
    paths.push_back(*pathOr);
  }
  return paths;
}

} // namespace

ErrorOr<llvm::SmallVector<std::string>>
M::MOGG::getBuiltInKernelPackagePaths(mlir::MLIRContext *ctx) {
  LITModuleResolver resolver(ctx);
  return resolveBuiltInKernelPackagePaths(resolver);
}

ErrorOr<llvm::SmallVector<std::string>>
M::MOGG::getKernelPackagePathsForCacheKey(mlir::MLIRContext *ctx) {
  LITModuleResolver resolver(ctx);

  // The built-in packages are required; fail loudly if they don't resolve.
  auto pathsOr = resolveBuiltInKernelPackagePaths(resolver);
  if (pathsOr.isError())
    return pathsOr.takeError();
  llvm::SmallVector<std::string> paths = std::move(*pathsOr);

  // Collect every binary package visible on the import path; the header
  // comment explains why their contents must reach the cache key.
  llvm::StringSet<> seen;
  resolver.getSharedState().traverseImportDirectories(
      /*importBufferFileId=*/0, [&](llvm::StringRef dir) {
        std::error_code ec;
        for (llvm::sys::fs::directory_iterator it(dir, ec), e; it != e && !ec;
             it.increment(ec)) {
          const std::string &path = it->path();
          if (M::Filesystem::isMojoBinaryPackagePath(path))
            seen.insert(path);
        }
        return mlir::WalkResult::advance();
      });
  // Append the collected packages, minus the built-ins already in `paths`,
  // sorted for a deterministic order independent of directory iteration.
  llvm::SmallVector<std::string> found;
  for (const auto &entry : seen)
    if (!llvm::is_contained(paths, entry.getKey()))
      found.push_back(entry.getKey().str());
  llvm::sort(found);
  llvm::append_range(paths, found);
  return paths;
}

ErrorOrSuccess M::MOGG::verifyCustomOp(const MojoRegistry &registry,
                                       mlir::Operation *op) {
  // Map operation to its registered Mojo kernel name + function label.
  MojoOperationInfo info = getMojoOperationInfo(op);
  if (info.operationName.empty())
    return Error(llvm::formatv("operation '{}' has no registered Mojo kernel",
                               op->getName().getStringRef())
                     .str());

  if (!registry.containsKernel(info.operationName)) {
    return Error(
        llvm::formatv("no kernel registered for '{}'", info.operationName)
            .str());
  }

  // Decode the kernel signature and check that operand types line up. This
  // catches mismatches between MO operand types and the kernel's argument
  // types before lowering attempts to emit Mojo.
  DeviceRefAttr device = MojoRegistry::getDeviceRefForOp(op);
  auto declOpOr = registry.getResolvedKernel(info.operationName, device);
  if (declOpOr.isError())
    return declOpOr.takeError();
  KernelDeclAdaptor adaptor(*declOpOr);
  if (!adaptor.executeFunction)
    return Error(
        llvm::formatv("kernel '{}' has no execute function", info.operationName)
            .str());
  auto mappingOr = mapMOOpToKernelImplementation(op, *adaptor.executeFunction);
  if (mappingOr.isError())
    return mappingOr.takeError();

  // For mo.custom, additionally check that user-supplied params match the
  // kernel's compile-time parameter list (struct + fn-level).
  if (auto custom = mlir::dyn_cast<MO::CustomOp>(op))
    return validateCustomOpParameters(registry, custom);

  return success();
}

ErrorOrSuccess M::MOGG::verifyCustomOp(mlir::ModuleOp kernelDecls,
                                       mlir::Operation *op) {
  MojoRegistry registry(kernelDecls);
  return verifyCustomOp(registry, op);
}
