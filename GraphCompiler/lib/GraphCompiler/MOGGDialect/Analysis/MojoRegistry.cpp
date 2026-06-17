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

static bool deviceInfoMatches(DeviceInfoAttr kernelDeviceInfo,
                              DeviceInfoAttr targetDeviceInfo) {
  auto matches = [](llvm::StringRef kernelField, llvm::StringRef runtimeField) {
    return kernelField.empty() || kernelField == runtimeField;
  };
  return matches(kernelDeviceInfo.getLabel(), targetDeviceInfo.getLabel()) &&
         matches(kernelDeviceInfo.getApi(), targetDeviceInfo.getApi()) &&
         matches(kernelDeviceInfo.getArch(), targetDeviceInfo.getArch()) &&
         matches(kernelDeviceInfo.getModel(), targetDeviceInfo.getModel());
}

static int deviceInfoSpecificity(DeviceInfoAttr infos) {
  // Each field outweighs all less-specific fields combined, so that e.g.
  // arch-only (4) beats label+api (1+2=3). Order: label < api < arch < model.
  return (!infos.getLabel().empty() ? 1 : 0) +
         (!infos.getApi().empty() ? 2 : 0) +
         (!infos.getArch().empty() ? 4 : 0) +
         (!infos.getModel().empty() ? 8 : 0);
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
    int specificity = deviceInfoSpecificity(kernelInfo);
    if (decl.getIsUserDefined()) {
      if (specificity > bestUserSpecificity) {
        bestUser = decl;
        bestUserSpecificity = specificity;
      }
    } else {
      if (specificity > bestBuiltinSpecificity) {
        bestBuiltin = decl;
        bestBuiltinSpecificity = specificity;
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

ErrorOr<std::string> resolveLITModulePath(mlir::MLIRContext *ctx,
                                          llvm::SourceMgr &sourceMgr,
                                          llvm::StringRef moduleName) {
  KGEN::CompilationOptions options;
  KGEN::LIT::ParserConfig config(ctx, options);
  KGEN::LIT::SharedState sharedState(sourceMgr, config);
  std::optional<std::string> modulePath =
      sharedState.resolveModulePath(moduleName, llvm::SMLoc());
  if (!modulePath)
    return Error(
        llvm::formatv("Failed to resolve module path for `{}`", moduleName)
            .str());
  if (!M::Filesystem::isMojoBinaryPackagePath(*modulePath))
    return Error(
        llvm::formatv("Resolved module path for `{}` to source directory `{}` "
                      "but a .mojoc precompiled file was expected instead",
                      moduleName, *modulePath)
            .str());
  return *modulePath;
}

} // namespace

ErrorOr<llvm::SmallVector<std::string>>
M::MOGG::getBuiltInKernelPackagePaths(mlir::MLIRContext *ctx) {
  llvm::SourceMgr sourceMgr;
  llvm::SmallVector<std::string> paths;
  auto kernelAPIPathOr =
      resolveLITModulePath(ctx, sourceMgr, kMoggKernelsModuleName);
  if (kernelAPIPathOr.isError())
    return kernelAPIPathOr.takeError();
  paths.push_back(*kernelAPIPathOr);
  auto primitivesPathOr =
      resolveLITModulePath(ctx, sourceMgr, kMoggPrimitivesModuleName);
  if (primitivesPathOr.isError())
    return primitivesPathOr.takeError();
  paths.push_back(*primitivesPathOr);
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
