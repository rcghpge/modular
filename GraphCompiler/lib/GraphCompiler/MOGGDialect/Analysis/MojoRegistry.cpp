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

MojoRegistry::MojoRegistry(mlir::ModuleOp module) {
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

KernelDeclOp MojoRegistry::getResolvedKernel(llvm::StringRef name) const {
  auto it = kernelByName.find(name);
  ASSERT_STREAM(it != kernelByName.end(),
                "no kernel registered for '" << name << "'");

  KernelDeclOp firstBuiltin;
  KernelDeclOp userDecl;
  for (KernelDeclOp decl : it->second) {
    if (decl.getIsUserDefined()) {
      ASSERT_STREAM(!userDecl,
                    "kernel '" << name
                               << "' is registered in multiple user packages; "
                                  "ImportPackages dedup should have caught "
                                  "this before the registry was built");
      userDecl = decl;
    } else if (!firstBuiltin) {
      firstBuiltin = decl;
    }
  }

  return userDecl ? userDecl : firstBuiltin;
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
  KernelDeclAdaptor adaptor(registry.getResolvedKernel(info.operationName));
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
