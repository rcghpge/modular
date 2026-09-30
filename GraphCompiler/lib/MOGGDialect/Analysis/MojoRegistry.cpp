//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MojoParameters.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoParser/ModuleLoader.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/AssertStream.h"
#include "Support/Filesystem/Paths.h"
#include "llvm/ADT/STLExtras.h"
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
  KernelDeclOp decl = lookupKernelBySymNameOrNull(symName);
  ASSERT_STREAM(decl, "no kernel registered for '" << symName << "'");
  return decl;
}

KernelDeclOp
MojoRegistry::lookupKernelBySymNameOrNull(llvm::StringRef symName) const {
  return kernelBySymName.lookup(symName);
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
  if (it == shapeFuncByName.end())
    return {};
  return it->second;
}

ShapeFunctionDeclOp
MojoRegistry::lookupShapeFunctionBySymName(llvm::StringRef symName) const {
  ShapeFunctionDeclOp decl = lookupShapeFunctionBySymNameOrNull(symName);
  ASSERT_STREAM(decl, "no shape function registered for sym_name '" << symName
                                                                    << "'");
  return decl;
}

ShapeFunctionDeclOp MojoRegistry::lookupShapeFunctionBySymNameOrNull(
    llvm::StringRef symName) const {
  return shapeFuncBySymName.lookup(symName);
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
// (_FusedComputeOutputTileTensor or _FusedOutputTileTensor). Such a kernel is
// only lowerable on a device configured for tile codegen (tile_based_fusion),
// so `getResolvedKernel` discounts it as a candidate on non-tile devices.
static bool kernelDeclaresTileOutput(KernelDeclOp decl) {
  KernelDeclAdaptor adaptor(decl);
  if (!adaptor.executeFunction.has_value())
    return false;
  for (const auto &arg : adaptor.executeFunction->outputArguments) {
    const auto *tensorArg =
        std::get_if<KernelDeclTensorAdaptor>(&arg.underlyingType);
    if (tensorArg &&
        (tensorArg->ioSpec == IOSpec::FusedComputeOutputTileTensor ||
         tensorArg->ioSpec == IOSpec::FusedOutputTileTensor))
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
    auto modulePath = sharedState.getModuleLoader().resolveModulePath(
        moduleName, llvm::SMLoc());
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
  resolver.getSharedState().getModuleLoader().traverseImportDirectories(
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

namespace M::MOGG {

// ---------------------------------------------------------------------------
// Custom op parameter validation using KernelDeclOp
// ---------------------------------------------------------------------------

static bool isStaticStringLoweredPattern(Type ty) {
  auto structTy = dyn_cast<KGEN::StructType>(ty);
  if (!structTy || structTy.getIsParamPack())
    return false;

  std::optional<SmallVector<Type>> elementTypes = structTy.getElementTypes();
  if (!elementTypes || elementTypes->size() != 2)
    return false;

  auto ptrTy = dyn_cast<KGEN::PointerType>((*elementTypes)[0]);
  if (!ptrTy || !isa<KGEN::NoneType>(ptrTy.getElementType()))
    return false;

  return isa<IndexType>((*elementTypes)[1]);
}

static bool isLITStaticStringType(Type ty) {
  auto structTy = KGEN::sugarDynCast<KGEN::LIT::StructType>(ty);
  if (!structTy) {
    if (auto ref = KGEN::sugarDynCast<KGEN::LIT::RefType>(ty)) {
      structTy =
          KGEN::sugarDynCast<KGEN::LIT::StructType>(ref.getElementType());
    }
  }
  if (!structTy)
    return false;

  auto symbol = structTy.getSymbol();
  std::string symbolName =
      llvm::formatv("{}::{}", symbol.getRootReference().strref(),
                    symbol.getLeafReference().strref())
          .str();
  if (symbolName != kMojoInternalStaticStringTypeName)
    return false;

  auto paramValues = structTy.getParamValues();
  if (paramValues.empty())
    return false;

  TypedAttr originAttr = paramValues[0];
  if (KGEN::sugarIsa<KGEN::ParamDeclRefAttr>(originAttr))
    return false;

  return true;
}

static StringAttr maybeDemangleName(StringAttr name) {
  StringRef demangled = KGEN::demangleParameterName(name);
  if (demangled != name)
    return StringAttr::get(name.getContext(), demangled);
  return name;
}

/// Return the "root::leaf" name of a parameter's declared struct type, or
/// null if the type is not a recognisable struct (e.g. primitive scalars).
/// Used by `validateCustomOpParameters` to dispatch on Mojo class names
/// like `std::StaticString` vs `std::StringSpan` — information not carried
/// by the bare KGEN type alone.
static StringAttr getParamTypeName(MLIRContext *ctx, Type paramType) {
  if (auto ref = KGEN::sugarDynCast<KGEN::LIT::RefType>(paramType))
    paramType = ref.getElementType();
  auto structTy = KGEN::sugarDynCast<KGEN::LIT::StructType>(paramType);
  if (!structTy)
    return nullptr;
  auto sym = structTy.getSymbol();
  return StringAttr::get(ctx, llvm::formatv("{}::{}",
                                            sym.getRootReference().strref(),
                                            sym.getLeafReference().strref())
                                  .str());
}

static ErrorOrSuccess validateCustomOpParameters(const MojoRegistry &registry,
                                                 MO::CustomOp customOp) {
  if (!registry.containsKernel(customOp.getSymbol())) {
    return Error(
        llvm::formatv("no kernel registered for '{}'", customOp.getSymbol())
            .str());
  }
  auto declOpOr =
      registry.getResolvedKernel(customOp.getSymbol(), customOp.getDevice());
  if (declOpOr.isError())
    return declOpOr.takeError();
  KernelDeclOp declOp = *declOpOr;
  // The user-supplied `parameters` dict on `mo.custom` can target struct-level
  // compile-time params (e.g. `[interleaved: Bool]`) or any fn-level params
  // declared on the registered functions. Collect struct params first to
  // mirror Mojo's merged `@register` parameter layout, then union the param
  // lists from each function variant (some kernels declare params only on
  // `elementwise` instead of `execute`).
  SmallVector<KGEN::ParamDeclAttr> inputParamsVec;
  llvm::SmallDenseSet<StringRef> seen;
  auto append = [&](KGEN::ParamDeclAttr p) {
    if (seen.insert(p.getName().strref()).second)
      inputParamsVec.push_back(p);
  };
  for (auto a : declOp.getStructParams())
    append(cast<KGEN::ParamDeclAttr>(a));
  auto appendFn = [&](std::optional<MOGG::KernelFunctionAttr> fn) {
    if (!fn)
      return;
    for (auto a : fn->getParams())
      append(cast<KGEN::ParamDeclAttr>(a));
  };
  appendFn(declOp.getExecute());
  appendFn(declOp.getElementwise());
  if (registry.containsShapeFunction(declOp.getRegistration()))
    appendFn(
        registry.lookupShapeFunction(declOp.getRegistration()).getFunction());
  appendFn(declOp.getUpdateView());
  ArrayRef<KGEN::ParamDeclAttr> inputParams = inputParamsVec;

  struct MojoParameterAdaptor {
    KGEN::ParamDeclAttr attr;
    StringAttr typeName;
  };
  llvm::DenseMap<StringAttr, MojoParameterAdaptor> paramDecls;

  // Compute the param type-name on demand from each ParamDeclAttr instead of
  // caching strings on KernelDeclOp.
  MLIRContext *ctx = customOp.getContext();
  for (auto param : inputParams)
    paramDecls[maybeDemangleName(param.getName())] = {
        param, getParamTypeName(ctx, param.getType())};

  for (const NamedAttribute &namedAttr : customOp.getParameters()) {
    auto it = paramDecls.find(namedAttr.getName());
    if (it == paramDecls.end()) {
      return Error(llvm::formatv(
          "parameter '{}' supplied to custom op '{}' does not exist in the "
          "corresponding Mojo code",
          namedAttr.getName().strref(), customOp.getSymbol()));
    }

    auto [customParamDecl, typeName] = it->second;
    Attribute mlirParamAttr = namedAttr.getValue();

    Type kgenParamTy = customParamDecl.getType();
    auto asIntegerType = dyn_cast<IntegerType>(kgenParamTy);

    if (isa<IndexType>(kgenParamTy)) {
      auto asInt = dyn_cast<IntegerAttr>(mlirParamAttr);
      if (asInt == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected an integer type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (asIntegerType && asIntegerType.getWidth() == 1) {
      auto asBool = dyn_cast<BoolAttr>(mlirParamAttr);
      if (asBool == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected an boolean type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (isa<KGEN::StringType>(kgenParamTy)) {
      if (typeName && typeName == MOGG::kMojoInternalStaticStringTypeName) {
        [[maybe_unused]] auto asString = dyn_cast<StringAttr>(mlirParamAttr);
        ASSERT_STREAM(asString != nullptr,
                      "failed to convert custom op attribute "
                          << mlirParamAttr
                          << " to expected mojo kernel param type "
                          << kgenParamTy << ": expected string type");
      } else if (typeName) {
        return Error(llvm::formatv(
            "in custom op '{}' attribute '{}' uses a string-like type that is "
            "not supported. Use type StaticString instead to define a string "
            "parameter for a custom op.",
            customOp.getSymbol(), customParamDecl.getName()));
      } else {
        [[maybe_unused]] auto asString = dyn_cast<StringAttr>(mlirParamAttr);
        ASSERT_STREAM(asString != nullptr,
                      "failed to convert custom op attribute "
                          << mlirParamAttr
                          << " to expected mojo kernel param type "
                          << kgenParamTy << ": expected string type");
      }
    } else if (isa<KGEN::DTypeType>(kgenParamTy)) {
      auto asType = dyn_cast<TypeAttr>(mlirParamAttr);
      if (asType == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected a DType",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName && typeName == MOGG::kMojoDtypeTypeName) {
      auto asType = dyn_cast<TypeAttr>(mlirParamAttr);
      if (asType == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected a DType",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName && typeName == MOGG::kMojoStaticStringTypeName) {
      auto asString = dyn_cast<StringAttr>(mlirParamAttr);
      if (asString == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected string type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName &&
               typeName == MOGG::kMojoInternalStaticStringTypeName) {
      bool isStaticString = isLITStaticStringType(kgenParamTy) ||
                            isStaticStringLoweredPattern(kgenParamTy);
      if (isStaticString) {
        auto asString = dyn_cast<StringAttr>(mlirParamAttr);
        if (asString == nullptr) {
          return Error(llvm::formatv(
              "failed to convert custom op '{}' attribute '{}' with value "
              "'{}' to the proper Mojo parameter type: expected string type",
              customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
        }
      } else {
        return Error(llvm::formatv(
            "in custom op '{}' attribute '{}' uses type StringSpan. Use type "
            "StaticString instead to define a string parameter for a custom "
            "op.",
            customOp.getSymbol(), customParamDecl.getName()));
      }
    } else if (typeName && (typeName == MOGG::kMojoInternalDpsSimdTypeName)) {
      auto asInt = dyn_cast<IntegerAttr>(mlirParamAttr);
      if (asInt == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected an integer type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName && typeName == MOGG::kMojoInternalDpsBoolTypeName) {
      auto asBool = dyn_cast<BoolAttr>(mlirParamAttr);
      if (asBool == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected a boolean type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName &&
               (typeName == MOGG::kMojoInternalStringSliceTypeName ||
                typeName == MOGG::kMojoStringTypeName)) {
      std::string userTypeName =
          typeName == MOGG::kMojoInternalStringSliceTypeName ? "StringSpan"
                                                             : "String";
      return Error(llvm::formatv(
          "in custom op '{}' attribute '{}' uses type {}. Use type "
          "StaticString instead to define a string parameter for a custom op.",
          customOp.getSymbol(), customParamDecl.getName(), userTypeName));
    } else if (!typeName && isStaticStringLoweredPattern(kgenParamTy)) {
      auto asString = dyn_cast<StringAttr>(mlirParamAttr);
      if (asString == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected string type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else {
      return Error(llvm::formatv(
          "failed to convert custom op '{}' attribute '{}' with value '{}' to "
          "the proper Mojo parameter type: no type conversion found to type "
          "'{}'",
          customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr,
          kgenParamTy));
    }
  }

  return success();
}

} // namespace M::MOGG

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
