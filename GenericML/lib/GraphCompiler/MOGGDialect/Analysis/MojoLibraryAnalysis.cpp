//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/OperandMapper.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/MOGGPreElab/MojoKernelAdaptor.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "Support/AssertStream.h"
#include "Support/Context.h"
#include "Support/Filesystem/Paths.h"
#include "Support/HashUtils.h"

#include <type_traits>

using namespace M;
using namespace MOGG;

// TODO(GEX-2488): The following are Mojo module names hardcoded for
// convenience. We should determine them programmatically instead.
constexpr StringLiteral MOGGKernelAPIModuleName = "MOGGKernelAPI";
constexpr StringLiteral MOGGPrimitivesModuleName = "MOGGPrimitives";

[[maybe_unused]] static ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getFile(llvm::StringRef filepath) {
  auto memBufferOr = toModularErrorOr(
      llvm::MemoryBuffer::getFile(filepath, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false));
  if (memBufferOr.isError())
    return Error(memBufferOr.getError() + Twine(" [" + filepath + "]"));

  return memBufferOr;
}

static ErrorOr<OwningOpRef<ModuleOp>>
parseMojoPackage(MLIRContext *ctx, StringRef path,
                 const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                 bool generateKGENLib) {
  AsyncRT::Runtime &runtime = *loadContext(ctx)->get<AsyncRT::Runtime>();

  // Intercept parsing diagnostics to display them at the end if an error occurs
  std::string errMsg;
  llvm::raw_string_ostream errMsgStream(errMsg);

  auto oldHandler = sourceMgr->getDiagHandler();
  auto oldCtx = sourceMgr->getDiagContext();

  // Intercept error messages to display them at the end for nice UX.
  sourceMgr->setDiagHandler(
      [](const llvm::SMDiagnostic &diag, void *context) {
        llvm::raw_string_ostream *errMsgStream =
            static_cast<llvm::raw_string_ostream *>(context);

        *errMsgStream << "\t" << diag.getMessage();
      },
      &errMsgStream);
  OwningOpRef<ModuleOp> ownedModule =
      KGEN::LIT::importStandaloneMojoBinaryPackage(runtime, sourceMgr, ctx,
                                                   path);
  if (!ownedModule)
    return Error("failed to load Mojo package: " + path +
                 "\nMojo Parser Trace:\n" + errMsg);
  else if (!errMsg.empty())
    llvm::errs() << errMsg << "\n";

  // Reset the old state of the diagnostic handler
  sourceMgr->setDiagHandler(oldHandler, oldCtx);

  if (!generateKGENLib)
    return ownedModule;

  // Generate a library from the package, ensuring that dependencies are
  // materialized to make it self contained.
  KGEN::KGENCompiler compiler(*(ownedModule->getContext()),
                              KGEN::CompilationOptions());

  // Show the mojo code when there is an error.
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, ctx);

  if (auto err = compiler.runGenerateLibraryPipeline(*ownedModule))
    return err.takeError();

  return ownedModule;
}

static ErrorOr<std::string> resolveLITModulePath(MLIRContext *ctx,
                                                 llvm::SourceMgr &sourceMgr,
                                                 StringRef moduleName) {
  KGEN::CompilationOptions options;
  KGEN::LIT::ParserConfig config(ctx, options);
  KGEN::LIT::SharedState sharedState(sourceMgr, config);
  std::optional<std::string> modulePath =
      sharedState.resolveModulePath(moduleName, SMLoc());
  if (!modulePath.has_value())
    return Error(
        llvm::formatv("Failed to resolve module path for `{}`", moduleName));
  if (!Filesystem::isMojoBinaryPackagePath(*modulePath))
    return Error(
        llvm::formatv("Managed to resolve module path for `{}` to source "
                      "directory `{}` but a .mojopkg file was expected instead",
                      moduleName, *modulePath));
  return *modulePath;
}

ErrorOr<SmallVector<std::string>>
MojoLibraryAnalysis::getBuiltInKernelPackagePaths(MLIRContext *ctx) {
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  SmallVector<std::string> paths;
  auto moggKernelAPIPkgPathOr =
      resolveLITModulePath(ctx, *sourceMgr, MOGGKernelAPIModuleName);
  if (moggKernelAPIPkgPathOr.isError())
    return moggKernelAPIPkgPathOr.takeError();
  paths.push_back(*moggKernelAPIPkgPathOr);
  auto moggPrimitivesPathOr =
      resolveLITModulePath(ctx, *sourceMgr, MOGGPrimitivesModuleName);
  if (moggPrimitivesPathOr.isError())
    return moggPrimitivesPathOr.takeError();
  paths.push_back(*moggPrimitivesPathOr);
  return paths;
}

KGEN::LIT::FnOp MojoLibraryAnalysis::getRegisteredLITMojoFunctionFor(
    StringRef name, StringRef functionLabel) const {
  auto registeredFunctions = registeredLITKernelFunctionSymbols.find(name);
  if (registeredFunctions == registeredLITKernelFunctionSymbols.end())
    return {};

  auto registeredFunction = registeredFunctions->second.find(functionLabel);
  if (registeredFunction == registeredFunctions->second.end())
    return {};

  return registeredFunction->getValue();
}

ErrorOrSuccess
MojoLibraryAnalysis::loadAllLITKernels(ArrayRef<std::string> paths) {
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();

  auto loadModule = [&](bool isMOGGKernelAPI,
                        StringRef path) -> ErrorOrSuccess {
    ErrorOr<OwningOpRef<ModuleOp>> module = parseMojoPackage(
        getContext(), path, sourceMgr, /*generateKGENLib=*/false);

    if (module.isError())
      return module.takeError();

    if (isMOGGKernelAPI)
      moggKernelAPILITPackage = std::move(*module);
    else
      litPackages.push_back(std::move(*module));
    return success();
  };

  // Load MOGGkernelAPI
  auto moggKernelAPIPkgPathOr =
      resolveLITModulePath(getContext(), *sourceMgr, MOGGKernelAPIModuleName);
  if (moggKernelAPIPkgPathOr.isError())
    return moggKernelAPIPkgPathOr.takeError();
  if (auto err = loadModule(/*isMOGGKernelAPI=*/true, *moggKernelAPIPkgPathOr))
    return err.takeError();

  // Load MOGGPrimitives
  auto moggPrimitivesPathOr =
      resolveLITModulePath(getContext(), *sourceMgr, MOGGPrimitivesModuleName);
  if (moggPrimitivesPathOr.isError())
    return moggPrimitivesPathOr.takeError();
  if (auto err = loadModule(/*isMOGGKernelAPI=*/false, *moggPrimitivesPathOr))
    return err.takeError();

  // Load custom libraries.
  for (const auto &path : paths) {
    auto errOr = loadModule(/*isMOGGKernelAPI=*/false, path);
    if (errOr.isError())
      return errOr.takeError();
  }

  return success();
}

ErrorOrSuccess
MojoLibraryAnalysis::addKernelLibraryPath(StringRef kernelLibraryPath) {
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  auto moduleOr = parseMojoPackage(getContext(), kernelLibraryPath, sourceMgr,
                                   /*generateKGENLib=*/false);
  if (moduleOr.isError())
    return moduleOr.takeError();
  auto module = moduleOr.takeValue();
  ErrorOrSuccess ret;
  module->walk([&](KGEN::LIT::FnOp funcOp) {
    ret = updateRegisteredKernelsMap(funcOp, false);
    if (ret.isError())
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::skip();
  });

  if (ret.isError())
    return ret;
  return registerOpaqueTypesFromModule(*module);
}

// If the generator is from the standard library and it has a default priority
// level, then we want to give it a higher priority, because it should be used
// instead of a pre-built kernel library. This allows the user to modify the
// open source standard library code and have it used by the graph compiler.
static bool shouldIncreasePriority(bool isStandardLibrary, int priority) {
  return isStandardLibrary && priority == -1;
}

ErrorOrSuccess MojoLibraryAnalysis::updateRegisteredKernelsMap(
    KGEN::LIT::FnOp func, bool isGeneratorFromStandardLibrary) {
  auto registerKernelFunction = [&](StringRef name, StringRef label,
                                    bool override = true) {
    auto [it, _] = registeredLITKernelFunctionSymbols.try_emplace(
        name, llvm::StringMap<KGEN::LIT::FnOp>{});

    auto [it2, inserted] = it->second.try_emplace(label, func);
    if (!inserted && override)
      it2->second = func;
  };

  auto registerKernelCandidate = [&](StringRef name,
                                     int64_t priority) -> ErrorOrSuccess {
    if (shouldIncreasePriority(isGeneratorFromStandardLibrary, priority))
      priority = 1;

    auto [it, inserted] = registeredLITMojoKernelSymbol.try_emplace(
        name, LITKernelCandidate(func, priority));
    if (inserted) {
      // First time we find the kernel, just register it.
      return success();
    }

    // Returns an error if two kernels have the same priority.
    int64_t oldPriority = it->getValue().priority;
    if (oldPriority == priority) {
      auto diag = func->emitError() << "override of operation named " << name
                                    << " has multiple candidates";
      diag.attachNote(it->getValue().fun.getLoc())
          << "override with priority " << priority;
      return Error("Operation overriding error");
    }

    // Replace it if the new kernel has an higher priority.
    if (priority > oldPriority)
      it->getValue() = LITKernelCandidate(func, priority);
    return success();
  };

  // Detect foreach kernel, treat it specially.
  if (func->hasAttr(KGEN::MOGGPreElab::MOGG_INTRINSIC_ELEMWISE_FOR_EACH)) {
    if (!litForeachKernel) {
      litForeachKernel = func;
    } else if (litForeachKernel.getLoc() != func.getLoc()) {
      if (isGeneratorFromStandardLibrary)
        litForeachKernel = func;
      else {
        return Error(
            llvm::formatv("Multiple elementwise foreach kernels detected.\n"
                          "  Previous: {0} at {1}\n"
                          "  New: {2} at {3}",
                          litForeachKernel.getSymName(),
                          litForeachKernel.getLoc(), func.getSymName(),
                          func.getLoc())
                .str());
      }
    }
  }

  if (func->hasAttr(KGEN::MOGGPreElab::MOGG_INTRINSIC_VIEW_MATERIALIZE)) {
    if (!litViewMaterializeKernel) {
      litViewMaterializeKernel = func;
    } else if (litViewMaterializeKernel.getLoc() != func.getLoc()) {
      if (isGeneratorFromStandardLibrary)
        litViewMaterializeKernel = func;
      else {
        return Error(
            llvm::formatv("Multiple materialize view kernels detected.\n"
                          "  Previous: {0} at {1}\n"
                          "  New: {2} at {3}",
                          litViewMaterializeKernel.getSymName(),
                          litViewMaterializeKernel.getLoc(), func.getSymName(),
                          func.getLoc())
                .str());
      }
    }
  }

  // Skipping functions that aren't exported.
  if (!func.isExported())
    return success();

  // Check if the generator has a the registration attribute directly
  // attached.
  if (auto regs = func->getAttrOfType<ArrayAttr>(
          KGEN::MOGGPreElab::kernelRegistrationAttr)) {
    ArrayRef<Attribute> kernels = regs.getValue();
    // Each kernel is stored as a pair of name / priority.
    for (size_t i = 0; i < kernels.size(); i += 2) {
      auto asString = cast<StringAttr>(kernels[i]).strref();
      auto intPriority = cast<IntegerAttr>(kernels[i + 1]);
      if (auto registerErr =
              registerKernelCandidate(asString, intPriority.getInt()))
        return registerErr;
    }
  }
  if (auto kernelName = func->getAttrOfType<StringAttr>(
          KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel)) {
    if (auto registerErr = registerKernelCandidate(kernelName, /*priority=*/0))
      return registerErr;
    registerKernelFunction(kernelName,
                           KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel);
  }
  if (auto name = func->getAttrOfType<StringAttr>(
          KGEN::MOGGPreElab::kMOGGShapeFunctionLabel)) {
    registerKernelFunction(name, KGEN::MOGGPreElab::kMOGGShapeFunctionLabel);
  }
  if (auto name = func->getAttrOfType<StringAttr>(
          KGEN::MOGGPreElab::kMOGGUpdateViewFunctionLabel)) {
    registerKernelFunction(name,
                           KGEN::MOGGPreElab::kMOGGUpdateViewFunctionLabel);
  }
  if (auto name = func->getAttrOfType<StringAttr>(
          KGEN::MOGGPreElab::kMOGGElementwiseFunctionLabel)) {
    registerKernelFunction(name,
                           KGEN::MOGGPreElab::kMOGGElementwiseFunctionLabel);
  }
  return success();
}

// TODO(GEX-2899): Consider moving this opaque type collection logic to the
// MOGG PreElab pass (mogg-annotate-kernels). The PreElab pass already walks
// kernel functions and annotates them with metadata, so it might be more
// appropriate to collect and validate opaque types there rather than during
// MojoLibraryAnalysis init. This would keep all kernel-related preprocessing
// in one place.

/// Recursively collect all struct type symbols from a type.
/// This handles nested types like pointers, tuples, etc.
static void collectStructTypesFromType(Type type,
                                       llvm::DenseSet<SymbolRefAttr> &symbols) {
  if (auto structType = dyn_cast<KGEN::LIT::StructType>(type)) {
    symbols.insert(structType.getSymbol());
    // Also recurse into parameter values in case they contain struct types.
    for (auto paramValue : structType.getParamValues()) {
      if (auto typeAttr = dyn_cast<TypeAttr>(paramValue))
        collectStructTypesFromType(typeAttr.getValue(), symbols);
    }
  } else if (auto refType = dyn_cast<KGEN::LIT::RefType>(type)) {
    collectStructTypesFromType(refType.getElementType(), symbols);
  } else if (auto ptrType = dyn_cast<KGEN::PointerType>(type)) {
    collectStructTypesFromType(ptrType.getElementType(), symbols);
  } else if (auto tupleType = dyn_cast<TupleType>(type)) {
    for (auto elementType : tupleType.getTypes())
      collectStructTypesFromType(elementType, symbols);
  }
  // Add more type unwrapping as needed for other container types.
}

/// Collect all struct type symbols that appear in kernel function arguments.
static void
collectKernelArgStructTypes(ModuleOp module,
                            llvm::DenseSet<SymbolRefAttr> &symbols) {
  module->walk([&](KGEN::LIT::FnOp funcOp) {
    // Only consider kernel functions (those with mogg.* labels).
    if (!KGEN::MOGGPreElab::isExtensibilityFunc(funcOp))
      return;

    // Collect struct types from all argument types.
    FunctionType funcType = funcOp.getFunctionType();
    for (Type argType : funcType.getInputs())
      collectStructTypesFromType(argType, symbols);

    // Also collect from result types.
    for (Type resultType : funcType.getResults())
      collectStructTypesFromType(resultType, symbols);
  });
}

/// Register opaque type names from struct declarations in a module.
/// Only registers structs that are actually used in kernel function arguments.
/// This builds a mapping from simple type names (e.g., "SIMDPair") to their
/// fully qualified Mojo names (e.g., "MOGGKernelAPITest.SIMDPair").
ErrorOrSuccess
MojoLibraryAnalysis::registerOpaqueTypesFromModule(ModuleOp module) {
  // First, collect all struct types used in kernel function arguments.
  llvm::DenseSet<SymbolRefAttr> kernelArgStructSymbols;
  collectKernelArgStructTypes(module, kernelArgStructSymbols);

  std::optional<Error> conflictError;

  module->walk([&](KGEN::LIT::StructDeclOp structOp) {
    // Get the fully qualified symbol reference.
    SymbolRefAttr symbolRef = KGEN::LIT::getFullyResolvedSymbolRef(structOp);

    // Only register structs that are used in kernel function arguments.
    if (!kernelArgStructSymbols.contains(symbolRef))
      return WalkResult::advance();

    // Convert symbol ref to Mojo-style dot-separated name.
    // Symbol refs use :: as separator, Mojo uses dots.
    std::string fullyQualifiedName;
    llvm::raw_string_ostream os(fullyQualifiedName);
    os << symbolRef.getRootReference().getValue();
    for (auto nested : symbolRef.getNestedReferences())
      os << "." << nested.getValue();

    stripInitModulePrefix(fullyQualifiedName);

    // The simple name is the struct's own name (last component).
    StringRef simpleName = structOp.getSymName();

    auto it = opaqueTypeNameToFullyQualified.find(simpleName);
    if (it == opaqueTypeNameToFullyQualified.end()) {
      // No conflict - register the type.
      opaqueTypeNameToFullyQualified.try_emplace(simpleName,
                                                 std::move(fullyQualifiedName));
      return WalkResult::advance();
    }

    // Same fully qualified name means same type - not a conflict.
    if (it->second == fullyQualifiedName)
      return WalkResult::advance();

    // Conflict detected - multiple packages define different types with the
    // same simple name. Users must use fully qualified names to disambiguate.
    conflictError =
        Error("Opaque type '" + simpleName.str() + "' ('" + fullyQualifiedName +
              "') conflicts with existing type '" + it->second +
              "'. Use fully qualified type names to disambiguate.");
    return WalkResult::interrupt();
  });

  if (conflictError)
    return std::move(*conflictError);
  return success();
}

ErrorOrSuccess
MojoLibraryAnalysis::init(ArrayRef<std::string> kernelLibraryPaths) {
  // The Mojo path is always used now, so we always load LIT kernels.
  auto loadLitKernelsOr = loadAllLITKernels(kernelLibraryPaths);
  if (loadLitKernelsOr.isError())
    return loadLitKernelsOr.takeError();

  ErrorOrSuccess maybeError = success();

  auto registerLITFunctions = [&](ModuleOp package,
                                  bool isGeneratorFromStandardLibrary) {
    package->walk([&](KGEN::LIT::FnOp funcOp) {
      if (auto updateErr = updateRegisteredKernelsMap(
              funcOp, isGeneratorFromStandardLibrary))
        maybeError = updateErr.takeError();
    });
  };

  // Register all functions in the LIT packages
  registerLITFunctions(*moggKernelAPILITPackage,
                       /*isGeneratorFromStandardLibrary=*/true);
  for (auto &package : litPackages)
    registerLITFunctions(*package, /*isGeneratorFromStandardLibrary=*/false);

  if (maybeError.isError())
    return maybeError.takeError();

  // Register all struct types for opaque type name resolution.
  // MOGGKernelAPI is registered first, then other packages.
  // Conflicts emit a warning and keep the first registered type.
  if (auto err = registerOpaqueTypesFromModule(*moggKernelAPILITPackage))
    return err.takeError();
  for (auto &package : litPackages) {
    if (auto err = registerOpaqueTypesFromModule(*package))
      return err.takeError();
  }

  return success();
}

MojoLibraryAnalysis::MojoLibraryAnalysis(MLIRContext *ctx)
    : config(ctx, /*verifyAfterParse=*/false) {}

MojoLibraryAnalysis::~MojoLibraryAnalysis() = default;

KGEN::LIT::FnOp
MojoLibraryAnalysis::getRegisteredLITMojoKernelFor(StringRef name) const {
  auto registeredFunction = registeredLITMojoKernelSymbol.find(name);
  if (registeredFunction == registeredLITMojoKernelSymbol.end())
    return nullptr;

  return registeredFunction->getValue().fun;
}

KGEN::LIT::FnOp MojoLibraryAnalysis::getRegisteredLITMojoShapeFunctionFor(
    StringRef name) const {
  return getRegisteredLITMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGShapeFunctionLabel);
}

KGEN::LIT::FnOp MojoLibraryAnalysis::getRegisteredLITMojoUpdateViewFunctionFor(
    StringRef name) const {
  return getRegisteredLITMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGUpdateViewFunctionLabel);
}

KGEN::LIT::FnOp MojoLibraryAnalysis::getRegisteredLITMojoExecuteFunctionFor(
    StringRef name) const {
  return getRegisteredLITMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel);
}

KGEN::LIT::FnOp MojoLibraryAnalysis::getRegisteredLITMojoElementwiseFunctionFor(
    StringRef name) const {
  return getRegisteredLITMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGElementwiseFunctionLabel);
}

std::optional<StringRef> MojoLibraryAnalysis::getFullyQualifiedOpaqueTypeName(
    StringRef simpleTypeName) const {
  auto it = opaqueTypeNameToFullyQualified.find(simpleTypeName);
  if (it == opaqueTypeNameToFullyQualified.end())
    return std::nullopt;
  return StringRef(it->second);
}

//===----------------------------------------------------------------------===//
// MO op to mojo kernel verification
//===----------------------------------------------------------------------===//

/// Check if the given type matches the StaticString lowered pattern:
/// `!kgen.struct<(pointer<none>, index)>`
static bool isStaticStringLoweredPattern(Type ty) {
  auto structTy = dyn_cast<KGEN::StructType>(ty);
  if (!structTy)
    return false;

  std::optional<SmallVector<Type>> elementTypes = structTy.getElementTypes();
  if (!elementTypes || elementTypes->size() != 2)
    return false;

  // First element should be pointer<none> (KGEN::NoneType, not MLIR NoneType)
  auto ptrTy = dyn_cast<KGEN::PointerType>((*elementTypes)[0]);
  if (!ptrTy || !isa<KGEN::NoneType>(ptrTy.getElementType()))
    return false;

  // Second element should be index
  return isa<IndexType>((*elementTypes)[1]);
}

/// Look through ref types and sugar to get underlying struct type if present.
/// This mirrors the getAsStructType helper in AnnotateKernels.cpp.
static KGEN::LIT::StructType getAsStructType(Type t) {
  // First try to unwrap LIT::RefType (for pass-by-reference parameters)
  if (auto asLitRef = KGEN::sugarDynCast<KGEN::LIT::RefType>(t))
    return KGEN::sugarDynCast<KGEN::LIT::StructType>(asLitRef.getElementType());
  return KGEN::sugarDynCast<KGEN::LIT::StructType>(t);
}

/// Check if the given LIT::StructType represents StaticString
/// (i.e., StringSlice[StaticConstantOrigin]) rather than a generic
/// StringSlice[O] where O is a parameter reference.
///
/// StaticString has a concrete origin value (StaticConstantOrigin), while
/// generic StringSlice[O] has a parameter reference (ParamDeclRefAttr) as its
/// origin.
static bool isLITStaticStringType(Type ty) {
  KGEN::LIT::StructType structTy = getAsStructType(ty);
  if (!structTy)
    return false;

  // Check if the symbol is std::StringSlice
  auto symbol = structTy.getSymbol();
  std::string symbolName = llvm::Twine(symbol.getRootReference().strref())
                               .concat("::")
                               .concat(symbol.getLeafReference().strref())
                               .str();
  if (symbolName != "std::StringSlice")
    return false;

  // Check if the origin parameter is a concrete value (not a parameter
  // reference)
  auto paramValues = structTy.getParamValues();
  if (paramValues.empty())
    return false;

  // The first parameter should be the origin
  TypedAttr originAttr = paramValues[0];

  // If the origin is a parameter reference (ParamDeclRefAttr), this is a
  // generic StringSlice[O], not StaticString. Return false.
  if (KGEN::sugarIsa<KGEN::ParamDeclRefAttr>(originAttr))
    return false;

  // If we get here, the origin is a concrete value (likely
  // StaticConstantOrigin). This is StaticString.
  return true;
}

static StringAttr maybeDemangleName(StringAttr name) {
  StringRef demangled = KGEN::LIT::demangleParameterName(name);
  if (demangled != name)
    return StringAttr::get(name.getContext(), demangled);
  return name;
}

static ErrorOrSuccess
validateCustomOpParametersImpl(const MojoLibraryAnalysis &libraryAnalysis,
                               MO::CustomOp customOp) {
  // Find the kernel from the library.
  auto maybeAdaptor = getMojoKernelAdaptor(libraryAnalysis, customOp);
  if (maybeAdaptor.isError())
    return maybeAdaptor.takeError();

  auto adaptor = maybeAdaptor.takeValue();
  auto mojoCode = adaptor.executeFunction->mojoCode;

  // For LIT::FnOp, parameters can be on both the parent StructDeclOp AND on the
  // function itself. For GeneratorOp, the parameters are on the generator.
  // NOTE: AnnotateKernels.cpp sets MOGG_INPUT_PARAM_TYPES with struct params
  // first, then function params.
  SmallVector<KGEN::ParamDeclAttr> inputParamsVec;
  ArrayAttr parameterTypeNames;

  // For LIT, get both the parent struct's parameters and the function's
  // parameters. Struct params come first, then function params.
  if (auto structDeclOp =
          mojoCode->template getParentOfType<KGEN::LIT::StructDeclOp>()) {
    llvm::append_range(inputParamsVec, structDeclOp.getInputParams());
  }
  llvm::append_range(inputParamsVec, mojoCode.getInputParams());
  parameterTypeNames = mojoCode->template getAttrOfType<ArrayAttr>(
      KGEN::MOGGPreElab::MOGG_INPUT_PARAM_TYPES);
  ArrayRef<KGEN::ParamDeclAttr> inputParams = inputParamsVec;

  struct MojoParameterAdaptor {
    KGEN::ParamDeclAttr attr;
    StringAttr typeName;
  };
  llvm::DenseMap<StringAttr, MojoParameterAdaptor> paramDecls;

  if (parameterTypeNames) {
    // Use llvm::zip to handle potential size mismatch gracefully - zip stops
    // at the shorter of the two ranges, matching original behavior.
    for (auto [param, value] : llvm::zip(inputParams, parameterTypeNames))
      paramDecls[maybeDemangleName(param.getName())] = {
          param, dyn_cast<StringAttr>(value)};
  } else {
    // If parameterTypeNames is not available, populate paramDecls with the
    // parameter declarations (without type names).
    for (auto param : inputParams)
      paramDecls[maybeDemangleName(param.getName())] = {param, nullptr};
  }

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
      // KGEN::StringType can be used for StaticString, StringSlice, and String.
      // Only StaticString should accept string parameters in custom ops.
      // We must verify this is explicitly StaticString (typeName ==
      // "std::StringSlice"). If typeName is not available, we preserve the
      // original behavior of accepting any StringAttr for KGEN::StringType.
      if (typeName &&
          typeName ==
              KGEN::MOGGPreElab::MOJO_INTERNAL_STATIC_STRING_TYPE_NAME) {
        // This is StaticString, accept StringAttr.
        [[maybe_unused]] auto asString = dyn_cast<StringAttr>(mlirParamAttr);
        ASSERT_STREAM(asString != nullptr,
                      "failed to convert custom op attribute "
                          << mlirParamAttr
                          << " to expected mojo kernel param type "
                          << kgenParamTy << ": expected string type");
      } else if (typeName) {
        // typeName is set but not StaticString - this is StringSlice or String.
        return Error(llvm::formatv(
            "in custom op '{}' attribute '{}' uses a string-like type that is "
            "not supported. Use type StaticString instead to define a string "
            "parameter for a custom op.",
            customOp.getSymbol(), customParamDecl.getName()));
      } else {
        // typeName is null (non-emit-mojo path fallback) - preserve original
        // behavior of accepting any StringAttr for KGEN::StringType.
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
    } else if (typeName &&
               typeName == KGEN::MOGGPreElab::MOJO_DTYPE_TYPE_NAME) {
      // In emit-mojo path, litTypeToSourceName returns "std::DType" for DType
      // parameters. Accept TypeAttr.
      auto asType = dyn_cast<TypeAttr>(mlirParamAttr);
      if (asType == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected a DType",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName &&
               typeName == KGEN::MOGGPreElab::MOJO_STATIC_STRING_TYPE_NAME) {
      // In emit-mojo path, litTypeToSourceName returns "std::StaticString" for
      // StaticString. Accept StringAttr.
      auto asString = dyn_cast<StringAttr>(mlirParamAttr);
      if (asString == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected string type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName &&
               typeName ==
                   KGEN::MOGGPreElab::MOJO_INTERNAL_STATIC_STRING_TYPE_NAME) {
      // typeName == "std::StringSlice" can be either StaticString or generic
      // StringSlice in emit-mojo path, since both produce the same typeName.
      // We need to check the actual type to distinguish them.
      // In non-emit-mojo path, we can also check isStaticStringLoweredPattern.
      bool isStaticString = isLITStaticStringType(kgenParamTy) ||
                            isStaticStringLoweredPattern(kgenParamTy);
      if (isStaticString) {
        // This is StaticString, accept StringAttr.
        auto asString = dyn_cast<StringAttr>(mlirParamAttr);
        if (asString == nullptr) {
          return Error(llvm::formatv(
              "failed to convert custom op '{}' attribute '{}' with value '{}' "
              "to the proper Mojo parameter type: expected string type",
              customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
        }
      } else {
        // This is StringSlice (not StaticString), reject string literals.
        return Error(llvm::formatv(
            "in custom op '{}' attribute '{}' uses type StringSlice. Use type "
            "StaticString instead to define a string parameter for a custom "
            "op.",
            customOp.getSymbol(), customParamDecl.getName()));
      }
    } else if (typeName &&
               (typeName ==
                    KGEN::MOGGPreElab::MOJO_INTERNAL_DPS_INT_TYPE_NAME ||
                typeName ==
                    KGEN::MOGGPreElab::MOJO_INTERNAL_DPS_UINT_TYPE_NAME)) {
      // Mojo's Int/UInt types should accept integer attributes.
      auto asInt = dyn_cast<IntegerAttr>(mlirParamAttr);
      if (asInt == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected an integer type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName &&
               typeName ==
                   KGEN::MOGGPreElab::MOJO_INTERNAL_DPS_BOOL_TYPE_NAME) {
      // Mojo's Bool type should accept boolean attributes.
      auto asBool = dyn_cast<BoolAttr>(mlirParamAttr);
      if (asBool == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected a boolean type",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName &&
               (typeName ==
                    KGEN::MOGGPreElab::MOJO_INTERNAL_STRING_SLICE_TYPE_NAME ||
                typeName == KGEN::MOGGPreElab::MOJO_STRING_TYPE_NAME)) {
      std::string userTypeName =
          typeName == KGEN::MOGGPreElab::MOJO_INTERNAL_STRING_SLICE_TYPE_NAME
              ? "StringSlice"
              : "String";
      return Error(llvm::formatv(
          "in custom op '{}' attribute '{}' uses type {}. Use type "
          "StaticString instead to define a string parameter for a custom op.",
          customOp.getSymbol(), customParamDecl.getName(), userTypeName));
    } else if (!typeName && isStaticStringLoweredPattern(kgenParamTy)) {
      // When typeName is not available (litTypeToSourceName returns UnitAttr
      // for non-struct types like StaticString), check if the type matches the
      // StaticString lowered pattern: !kgen.struct<(pointer<none>, index)>
      // Accept StringAttr, reject other attribute types.
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

ErrorOrSuccess
MojoLibraryAnalysis::validateCustomOpParameters(MO::CustomOp customOp) const {
  return validateCustomOpParametersImpl(*this, customOp);
}

static ErrorOrSuccess
verifyMOOpMatchesKernelImpl(const MojoLibraryAnalysis &libraryAnalysis,
                            Operation *op) {
  // Find the kernel from the library.
  auto maybeAdaptor = getMojoKernelAdaptor(libraryAnalysis, op);
  if (maybeAdaptor.isError())
    return maybeAdaptor.takeError();

  auto adaptor = maybeAdaptor.takeValue();
  ErrorOr<OperandMapping> errOr =
      mapMOOpToKernelImplementation(op, *adaptor.executeFunction);
  if (errOr.isError())
    return errOr.takeError();

  if (auto custom = dyn_cast<MO::CustomOp>(op))
    return libraryAnalysis.validateCustomOpParameters(custom);

  return success();
}

ErrorOrSuccess
MojoLibraryAnalysis::verifyMOOpMatchesKernel(Operation *op) const {
  return verifyMOOpMatchesKernelImpl(*this, op);
}

ErrorOr<std::shared_ptr<MojoLibraryAnalysis>>
M::MOGG::MojoLibraryAnalysis::fromPaths(MLIRContext *ctx,
                                        ArrayRef<std::string> paths) {
  auto result = std::make_shared<MOGG::MojoLibraryAnalysis>(ctx);
  if (auto status = result->init(paths))
    return status.takeError();
  return result;
}
