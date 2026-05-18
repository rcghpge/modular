//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITInterfaces.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/Support/MojoPrecompiledFile.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/ErrorOr.h"
#include "Support/Filesystem/Paths.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>
#include <sstream>

using namespace M;
using namespace MOGG;

namespace M::MOGG {

#define GEN_PASS_DEF_IMPORTPACKAGES
#define GEN_PASS_DEF_STRIPKERNELDECLS
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

static constexpr llvm::StringLiteral kMOGGKernelAPIModuleName = "MOGGKernelAPI";
static constexpr llvm::StringLiteral kMOGGPrimitivesModuleName =
    "MOGGPrimitives";

static ErrorOr<std::string> resolveLITModulePath(mlir::MLIRContext *ctx,
                                                 llvm::SourceMgr &sourceMgr,
                                                 llvm::StringRef moduleName) {
  KGEN::CompilationOptions options;
  KGEN::LIT::ParserConfig config(ctx, options);
  KGEN::LIT::SharedState sharedState(sourceMgr, config);
  std::optional<std::string> modulePath =
      sharedState.resolveModulePath(moduleName, llvm::SMLoc());
  if (!modulePath) {
    return Error(
        llvm::formatv("Failed to resolve module path for `{}`", moduleName)
            .str());
  }
  if (!M::Filesystem::isMojoBinaryPackagePath(*modulePath)) {
    return Error(
        llvm::formatv("Resolved module path for `{}` to source directory `{}` "
                      "but a .mojoc precompiled file was expected instead",
                      moduleName, *modulePath)
            .str());
  }
  return *modulePath;
}

static ErrorOr<llvm::SmallVector<std::string>>
getBuiltInKernelPackagePaths(mlir::MLIRContext *ctx) {
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  llvm::SmallVector<std::string> paths;
  auto kernelAPIPathOr =
      resolveLITModulePath(ctx, *sourceMgr, kMOGGKernelAPIModuleName);
  if (kernelAPIPathOr.isError())
    return kernelAPIPathOr.takeError();
  paths.push_back(*kernelAPIPathOr);
  auto primitivesPathOr =
      resolveLITModulePath(ctx, *sourceMgr, kMOGGPrimitivesModuleName);
  if (primitivesPathOr.isError())
    return primitivesPathOr.takeError();
  paths.push_back(*primitivesPathOr);
  return paths;
}

static constexpr llvm::StringLiteral kExecuteFuncName = "execute";
static constexpr llvm::StringLiteral kShapeFuncName = "shape";
static constexpr llvm::StringLiteral kElementwiseFuncName = "elementwise";
static constexpr llvm::StringLiteral kUpdateViewFuncName = "update_input_view";

/// Return the slot names of a well-known stdlib parameterised struct whose
/// declaration is not present in the loaded kernel package (its decl lives
/// in a dependency package the fast loader deliberately skips). The returned
/// list, when non-empty, must have the same length as the corresponding
/// `LIT::StructType::getParamValues()`; otherwise `computeValueParams` will
/// assert.
///
/// Add a new entry here when a kernel `shape`/`elementwise` function starts
/// using a parameterised stdlib struct as a *result* type that isn't already
/// covered. Today this is `SIMD` (returned from elementwise functions) and
/// `IndexList` (returned from shape functions). Result types of `execute`,
/// `update_view`, and primitives are passed through opaquely and never need
/// a slot-name lookup, so the slot mapping isn't computed for them.
static llvm::ArrayRef<llvm::StringLiteral>
getStdlibStructParamNames(mlir::SymbolRefAttr symbol) {
  static constexpr llvm::StringLiteral kSimdParams[] = {"dtype", "size"};
  static constexpr llvm::StringLiteral kIndexListParams[] = {"size",
                                                             "element_type"};
  auto leaf = symbol.getLeafReference().strref();
  if (leaf == "SIMD")
    return kSimdParams;
  if (leaf == "IndexList")
    return kIndexListParams;
  return {};
}

/// Build per-result {struct-slot-name -> bound-value} mappings from `types`.
/// For each parameterised StructType, produces a DictionaryAttr keyed by the
/// struct's slot names. Two paths are tried in order:
///
///   1. StructDeclOp reachable via SymbolTable lookup in `pkg` (the
///      high-value path for structs whose decls are in the loaded module).
///   2. The symbol is in `getStdlibStructParamNames`'s table (a curated list
///      of stdlib structs whose layouts we recognise without loading their
///      decls).
///
/// Non-struct types and parameterless structs get a UnitAttr placeholder.
/// A parameterised struct that fits neither path is an error: we cannot
/// emit a slot-name mapping without knowing the struct's parameter names.
/// Returns null on failure (after emitting an error at `loc`).
static mlir::ArrayAttr computeValueParams(mlir::MLIRContext *ctx,
                                          mlir::ModuleOp pkg,
                                          mlir::Location loc,
                                          mlir::TypeRange types) {
  mlir::Builder builder(ctx);
  llvm::SmallVector<mlir::Attribute> perArgAttrs;
  for (mlir::Type ty : types) {
    if (auto ref = KGEN::sugarDynCast<KGEN::LIT::RefType>(ty))
      ty = ref.getElementType();
    auto structType = KGEN::sugarDynCast<KGEN::LIT::StructType>(ty);
    if (!structType || structType.getParamValues().empty()) {
      perArgAttrs.push_back(builder.getUnitAttr());
      continue;
    }

    // Path 1: struct decl reachable in the loaded module.
    auto *declOp =
        mlir::SymbolTable::lookupSymbolIn(pkg, structType.getSymbol());
    auto structDecl = mlir::dyn_cast_or_null<KGEN::LIT::StructDeclOp>(declOp);
    if (structDecl && structDecl.getInputParams().size() ==
                          structType.getParamValues().size()) {
      llvm::SmallVector<mlir::NamedAttribute> namedParams;
      for (auto [paramDecl, value] : llvm::zip(structDecl.getInputParams(),
                                               structType.getParamValues())) {
        mlir::Attribute val = value;
        if (!val)
          val = builder.getUnitAttr();
        namedParams.emplace_back(
            KGEN::LIT::demangleParameterName(paramDecl.getName()), val);
      }
      perArgAttrs.push_back(builder.getDictionaryAttr(namedParams));
      continue;
    }

    // Path 2: hardcoded known-stdlib table.
    auto stdlibNames = getStdlibStructParamNames(structType.getSymbol());
    auto paramValues = structType.getParamValues();
    if (!stdlibNames.empty() && stdlibNames.size() == paramValues.size()) {
      llvm::SmallVector<mlir::NamedAttribute> namedParams;
      for (auto [name, value] : llvm::zip(stdlibNames, paramValues)) {
        mlir::Attribute val = value;
        if (!val)
          val = builder.getUnitAttr();
        namedParams.emplace_back(name, val);
      }
      perArgAttrs.push_back(builder.getDictionaryAttr(namedParams));
      continue;
    }

    mlir::emitError(loc)
        << "unable to compute parameter-name mapping for parameterised "
           "struct '"
        << structType.getSymbol()
        << "': struct decl not present in the loaded package and not in "
           "the stdlib slot-name table";
    return {};
  }
  return builder.getArrayAttr(perArgAttrs);
}

static std::optional<mlir::SymbolRefAttr>
getDecoratorSymbol(mlir::TypedAttr attr) {
  if (auto sym = KGEN::sugarDynCast<KGEN::SymbolConstantAttr>(attr))
    return sym.getSymbol();
  if (auto apply = KGEN::sugarDynCast<KGEN::ParamOperatorAttr>(attr)) {
    if (auto sym =
            KGEN::sugarDynCast<KGEN::SymbolConstantAttr>(apply.getOperand(0)))
      return sym.getSymbol();
  }
  return std::nullopt;
}

static bool isMAXCompilerSymbol(mlir::SymbolRefAttr symbol) {
  auto root = symbol.getRootReference().strref();
  return root == "max" || root == "register" || root == "compiler" ||
         root == "compiler_internal";
}

static mlir::StringAttr
getFullyResolvedMojoName(KGEN::LIT::ASTDeclInterface op) {
  mlir::MLIRContext *ctx = op->getContext();
  llvm::SmallVector<std::string> names;
  do {
    names.push_back(op.getDeclName().str());
  } while (
      (op = mlir::dyn_cast<KGEN::LIT::ASTDeclInterface>(op->getParentOp())));

  std::stringstream ss;
  ss << names.back();
  names.pop_back();
  for (auto name : llvm::reverse(names))
    ss << "." << name;

  return mlir::StringAttr::get(ctx, ss.str());
}

static mlir::StringAttr
getStringAttrFromStaticStringDecorator(mlir::TypedAttr operand) {
  operand = KGEN::ParamOperatorAttr::stripRebind(operand);
  auto initApply = KGEN::sugarDynCast<KGEN::ParamOperatorAttr>(operand);
  assert(initApply && initApply.getOpcode() == KGEN::POC::Apply &&
         "Expected a call to StringSlice.__init__");
  auto callee =
      KGEN::sugarDynCast<KGEN::SymbolConstantAttr>(initApply.getOperand(0));
  assert(callee && callee.getParamValues().size() == 4 &&
         "Expected a symbol reference to StringSlice.__init__");
  return mlir::cast<mlir::StringAttr>(callee.getParamValues()[3]);
}

static std::optional<llvm::SmallVector<mlir::TypedAttr>>
getDecoratorLambdaArgument(mlir::TypedAttr decorator,
                           llvm::ArrayRef<size_t> indicesToFetch) {
  auto apply = KGEN::sugarDynCast<KGEN::ParamOperatorAttr>(decorator);
  if (!apply)
    return std::nullopt;

  auto sym = KGEN::sugarDynCast<KGEN::SymbolConstantAttr>(apply.getOperand(0));
  if (!sym)
    return std::nullopt;

  // The decorator function definition lives in a dependency package and is not
  // present in the primary body. Match by namespace instead: only accept calls
  // to functions in the MAX compiler's own namespaces, which users cannot
  // access.
  if (!isMAXCompilerSymbol(sym.getSymbol()))
    return std::nullopt;

  llvm::SmallVector<mlir::TypedAttr> answer;
  for (size_t i : indicesToFetch) {
    if (i >= apply.getNumOperands())
      return std::nullopt;
    answer.push_back(apply.getOperand(i));
  }
  return answer;
}

struct ExtensibilityAPIStructInfo {
  bool isElementwiseKernel = false;
  mlir::StringAttr registrationName{};
};

static ErrorOr<bool>
isExtensibilityAPIStruct(KGEN::LIT::StructDeclOp structDeclOp,
                         ExtensibilityAPIStructInfo &info) {
  for (mlir::TypedAttr decorator : structDeclOp.getDecorators()) {
    auto registerOperand =
        getDecoratorLambdaArgument(decorator, llvm::SmallVector<size_t>{1});
    if (!registerOperand)
      continue;

    auto name =
        getStringAttrFromStaticStringDecorator(registerOperand->front());
    if (info.registrationName)
      return Error("Only one op can be registered per kernel");
    info.registrationName = name;
  }

  return info.registrationName != nullptr;
}

/// One registered kernel struct found while scanning an unannotated Mojo
/// package module.
struct ScannedKernel {
  mlir::StringAttr registration;
  mlir::StringAttr mojoName;
  KGEN::LIT::FnOp executeFn;
  KGEN::LIT::FnOp shapeFn;       // may be null
  KGEN::LIT::FnOp elementwiseFn; // may be null
  KGEN::LIT::FnOp updateViewFn;  // may be null; only present on view kernels
};

/// One primitive (non-struct) function found while scanning.
/// Primitives are standalone functions decorated with @register_internal,
/// as opposed to kernels which are structs decorated with @compiler.register.
struct ScannedPrimitiveFunction {
  mlir::StringAttr registration;
  KGEN::LIT::FnOp func;
};

/// Walk `mod` for struct declarations decorated with @compiler.register
/// (kernels) and standalone functions decorated with @register_internal
/// (primitives), collecting their signatures without annotating the module.
static mlir::LogicalResult scanForKernelDecls(
    mlir::ModuleOp mod, llvm::SmallVectorImpl<ScannedKernel> &kernels,
    llvm::SmallVectorImpl<ScannedPrimitiveFunction> &primitives) {
  // Collect struct-based kernels.
  mlir::LogicalResult result = mlir::success();
  mod.walk([&](KGEN::LIT::StructDeclOp structDeclOp) -> mlir::WalkResult {
    if (structDeclOp.getDecorators().empty())
      return mlir::WalkResult::advance();

    ExtensibilityAPIStructInfo info;
    ErrorOr<bool> isExtensibility =
        isExtensibilityAPIStruct(structDeclOp, info);
    if (isExtensibility.isError()) {
      structDeclOp.emitError(isExtensibility.getError());
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    }
    if (!isExtensibility.takeValue())
      return mlir::WalkResult::advance();

    ScannedKernel kernel;
    kernel.registration = info.registrationName;
    kernel.mojoName = getFullyResolvedMojoName(structDeclOp);

    for (auto &op : structDeclOp.getFields().front()) {
      auto func = mlir::dyn_cast<KGEN::LIT::FnOp>(op);
      if (!func)
        continue;
      if (func.getSourceName() == kExecuteFuncName)
        kernel.executeFn = func;
      else if (func.getSourceName() == kShapeFuncName)
        kernel.shapeFn = func;
      else if (func.getSourceName() == kElementwiseFuncName)
        kernel.elementwiseFn = func;
      else if (func.getSourceName() == kUpdateViewFuncName)
        kernel.updateViewFn = func;
    }

    kernels.push_back(kernel);
    return mlir::WalkResult::advance();
  });

  if (failed(result))
    return result;

  // Collect every standalone (non-struct-member) function decorated with
  // @register_internal, regardless of the registration string. The decorator
  // identity is established by the four checks below (max/compiler namespace,
  // leaf == "register_internal", and a ParamOperatorAttr application carrying
  // a static-string registration argument); any function passing those checks
  // is a legitimate primitive, so we do not need to whitelist registration
  // prefixes.
  mod.walk([&](KGEN::LIT::FnOp func) -> mlir::WalkResult {
    using namespace KGEN::MOGGPreElab::Decorators;
    // Skip struct member functions; only collect standalone functions.
    if (mlir::isa<KGEN::LIT::StructDeclOp>(func->getParentOp()))
      return mlir::WalkResult::advance();
    for (mlir::TypedAttr decorator : func.getDecorators()) {
      auto decoratorSymbolOr = getDecoratorSymbol(decorator);
      if (!decoratorSymbolOr)
        continue;
      if (!isMAXCompilerSymbol(*decoratorSymbolOr))
        continue;
      if (!decoratorSymbolOr->getLeafReference().strref().starts_with(
              REGISTER_INTERNAL_FUNCTION))
        continue;
      auto apply = KGEN::sugarDynCast<KGEN::ParamOperatorAttr>(decorator);
      if (!apply)
        continue;
      auto nameAttr =
          getStringAttrFromStaticStringDecorator(apply.getOperand(1));
      if (!nameAttr)
        continue;
      ScannedPrimitiveFunction prim;
      prim.registration = nameAttr;
      prim.func = func;
      primitives.push_back(prim);
      break;
    }
    return mlir::WalkResult::advance();
  });

  return result;
}

/// Load the post-elaboration primary body from a `.mojopkg` file by fully
/// materializing the bytecode IR. No dependency packages are loaded and the
/// postParseModule blob is not used: decorator matching uses the callee
/// namespace instead of a symbol lookup into dep packages.
static OwningOpRef<ModuleOp> loadPackageModule(MLIRContext *ctx,
                                               llvm::StringRef path,
                                               ModuleOp parentModule) {
  auto packageBuffer = llvm::MemoryBuffer::getFile(path);
  if (!packageBuffer) {
    parentModule.emitError("failed to open kernel package: " + path.str());
    return {};
  }
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  unsigned bufferId =
      sourceMgr->AddNewSourceBuffer(std::move(*packageBuffer), llvm::SMLoc());
  const llvm::MemoryBuffer *packageMemBuf =
      sourceMgr->getMemoryBuffer(bufferId);

  auto mlirBufOrErr = KGEN::getMLIRBufferFromPrecompiledFile(
      packageMemBuf->getMemBufferRef(),
      /*ignoreIncompatiblePrecompiledFiles=*/false);
  if (mlirBufOrErr.isError()) {
    parentModule.emitError("failed to load kernel package: " + path.str());
    return {};
  }
  auto mlirResult = std::move(*mlirBufOrErr);
  if (mlirResult.ownedData)
    sourceMgr->AddNewSourceBuffer(std::move(mlirResult.ownedData),
                                  llvm::SMLoc());

  // Fully materialize the primary body so scanForKernelDecls can walk
  // lit.struct.decl ops and their lit.fn members. verifyAfterParse=false
  // tolerates cross-package type references whose definitions live in
  // dependency packages not loaded here.
  mlir::ParserConfig config(ctx, /*verifyAfterParse=*/false);
  mlir::BytecodeReader bytecodeReader(mlirResult.buffer, config,
                                      /*lazyLoad=*/true, sourceMgr);
  mlir::Block block;
  if (failed(bytecodeReader.readTopLevel(&block))) {
    (void)bytecodeReader.finalize([](mlir::Operation *) { return false; });
    parentModule.emitError("failed to read primary body from package: " +
                           path.str());
    return {};
  }
  auto theModule = mlir::cast<ModuleOp>(block.front());
  if (failed(bytecodeReader.materialize(theModule)) ||
      failed(bytecodeReader.finalize([](mlir::Operation *) { return true; }))) {
    parentModule.emitError("failed to materialize primary body from package: " +
                           path.str());
    return {};
  }
  block.getOperations().remove(theModule.getOperation());
  return OwningOpRef<ModuleOp>(theModule);
}

/// Populate `mapping` with simple_name → qualified-name entries for every
/// struct declared in `pkg`. The format matches KernelDeclAdaptor's
/// buildOpaqueSymbol: every segment is joined by '.', producing e.g.
/// "Pkg.Pkg.Struct" for a struct inside two nested lit.file_module @Pkg ops.
/// The string is directly usable as a Mojo qualified name (no further
/// rewriting needed). Duplicate simple names are silently ignored (first
/// package wins).
static void buildOpaqueTypeMapping(mlir::ModuleOp pkg,
                                   llvm::StringMap<std::string> &mapping) {
  pkg->walk([&](KGEN::LIT::StructDeclOp structOp) -> mlir::WalkResult {
    // Walk all ancestor FileModuleOp / StructDeclOp ops to collect the full
    // path. parentNames is filled innermost-first and then reversed below.
    llvm::SmallVector<llvm::StringRef> parentNames;
    mlir::Operation *parentOp = structOp->getParentOp();
    while (parentOp) {
      if (auto fileMod = mlir::dyn_cast<KGEN::LIT::FileModuleOp>(parentOp)) {
        parentNames.push_back(fileMod.getSymName());
        parentOp = fileMod->getParentOp();
      } else if (auto pkg = mlir::dyn_cast<KGEN::LIT::PackageOp>(parentOp)) {
        parentNames.push_back(pkg.getSymName());
        parentOp = pkg->getParentOp();
      } else if (auto parentStruct =
                     mlir::dyn_cast<KGEN::LIT::StructDeclOp>(parentOp)) {
        parentNames.push_back(parentStruct.getSymName());
        parentOp = parentStruct->getParentOp();
      } else {
        break;
      }
    }

    if (parentNames.empty())
      return mlir::WalkResult::advance();

    // Build "outermost.intermediate...StructName" (parentNames is reversed).
    std::string qualifiedName = parentNames.back().str();
    for (int i = (int)parentNames.size() - 2; i >= 0; --i) {
      qualifiedName += ".";
      qualifiedName += parentNames[i].str();
    }
    qualifiedName += ".";
    qualifiedName += structOp.getSymName().str();

    mapping.try_emplace(structOp.getSymName(), std::move(qualifiedName));
    return mlir::WalkResult::advance();
  });
}

/// Replace every !mo.opaque<"simple_name"> type in `module` with
/// !mo.opaque<"fully.qualified.name"> using `mapping`. Types whose symbol is
/// not present in the mapping are left unchanged.
static void rewriteOpaqueTypes(mlir::ModuleOp module,
                               const llvm::StringMap<std::string> &mapping) {
  if (mapping.empty())
    return;
  mlir::MLIRContext *ctx = module.getContext();
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](MO::OpaqueType opaqueTy) -> std::optional<mlir::Type> {
        auto it = mapping.find(opaqueTy.getSymbol().getValue());
        if (it == mapping.end())
          return std::nullopt;
        return MO::OpaqueType::get(mlir::StringAttr::get(ctx, it->second),
                                   opaqueTy.getParameters());
      });
  replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

struct ImportPackagesPass
    : public impl::ImportPackagesBase<ImportPackagesPass> {
  using Base::Base;

  // Emit mogg.kernel.decl ops for @compiler.register kernels and
  // mogg.primitive.decl ops for @register_internal primitives found in `pkg`.
  // Also populates `opaqueTypeMapping` with simple → fully-qualified struct
  // name entries discovered in `pkg`. `isUserDefined` sets the is_user_defined
  // attribute on emitted KernelDeclOps; MojoRegistry::getResolvedKernel uses
  // this flag to apply override semantics.
  void emitDecls(OpBuilder &builder, ModuleOp module, ModuleOp pkg,
                 llvm::StringSet<> &seenRegistrations,
                 llvm::StringSet<> &seenPrimitives,
                 llvm::StringMap<std::string> &opaqueTypeMapping,
                 bool isUserDefined) {
    MLIRContext *ctx = module.getContext();

    SmallVector<ScannedKernel> kernels;
    SmallVector<ScannedPrimitiveFunction> primitives;
    if (failed(scanForKernelDecls(pkg, kernels, primitives))) {
      module.emitError("failed to scan kernel package");
      signalPassFailure();
      return;
    }

    buildOpaqueTypeMapping(pkg, opaqueTypeMapping);

    auto emptyArray = mlir::ArrayAttr::get(ctx, {});

    // Build a MOGG_KernelFunctionAttr (signature + fn-level params + cached
    // per-result mapping) for one function variant. Result-side slot names
    // are cached because consumers look them up by name (e.g. "dtype" for
    // SIMD-shaped result types). Argument-side has no analogous cache:
    // consumers walk `signature.getArguments()` directly via
    // `collectParameterReferences`, which doesn't need slot names.
    // Build a MOGG::KernelFunctionAttr for `fn`. When `computeResultParams`
    // is true, also build a per-result {slot-name -> bound-value} mapping;
    // this is only needed for function variants whose return type is queried
    // by slot name (shape and elementwise). For other variants (execute,
    // update_view) and for primitives, the result is passed through opaquely
    // and the slot mapping is left empty. Returns failure if a requested
    // result mapping cannot be computed (the helper has already emitted an
    // error). A null `fn` is not an error: it yields a null attr that the
    // caller stores in the corresponding optional slot.
    auto buildFunctionAttr = [&](KGEN::LIT::FnOp fn, bool computeResultParams)
        -> mlir::FailureOr<MOGG::KernelFunctionAttr> {
      if (!fn)
        return MOGG::KernelFunctionAttr{};
      llvm::SmallVector<mlir::Attribute> paramAttrs;
      for (auto p : fn.getInputParams())
        paramAttrs.push_back(p);
      mlir::ArrayAttr resultParams = emptyArray;
      if (computeResultParams) {
        resultParams =
            computeValueParams(ctx, pkg, fn.getLoc(), fn.getResultTypes());
        if (!resultParams)
          return mlir::failure();
      }
      return MOGG::KernelFunctionAttr::get(
          ctx, fn.getFullSignature(), mlir::ArrayAttr::get(ctx, paramAttrs),
          resultParams);
    };

    for (auto kernel : kernels) {
      if (!kernel.executeFn && !kernel.elementwiseFn)
        continue;
      auto [_, inserted] =
          seenRegistrations.insert(kernel.registration.strref());
      if (!inserted)
        continue;

      llvm::SmallVector<mlir::Location> fnLocs;
      if (kernel.executeFn)
        fnLocs.push_back(kernel.executeFn.getLoc());
      if (kernel.shapeFn)
        fnLocs.push_back(kernel.shapeFn.getLoc());
      if (kernel.elementwiseFn)
        fnLocs.push_back(kernel.elementwiseFn.getLoc());
      if (kernel.updateViewFn)
        fnLocs.push_back(kernel.updateViewFn.getLoc());

      auto executeAttr = buildFunctionAttr(kernel.executeFn,
                                           /*computeResultParams=*/false);
      auto shapeAttr = buildFunctionAttr(kernel.shapeFn,
                                         /*computeResultParams=*/true);
      auto elementwiseAttr = buildFunctionAttr(kernel.elementwiseFn,
                                               /*computeResultParams=*/true);
      auto updateViewAttr = buildFunctionAttr(kernel.updateViewFn,
                                              /*computeResultParams=*/false);
      if (failed(executeAttr) || failed(shapeAttr) || failed(elementwiseAttr) ||
          failed(updateViewAttr)) {
        signalPassFailure();
        return;
      }

      // Collect struct-level ParamDeclAttrs (shared across all functions).
      mlir::ArrayAttr structParams = emptyArray;
      {
        KGEN::LIT::FnOp paramSrcFn =
            kernel.executeFn ? kernel.executeFn : kernel.elementwiseFn;
        if (!paramSrcFn)
          paramSrcFn = kernel.shapeFn ? kernel.shapeFn : kernel.updateViewFn;
        if (paramSrcFn) {
          if (auto structDecl = mlir::dyn_cast<KGEN::LIT::StructDeclOp>(
                  paramSrcFn->getParentOp())) {
            llvm::SmallVector<mlir::Attribute> attrs;
            for (auto p : structDecl.getInputParams())
              attrs.push_back(p);
            structParams = mlir::ArrayAttr::get(ctx, attrs);
          }
        }
      }

      KernelDeclOp::create(builder, mlir::FusedLoc::get(ctx, fnLocs),
                           kernel.mojoName, kernel.registration, *executeAttr,
                           *shapeAttr, *elementwiseAttr, *updateViewAttr,
                           structParams,
                           mlir::BoolAttr::get(ctx, isUserDefined));
    }

    for (auto &prim : primitives) {
      llvm::StringRef registration = prim.registration.strref();
      if (!seenPrimitives.insert(registration).second)
        continue;

      // sym_name is the fully-qualified Mojo source name so that consumers can
      // emit the correct identifier without consulting a separate attribute.
      // (This mirrors KernelDeclOp's convention.) Each primitive's Mojo source
      // function must therefore have a unique fully-qualified name; the MLIR
      // symbol table would otherwise reject a duplicate sym_name.
      mlir::StringAttr symName = getFullyResolvedMojoName(prim.func);
      // Primitives are well-known to the graph compiler; consumers identify
      // them by registration string and never look up a result type's slot
      // names by name. Skip the slot-name mapping (which would fail for
      // wrapper tensor structs like ManagedTensorSlice that primitives
      // commonly return).
      auto primFunctionAttr = buildFunctionAttr(prim.func,
                                                /*computeResultParams=*/false);
      if (failed(primFunctionAttr)) {
        signalPassFailure();
        return;
      }
      PrimitiveDeclOp::create(builder, prim.func.getLoc(), symName,
                              mlir::StringAttr::get(ctx, registration),
                              *primFunctionAttr);
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());

    // Separate deduplication sets let user packages override built-ins;
    // MojoRegistry::getResolvedKernel applies override semantics at lookup
    // time.
    llvm::StringSet<> seenBuiltinRegistrations;
    llvm::StringSet<> seenUserRegistrations;
    llvm::StringSet<> seenPrimitives;
    // Mapping from simple struct names to fully-qualified Mojo names, built
    // across all packages and applied to !mo.opaque types in the module.
    llvm::StringMap<std::string> opaqueTypeMapping;

    // mojo-pkg-paths mode: load built-in packages first, then user packages.
    // Built-in packages are skipped when only lit-mlir-paths is provided.
    bool loadBuiltIns = !mojoPkgPaths.empty() || litMlirPaths.empty();
    if (loadBuiltIns) {
      auto builtInPathsOr = getBuiltInKernelPackagePaths(ctx);
      if (builtInPathsOr.isError()) {
        module.emitError("failed to resolve built-in kernel package paths");
        signalPassFailure();
        return;
      }

      for (const auto &path : builtInPathsOr.takeValue()) {
        OwningOpRef<ModuleOp> pkg = loadPackageModule(ctx, path, module);
        if (!pkg) {
          signalPassFailure();
          return;
        }
        emitDecls(builder, module, *pkg, seenBuiltinRegistrations,
                  seenPrimitives, opaqueTypeMapping, /*isUserDefined=*/false);
      }

      for (const auto &path : mojoPkgPaths) {
        OwningOpRef<ModuleOp> pkg = loadPackageModule(ctx, path, module);
        if (!pkg) {
          signalPassFailure();
          return;
        }
        emitDecls(builder, module, *pkg, seenUserRegistrations, seenPrimitives,
                  opaqueTypeMapping, /*isUserDefined=*/true);
      }
    }

    // lit-mlir-paths mode: parse pre-compiled MLIR files (kgen-translate
    // -import-mojo output) directly, without going through .mojopkg format.
    for (const auto &path : litMlirPaths) {
      OwningOpRef<ModuleOp> litModule =
          mlir::parseSourceFile<ModuleOp>(path, ctx);
      if (!litModule) {
        module.emitError("failed to load LIT MLIR package: " + path);
        signalPassFailure();
        return;
      }
      emitDecls(builder, module, *litModule, seenBuiltinRegistrations,
                seenPrimitives, opaqueTypeMapping, /*isUserDefined=*/false);
    }

    rewriteOpaqueTypes(module, opaqueTypeMapping);
  }
};

struct StripKernelDeclsPass
    : public impl::StripKernelDeclsBase<StripKernelDeclsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto decl : llvm::make_early_inc_range(module.getOps<KernelDeclOp>()))
      decl.erase();
    for (auto decl :
         llvm::make_early_inc_range(module.getOps<PrimitiveDeclOp>()))
      decl.erase();
  }
};

} // namespace

} // namespace M::MOGG
