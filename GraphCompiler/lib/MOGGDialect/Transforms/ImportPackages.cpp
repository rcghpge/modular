//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITInterfaces.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/Support/MojoPrecompiledFile.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/ErrorOr.h"
#include "Support/Filesystem/Paths.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>
#include <sstream>

using namespace M;
using namespace MOGG;

namespace M::MOGG {

#define GEN_PASS_DEF_IMPORTPACKAGES
#define GEN_PASS_DEF_STRIPKERNELDECLS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Mojo-source method names of extensibility kernels live in MojoIdentifiers.h
// (`kMoggExecuteFuncName`, `kMoggShapeFuncName`, `kMoggElementwiseFuncName`,
// `kMoggBuildGraphFuncName`).

/// Return the slot names of a well-known stdlib parameterised struct whose
/// decl lives in a dependency package the fast loader skips. A non-empty list
/// must match `LIT::StructType::getParamValues()` in length, or
/// `computeValueParams` asserts.
///
/// Extend this when a `shape`/`elementwise` function returns another
/// parameterised stdlib struct (today `SIMD` and `IndexList`).
static llvm::ArrayRef<llvm::StringLiteral>
getStdlibStructParamNames(mlir::SymbolRefAttr symbol) {
  // Slot names and order must match `getParamValues()` exactly, including
  // inferred (`//`) and keyword-only (`*`) parameters hoisted onto the LIT
  // type.
  static constexpr llvm::StringLiteral kSimdParams[] = {kParameterDType,
                                                        kParameterSize};
  static constexpr llvm::StringLiteral kIndexListParams[] = {
      kParameterSize, kParameterElementType};
  static constexpr llvm::StringLiteral kTileTensorParams[] = {
      kParameterMut,           kParameterDType,   kParameterLayoutType,
      kParameterOrigin,        kParameterStorage, kParameterAddressSpace,
      kParameterLinearIdxType, "origin_mut",
  };
  auto leaf = symbol.getLeafReference().strref();
  if (leaf == kLeafSIMD)
    return kSimdParams;
  if (leaf == kLeafIndexList)
    return kIndexListParams;
  if (leaf == kLeafTileTensor)
    return kTileTensorParams;
  return {};
}

/// Build per-type {struct-slot-name -> bound-value} mappings from `types`.
/// For each parameterised StructType, produces a DictionaryAttr keyed by the
/// struct's slot names, from either:
///
///   1. its StructDeclOp, via SymbolTable lookup in `pkg`, or
///   2. the `getStdlibStructParamNames` table.
///
/// Non-struct types and parameterless structs get a UnitAttr placeholder.
///
/// `strict` controls parameterised structs that fit neither path:
///   - `true`: emit an error at `loc` and return null. For `shape` /
///     `elementwise` result types, whose mapping is always consumed.
///   - `false`: fall back to a UnitAttr placeholder. For argument types, where
///     opaque-parameter inference only reads MO::OpaqueType entries, so tensor
///     and device-context types fall through harmlessly.
static mlir::ArrayAttr computeValueParams(mlir::MLIRContext *ctx,
                                          mlir::ModuleOp pkg,
                                          mlir::Location loc,
                                          mlir::TypeRange types, bool strict) {
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
            KGEN::demangleParameterName(paramDecl.getName()), val);
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

    if (!strict) {
      perArgAttrs.push_back(builder.getUnitAttr());
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
  return root == kPackageMax || root == kPackageExtensibility;
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
         "Expected a call to StringSpan.__init__");
  auto callee =
      KGEN::sugarDynCast<KGEN::SymbolConstantAttr>(initApply.getOperand(0));
  assert(callee && callee.getParamValues().size() == 3 &&
         "Expected a symbol reference to StringSpan.__init__");
  return mlir::cast<mlir::StringAttr>(callee.getParamValues()[2]);
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

  // The decorator's definition lives in an unloaded dependency package, so
  // match by namespace: only the MAX compiler's own, which users cannot access.
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

/// If `decorator` is a MAX compiler decorator whose leaf name starts with
/// `decoratorName`, return the static-string registration argument (operand 1).
/// Returns a null StringAttr if the decorator does not match.
static mlir::StringAttr
getRegisterDecoratorStringValue(mlir::TypedAttr decorator,
                                llvm::StringRef decoratorName) {
  auto decoratorSymbolOr = getDecoratorSymbol(decorator);
  if (!decoratorSymbolOr || !isMAXCompilerSymbol(*decoratorSymbolOr))
    return {};
  if (!decoratorSymbolOr->getLeafReference().strref().starts_with(
          decoratorName))
    return {};
  auto apply = KGEN::sugarDynCast<KGEN::ParamOperatorAttr>(decorator);
  if (!apply)
    return {};
  return getStringAttrFromStaticStringDecorator(apply.getOperand(1));
}

struct ExtensibilityAPIStructInfo {
  bool isElementwiseKernel = false;
  mlir::StringAttr registrationName{};
  DeviceInfoAttr deviceInfo{};
};

static ErrorOr<bool>
isExtensibilityAPIStruct(KGEN::LIT::StructDeclOp structDeclOp,
                         ExtensibilityAPIStructInfo &info) {
  for (mlir::TypedAttr decorator : structDeclOp.getDecorators()) {
    auto operands = getDecoratorLambdaArgument(decorator, {1, 2, 3, 4, 5});
    if (!operands)
      continue;

    auto name = getStringAttrFromStaticStringDecorator(
        (*operands)[kRegisterDecoratorArgName]);
    auto type = getStringAttrFromStaticStringDecorator(
        (*operands)[kRegisterDecoratorArgType]);
    auto api = getStringAttrFromStaticStringDecorator(
        (*operands)[kRegisterDecoratorArgApi]);
    auto arch = getStringAttrFromStaticStringDecorator(
        (*operands)[kRegisterDecoratorArgArch]);
    auto model = getStringAttrFromStaticStringDecorator(
        (*operands)[kRegisterDecoratorArgModel]);

    if (info.registrationName)
      return Error("Only one op can be registered per kernel");
    info.registrationName = name;
    info.deviceInfo =
        DeviceInfoAttr::get(structDeclOp.getContext(), type.strref(),
                            api.strref(), arch.strref(), model.strref(),
                            /*tile_based_fusion=*/false);
  }

  return info.registrationName != nullptr;
}

/// One registered kernel struct found while scanning an unannotated Mojo
/// package module.
struct ScannedKernel {
  KGEN::LIT::StructDeclOp structDeclOp;
  mlir::StringAttr registration;
  mlir::StringAttr mojoName;
  DeviceInfoAttr deviceInfo;
  KGEN::LIT::FnOp executeFn;
  KGEN::LIT::FnOp elementwiseFn;       // SIMD overload, may be null
  KGEN::LIT::FnOp tileElementwiseFn;   // TileTensor overload, may be null
  KGEN::LIT::FnOp buildGraphFn;        // may be null
  MOGG::KernelReductionAttr reduction; // null unless a `ReduceOp` monoid
};

/// Returns true if `decl` conforms to `algorithm.reduce_op.ReduceOp`.
static bool conformsToReduceOp(KGEN::LIT::StructDeclOp decl) {
  return llvm::any_of(
      decl.getCanonicalTrait().getSymbols(), [](KGEN::TraitSymbolAttr trait) {
        return symbolMatchesRootAndLeaf(trait.getSymbol(), kReduceOpTraitRoot,
                                        kReduceOpTraitLeaf);
      });
}

/// Returns the name of the struct parameter `attr` refers to, or "".
static llvm::StringRef getParamRefName(mlir::Attribute attr) {
  if (!attr)
    return {};
  auto ref =
      mlir::dyn_cast<KGEN::ParamDeclRefAttr>(KGEN::SugarAttr::strip(attr));
  return ref ? KGEN::demangleParameterName(ref.getName().strref())
             : llvm::StringRef();
}

/// Returns the struct parameter the trait's `width` alias names, or "" when
/// `width` isn't a bare struct parameter.
static llvm::StringRef getReduceOpWidthParam(KGEN::LIT::StructDeclOp decl) {
  for (auto alias : decl.getFields().front().getOps<KGEN::LIT::AliasDeclOp>()) {
    if (KGEN::demangleParameterName(alias.getParamDecl().getName().strref()) !=
        kReduceOpWidthAlias)
      continue;
    llvm::StringRef name = getParamRefName(alias.getValueAttr());
    for (KGEN::ParamDeclAttr param : decl.getInputParams())
      if (!name.empty() &&
          KGEN::demangleParameterName(param.getName().strref()) == name)
        return name;
    return {};
  }
  return {};
}

/// Returns true if `size` is `widthParam`, bare or through the implicit
/// `SIMDLength` conversion that `SIMD[dtype, W]` applies to it.
static bool isReduceOpWidth(mlir::Attribute size, llvm::StringRef widthParam) {
  if (getParamRefName(size) == widthParam)
    return true;
  auto sugar = mlir::dyn_cast<KGEN::SugarAttr>(size);
  auto apply = sugar
                   ? mlir::dyn_cast<KGEN::ParamOperatorAttr>(sugar.getSugared())
                   : nullptr;
  return apply && apply.getOpcode() == KGEN::POC::Apply &&
         apply.getOperands().size() == 2 &&
         getParamRefName(apply.getOperands()[1]) == widthParam;
}

/// Describes a `ReduceOp` monoid: its signature, its width parameter, and its
/// output fields (the `SIMD` fields whose width is the width parameter).
static MOGG::KernelReductionAttr getReduction(KGEN::LIT::StructDeclOp decl) {
  mlir::MLIRContext *ctx = decl.getContext();
  llvm::StringRef widthParam = getReduceOpWidthParam(decl);
  llvm::SmallVector<mlir::Attribute> outputs;
  for (auto field :
       decl.getFields().front().getOps<KGEN::LIT::StructFieldOp>()) {
    auto simd = KGEN::sugarDynCast<KGEN::LIT::StructType>(field.getType());
    if (widthParam.empty() || !simd ||
        simd.getSymbol().getLeafReference() != kLeafSIMD ||
        simd.getParamValues().size() != 2 ||
        !isReduceOpWidth(simd.getParamValues()[1], widthParam))
      continue;
    outputs.push_back(MOGG::KernelReductionOutputAttr::get(
        ctx, mlir::StringAttr::get(ctx, field.getName()),
        KGEN::SugarAttr::strip(simd.getParamValues()[0])));
  }
  return MOGG::KernelReductionAttr::get(ctx, decl.getSignature(),
                                        mlir::StringAttr::get(ctx, widthParam),
                                        mlir::ArrayAttr::get(ctx, outputs));
}

/// One primitive (non-struct) function found while scanning.
/// Primitives are standalone functions decorated with @register_internal,
/// as opposed to kernels which are structs decorated with
/// @extensibility.register.
struct ScannedPrimitiveFunction {
  mlir::StringAttr registration;
  KGEN::LIT::FnOp func;
};

/// One standalone shape function found during scanning, decorated with
/// @extensibility.register_shape_function. `funcSymName` is the FQN of the
/// function itself, used as the sym_name of the emitted ShapeFunctionDeclOp.
struct ScannedShapeFunction {
  mlir::StringAttr registration;
  mlir::StringAttr funcSymName; // ready-to-use sym_name for ShapeFunctionDeclOp
  KGEN::LIT::FnOp func;
};

/// Walk `mod` for struct declarations decorated with @extensibility.register
/// (kernels), standalone functions decorated with @register_internal
/// (primitives), and standalone functions decorated with
/// @extensibility.register_shape_function (shapeFunctions).
static mlir::LogicalResult scanForKernelDecls(
    mlir::ModuleOp mod, llvm::SmallVectorImpl<ScannedKernel> &kernels,
    llvm::SmallVectorImpl<ScannedPrimitiveFunction> &primitives,
    llvm::SmallVectorImpl<ScannedShapeFunction> &shapeFunctions) {
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
    kernel.structDeclOp = structDeclOp;
    kernel.registration = info.registrationName;
    kernel.deviceInfo = info.deviceInfo;
    kernel.mojoName = getFullyResolvedMojoName(structDeclOp);

    for (auto &op : structDeclOp.getFields().front()) {
      auto func = mlir::dyn_cast<KGEN::LIT::FnOp>(op);
      if (!func)
        continue;
      auto srcNameOpt = func.getSourceName();
      if (!srcNameOpt.has_value())
        continue;
      llvm::StringRef srcName = *srcNameOpt;
      bool recognized = true;
      if (srcName == kMoggExecuteFuncName)
        kernel.executeFn = func;
      else if (srcName == kMoggShapeFuncName) {
        func.emitError()
            << "shape functions defined as methods of a kernel struct are no "
               "longer supported; use @extensibility.register_shape_function "
               "instead";
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      } else if (srcName == kMoggElementwiseFuncName) {
        // Route to the SIMD or TileTensor slot by return type leaf name,
        // unwrapping LIT::RefType sugar as computeValueParams does.
        bool isTile = false;
        auto checkLeaf = [&](mlir::Type ty) -> bool {
          if (auto ref = KGEN::sugarDynCast<KGEN::LIT::RefType>(ty))
            ty = ref.getElementType();
          if (auto st = KGEN::sugarDynCast<KGEN::LIT::StructType>(ty))
            return st.getSymbol().getLeafReference() == kLeafTileTensor;
          return false;
        };
        if (!func.getResultTypes().empty())
          isTile = checkLeaf(func.getResultTypes().front());
        // Fallback: check first argument type in case the result type is
        // a generator that does not resolve to a StructType directly.
        if (!isTile && !func.getArgumentTypes().empty())
          isTile = checkLeaf(func.getArgumentTypes().front());
        if (isTile)
          kernel.tileElementwiseFn = func;
        else
          kernel.elementwiseFn = func;
      } else if (srcName == kMoggBuildGraphFuncName)
        kernel.buildGraphFn = func;
      else
        recognized = false;
      if (recognized && !func.getIsStatic()) {
        func.emitError()
            << "Kernel entry point '" << srcName
            << "' must be a @staticmethod (non-static methods are not "
               "supported)";
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
    }

    if (conformsToReduceOp(structDeclOp))
      kernel.reduction = getReduction(structDeclOp);

    kernels.push_back(kernel);
    return mlir::WalkResult::advance();
  });

  if (failed(result))
    return result;

  // Collect standalone functions registered with a decorator
  // (@register_internal or @register_shape_function).
  mod.walk([&](KGEN::LIT::FnOp func) -> mlir::WalkResult {
    // Skip struct member functions; only collect standalone functions.
    if (mlir::isa<KGEN::LIT::StructDeclOp>(func->getParentOp()))
      return mlir::WalkResult::advance();
    for (mlir::TypedAttr decorator : func.getDecorators()) {
      // Match @register_internal (PrimitiveDecl).
      if (auto primitiveRegistrationName =
              getRegisterDecoratorStringValue(decorator, kFnRegisterInternal)) {
        ScannedPrimitiveFunction prim;
        prim.registration = primitiveRegistrationName;
        prim.func = func;
        primitives.push_back(prim);
        break;
      }

      // Match @register_shape_function (ShapeFunctionDecl).
      if (auto shapeFunctionRegistrationName = getRegisterDecoratorStringValue(
              decorator, KGEN::kFnRegisterShapeFunction)) {
        ScannedShapeFunction sf;
        sf.registration = shapeFunctionRegistrationName;
        sf.funcSymName = getFullyResolvedMojoName(func);
        sf.func = func;
        shapeFunctions.push_back(sf);
        break;
      }
    }
    return mlir::WalkResult::advance();
  });

  return result;
}

/// Load the post-elaboration primary body from a `.mojoc` file by fully
/// materializing the bytecode IR. No dependency packages are loaded: decorator
/// matching uses the callee namespace instead of a symbol lookup into dep
/// packages.
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
    parentModule.emitError("failed to load kernel package: " + path.str() +
                           ": " + mlirBufOrErr.getError());
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

/// Populate `mapping` with simple_name -> qualified-name entries for every
/// struct declared in `pkg`, '.'-joined as in KernelDeclAdaptor's
/// buildOpaqueSymbol (e.g. "Pkg.Pkg.Struct" for two nested lit.file_module
/// @Pkg ops). The first package wins on duplicate simple names.
static void buildOpaqueTypeMapping(mlir::ModuleOp pkg,
                                   llvm::StringMap<std::string> &mapping) {
  // Pre-order + skip: struct decls don't nest other struct decls.
  pkg->walk<mlir::WalkOrder::PreOrder>([&](KGEN::LIT::StructDeclOp structOp)
                                           -> mlir::WalkResult {
    // Skip @extensibility.register kernel structs: they are resolved by
    // registration string and never flow as !mo.opaque values, so they would
    // only risk simple-name collisions. Same predicate as scanForKernelDecls,
    // which already validated them; exclude on error too.
    ExtensibilityAPIStructInfo info;
    ErrorOr<bool> isKernel = isExtensibilityAPIStruct(structOp, info);
    if (isKernel.isError() || isKernel.takeValue())
      return mlir::WalkResult::skip();

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
      return mlir::WalkResult::skip();

    // Build "outermost.intermediate...StructName" (parentNames is reversed).
    std::string qualifiedName = parentNames.back().str();
    for (int i = (int)parentNames.size() - 2; i >= 0; --i) {
      qualifiedName += ".";
      qualifiedName += parentNames[i].str();
    }
    qualifiedName += ".";
    qualifiedName += structOp.getSymName().str();

    mapping.try_emplace(structOp.getSymName(), std::move(qualifiedName));
    return mlir::WalkResult::skip();
  });
}

/// Persist `mapping` on `module` as a `mogg.opaque_type_mapping`
/// DictionaryAttr (simple -> fully-qualified), merged with any existing one.
/// New entries win: this run is authoritative for the packages it loaded.
static void persistOpaqueTypeMapping(mlir::ModuleOp module,
                                     llvm::StringMap<std::string> &mapping) {
  mlir::MLIRContext *ctx = module.getContext();

  if (auto existing = module->getAttrOfType<mlir::DictionaryAttr>(
          kOpaqueTypeMappingAttrName)) {
    for (mlir::NamedAttribute entry : existing) {
      auto valueStr = mlir::dyn_cast<mlir::StringAttr>(entry.getValue());
      if (!valueStr)
        continue;
      mapping.try_emplace(entry.getName(), valueStr.str());
    }
  }

  if (mapping.empty())
    return;

  llvm::SmallVector<mlir::NamedAttribute> entries;
  entries.reserve(mapping.size());
  for (auto &kv : mapping) {
    entries.emplace_back(mlir::StringAttr::get(ctx, kv.getKey()),
                         mlir::StringAttr::get(ctx, kv.getValue()));
  }
  module->setAttr(kOpaqueTypeMappingAttrName,
                  mlir::DictionaryAttr::get(ctx, entries));
}

struct ImportPackagesPass
    : public impl::ImportPackagesBase<ImportPackagesPass> {
  using Base::Base;

  // Emit mogg.kernel.decl ops for @extensibility.register kernels and
  // mogg.primitive.decl ops for @register_internal primitives found in `pkg`.
  // Also populates `opaqueTypeMapping` with simple → fully-qualified struct
  // name entries discovered in `pkg`. `isUserDefined` sets the is_user_defined
  // attribute on emitted KernelDeclOps; MojoRegistry::getResolvedKernel uses
  // this flag to apply override semantics.
  using RegistrationKey = std::pair<llvm::StringRef, DeviceInfoAttr>;
  void emitDecls(OpBuilder &builder, SymbolTable &symbolTable, ModuleOp pkg,
                 llvm::DenseSet<RegistrationKey> &seenRegistrations,
                 llvm::StringSet<> &seenPrimitives,
                 llvm::StringSet<> &seenShapeFunctions,
                 llvm::StringMap<std::string> &opaqueTypeMapping,
                 bool isUserDefined) {
    Operation *module = symbolTable.getOp();
    MLIRContext *ctx = module->getContext();

    SmallVector<ScannedKernel> kernels;
    SmallVector<ScannedPrimitiveFunction> primitives;
    SmallVector<ScannedShapeFunction> shapeFunctions;
    if (failed(scanForKernelDecls(pkg, kernels, primitives, shapeFunctions))) {
      module->emitError("failed to scan kernel package");
      signalPassFailure();
      return;
    }

    llvm::sort(kernels, [](const ScannedKernel &lhs, const ScannedKernel &rhs) {
      return StringRef(lhs.mojoName) < StringRef(rhs.mojoName);
    });

    llvm::sort(primitives, [](const ScannedPrimitiveFunction &lhs,
                              const ScannedPrimitiveFunction &rhs) {
      auto lhsName = getFullyResolvedMojoName(lhs.func);
      auto rhsName = getFullyResolvedMojoName(rhs.func);
      return StringRef(lhsName) < StringRef(rhsName);
    });

    buildOpaqueTypeMapping(pkg, opaqueTypeMapping);

    auto emptyArray = mlir::ArrayAttr::get(ctx, {});

    // Build a MOGG::KernelFunctionAttr for `fn`:
    // - `arg_params` is always soft: unreachable parameterised structs
    //   (tensors, device contexts) become UnitAttr placeholders.
    // - `result_params` is computed only with `computeResultParams`;
    //   `strictResultParams` errors on an unresolvable parameterised result
    //   instead of falling back to UnitAttr.
    //
    // Fails only on a strict result-mapping failure (error already emitted).
    // A null `fn` yields a null attr.
    auto buildFunctionAttr =
        [&](KGEN::LIT::FnOp fn, bool computeResultParams,
            bool strictResultParams, bool distillTraitArgs,
            bool isShapeFunc =
                false) -> mlir::FailureOr<MOGG::KernelFunctionAttr> {
      if (!fn)
        return MOGG::KernelFunctionAttr{};
      llvm::SmallVector<mlir::Attribute> paramAttrs;
      for (auto p : fn.getInputParams())
        paramAttrs.push_back(p);
      auto argParams =
          computeValueParams(ctx, pkg, fn.getLoc(), fn.getArgumentTypes(),
                             /*strict=*/false);
      mlir::ArrayAttr resultParams = emptyArray;
      if (computeResultParams) {
        resultParams =
            computeValueParams(ctx, pkg, fn.getLoc(), fn.getResultTypes(),
                               /*strict=*/strictResultParams);
        if (!resultParams)
          return mlir::failure();
      }

      // Distill a contract per trait-bound tensor argument. Other args get a
      // UnitAttr placeholder; a function with no trait args keeps an empty
      // array. Skipped for primitives, which accept any arguments.
      llvm::SmallVector<mlir::Attribute> contractAttrs;
      bool anyTraitArg = false;
      bool contractFailed = false;
      if (distillTraitArgs) {
        for (mlir::Type argType : fn.getArgumentTypes()) {
          if (getTensorArgPackTraitType(argType)) {
            mlir::emitError(fn.getLoc())
                << "variadic packs of tensor-trait arguments are not "
                   "supported; take a `TensorTuple` instead";
            contractFailed = true;
            contractAttrs.push_back(mlir::UnitAttr::get(ctx));
            continue;
          }
          llvm::SmallVector<KGEN::TraitSymbolAttr> traitSymbols =
              getTensorArgTraitSymbols(argType);
          if (traitSymbols.empty()) {
            contractAttrs.push_back(mlir::UnitAttr::get(ctx));
            continue;
          }
          anyTraitArg = true;
          mlir::FailureOr<TensorArgContract> contract =
              distillTensorArgContract(
                  traitSymbols,
                  [&](const llvm::Twine &msg) {
                    mlir::emitError(fn.getLoc()) << msg;
                  },
                  isShapeFunc);
          if (mlir::failed(contract)) {
            // Diagnostic already emitted; keep scanning so every invalid
            // argument is reported, then fail the function below.
            contractFailed = true;
            contractAttrs.push_back(mlir::UnitAttr::get(ctx));
            continue;
          }
          contractAttrs.push_back(tensorArgContractToDict(ctx, *contract));
        }
      }
      if (contractFailed)
        return mlir::failure();
      mlir::ArrayAttr tensorArgContracts =
          anyTraitArg ? mlir::ArrayAttr::get(ctx, contractAttrs) : emptyArray;

      return MOGG::KernelFunctionAttr::get(
          ctx, fn.getFullSignature(), mlir::ArrayAttr::get(ctx, paramAttrs),
          argParams, resultParams, tensorArgContracts);
    };

    bool sawFailure = false;

    for (auto kernel : kernels) {
      auto [_, inserted] = seenRegistrations.insert(
          {kernel.registration.strref(), kernel.deviceInfo});
      if (!inserted) {
        kernel.structDeclOp.emitError()
            << "kernel '" << kernel.registration.strref()
            << "' is registered multiple times with the same device infos";
        signalPassFailure();
        return;
      }

      // Always attach the struct decl loc; verification of missing entry
      // points (no execute / elementwise) blames the struct itself.
      llvm::SmallVector<mlir::Location> fnLocs;
      fnLocs.push_back(kernel.structDeclOp.getLoc());
      if (kernel.executeFn)
        fnLocs.push_back(kernel.executeFn.getLoc());
      if (kernel.elementwiseFn)
        fnLocs.push_back(kernel.elementwiseFn.getLoc());
      if (kernel.tileElementwiseFn)
        fnLocs.push_back(kernel.tileElementwiseFn.getLoc());
      if (kernel.buildGraphFn)
        fnLocs.push_back(kernel.buildGraphFn.getLoc());

      // execute / build_graph may return user-defined parameterised opaque
      // structs (e.g. `SIMDPair[P0, P1]`) that `inferOpaqueParameters` needs,
      // so compute soft. elementwise returns stdlib `SIMD`; strict catches
      // misuse.
      auto executeAttr = buildFunctionAttr(
          kernel.executeFn, /*computeResultParams=*/true,
          /*strictResultParams=*/false, /*distillTraitArgs=*/true);
      auto elementwiseAttr = buildFunctionAttr(
          kernel.elementwiseFn, /*computeResultParams=*/true,
          /*strictResultParams=*/true, /*distillTraitArgs=*/true);
      auto tileElementwiseAttr = buildFunctionAttr(
          kernel.tileElementwiseFn, /*computeResultParams=*/true,
          /*strictResultParams=*/true, /*distillTraitArgs=*/true);
      auto buildGraphAttr = buildFunctionAttr(
          kernel.buildGraphFn, /*computeResultParams=*/true,
          /*strictResultParams=*/false, /*distillTraitArgs=*/true);
      if (failed(executeAttr) || failed(elementwiseAttr) ||
          failed(tileElementwiseAttr) || failed(buildGraphAttr)) {
        // Diagnostics already emitted; keep going so every kernel is reported
        // rather than aborting on the first invalid one.
        sawFailure = true;
        continue;
      }

      // Collect struct-level ParamDeclAttrs (shared across all functions).
      llvm::SmallVector<mlir::Attribute> paramAttrs(
          kernel.structDeclOp.getInputParams());
      mlir::ArrayAttr structParams = mlir::ArrayAttr::get(ctx, paramAttrs);

      auto declOp = KernelDeclOp::create(
          builder, mlir::FusedLoc::get(ctx, fnLocs), kernel.mojoName,
          kernel.registration, kernel.deviceInfo, *executeAttr,
          *elementwiseAttr, *tileElementwiseAttr, *buildGraphAttr,
          kernel.reduction, structParams,
          mlir::BoolAttr::get(ctx, isUserDefined),
          /*sym_visibility=*/nullptr);
      symbolTable.insert(declOp);
    }

    // Emit mogg.shape_func.decl ops. At most one per registration is allowed
    // across all packages.
    for (auto &sf : shapeFunctions) {
      if (!seenShapeFunctions.insert(sf.registration.strref()).second) {
        sf.func.emitError() << "op '" << sf.registration.strref()
                            << "' has multiple shape functions registered; "
                               "only one is allowed per registration";
        sawFailure = true;
        continue;
      }
      // Trait-bound shape-function tensor arguments distill with
      // `isShapeFunc=true`: no io-role, bound by `Tensor`, `DenseTensor`, or
      // `TileTensorable`.
      auto sfFunctionAttr = buildFunctionAttr(
          sf.func, /*computeResultParams=*/true,
          /*strictResultParams=*/true, /*distillTraitArgs=*/true,
          /*isShapeFunc=*/true);
      if (failed(sfFunctionAttr)) {
        sawFailure = true;
        continue;
      }
      symbolTable.insert(ShapeFunctionDeclOp::create(
          builder, sf.func.getLoc(), sf.funcSymName,
          /*sym_visibility=*/nullptr, sf.registration, *sfFunctionAttr));
    }

    for (auto &prim : primitives) {
      llvm::StringRef registration = prim.registration.strref();
      if (!seenPrimitives.insert(registration).second)
        continue;

      // sym_name is the fully-qualified Mojo source name, as for KernelDeclOp,
      // so each primitive's Mojo function needs a unique one.
      mlir::StringAttr symName = getFullyResolvedMojoName(prim.func);
      // Primitives are identified by registration string, so skip the
      // slot-name mapping (it fails for wrapper tensor structs like
      // ManagedTensorSlice). They accept any arguments, so skip the trait
      // contract too.
      auto primFunctionAttr = buildFunctionAttr(
          prim.func, /*computeResultParams=*/false,
          /*strictResultParams=*/false, /*distillTraitArgs=*/false);
      if (failed(primFunctionAttr)) {
        sawFailure = true;
        continue;
      }
      symbolTable.insert(PrimitiveDeclOp::create(
          builder, prim.func.getLoc(), symName, /*sym_visibility=*/nullptr,
          mlir::StringAttr::get(ctx, registration), *primFunctionAttr));
    }

    if (sawFailure)
      signalPassFailure();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());

    // The cached SymbolTableAnalysis is never invalidated, so register every
    // emitted decl in it.
    SymbolTable &symbolTable =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

    // Separate deduplication sets let user packages override built-ins;
    // MojoRegistry::getResolvedKernel applies override semantics at lookup
    // time.
    llvm::DenseSet<RegistrationKey> seenBuiltinRegistrations;
    llvm::DenseSet<RegistrationKey> seenUserRegistrations;
    llvm::StringSet<> seenPrimitives;
    llvm::StringSet<> seenShapeFunctions;
    // Mapping from simple struct names to fully-qualified Mojo names, built
    // across all packages and applied to !mo.opaque types in the module.
    llvm::StringMap<std::string> opaqueTypeMapping;

    // Append mode: seed the dedup sets from existing decls so re-loading them
    // is a no-op. All count as `user` here; `is_user_defined` keeps the real
    // origin for `MojoRegistry::getResolvedKernel`.
    if (append) {
      for (auto decl : module.getOps<KernelDeclOp>()) {
        auto &set = decl.getIsUserDefined() ? seenUserRegistrations
                                            : seenBuiltinRegistrations;
        set.insert({decl.getRegistration(), decl.getRegistrationDeviceInfo()});
      }
      for (auto decl : module.getOps<PrimitiveDeclOp>())
        seenPrimitives.insert(decl.getRegistration());
      for (auto decl : module.getOps<ShapeFunctionDeclOp>())
        seenShapeFunctions.insert(decl.getRegistration());
    }

    // Load built-in packages unless we're in append mode (already in the
    // module).
    bool loadBuiltIns = !append;
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
        emitDecls(builder, symbolTable, *pkg, seenBuiltinRegistrations,
                  seenPrimitives, seenShapeFunctions, opaqueTypeMapping,
                  /*isUserDefined=*/false);
      }
    }

    // Load user-supplied packages (mojo-pkg-paths). In append mode this is
    // the delta the caller wants to add on top of what's already in the
    // module.
    for (const auto &path : mojoPkgPaths) {
      OwningOpRef<ModuleOp> pkg = loadPackageModule(ctx, path, module);
      if (!pkg) {
        signalPassFailure();
        return;
      }
      emitDecls(builder, symbolTable, *pkg, seenUserRegistrations,
                seenPrimitives, seenShapeFunctions, opaqueTypeMapping,
                /*isUserDefined=*/true);
    }

    // lit-mlir-paths mode: parse pre-compiled MLIR files (kgen-translate
    // -import-mojo output) directly, without going through .mojoc format.
    for (const auto &path : litMlirPaths) {
      OwningOpRef<ModuleOp> litModule =
          mlir::parseSourceFile<ModuleOp>(path, ctx);
      if (!litModule) {
        module.emitError("failed to load LIT MLIR package: " + path);
        signalPassFailure();
        return;
      }
      emitDecls(builder, symbolTable, *litModule, seenUserRegistrations,
                seenPrimitives, seenShapeFunctions, opaqueTypeMapping,
                /*isUserDefined=*/true);
    }

    persistOpaqueTypeMapping(module, opaqueTypeMapping);
  }
};

struct StripKernelDeclsPass
    : public impl::StripKernelDeclsBase<StripKernelDeclsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Erase through the cached SymbolTableAnalysis table (never invalidated
    // by the pass manager) so removed decls don't linger as dangling entries.
    SymbolTable &symbolTable =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

    // Collect every kernel sym_name still referenced in the module.
    llvm::DenseSet<mlir::StringAttr> usedSymNames;
    module.walk([&](KernelCallInterface op) {
      usedSymNames.insert(op.getKernelSymNameAttr());
    });

    for (auto decl : llvm::make_early_inc_range(module.getOps<KernelDeclOp>()))
      if (!usedSymNames.contains(decl.getSymNameAttr()))
        symbolTable.erase(decl);
    for (auto decl :
         llvm::make_early_inc_range(module.getOps<PrimitiveDeclOp>()))
      if (!usedSymNames.contains(decl.getSymNameAttr()))
        symbolTable.erase(decl);
    for (auto decl :
         llvm::make_early_inc_range(module.getOps<ShapeFunctionDeclOp>()))
      if (!usedSymNames.contains(decl.getSymNameAttr()))
        symbolTable.erase(decl);
  }
};

} // namespace

} // namespace M::MOGG
