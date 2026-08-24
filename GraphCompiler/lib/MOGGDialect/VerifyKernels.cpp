//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"

using namespace M;
using namespace KGEN;
using namespace M::MOGG;

namespace M::MOGG {

#define GEN_PASS_DEF_VERIFYKERNELS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

template <typename Func, typename... Args>
[[nodiscard]] LogicalResult setFlagIfFailed(bool &flag, Func &&func,
                                            Args &&...args) {
  LogicalResult ret = func(std::forward<Args>(args)...);
  if (ret.failed())
    flag |= true;

  return ret;
}

//===----------------------------------------------------------------------===//
// KernelDeclAdaptor-based verification (mogg.kernel.decl path)
//===----------------------------------------------------------------------===//

LogicalResult declVerifyShapeFunctionHasNoDPSOutputs(
    KernelDeclOp decl,
    const std::optional<KernelDeclFunctionAdaptor> &shapeFunction) {
  if (!shapeFunction.has_value())
    return success();
  if (!shapeFunction->outputArguments.empty()) {
    decl->emitError() << "Shape functions must not have any output tensor "
                         "arguments";
    return failure();
  }
  if (!shapeFunction->outputResult.has_value()) {
    decl->emitError() << "Shape functions must return a shape";
    return failure();
  }
  return success();
}

// Return `s` with its first ASCII letter uppercased. Used so error messages
// that start with a role label like "execute" or "build_graph" begin with a
// capital, while mid-sentence uses keep the original lowercase.
static std::string capitalize(llvm::StringRef s) {
  if (s.empty())
    return "";
  std::string out = s.str();
  out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
  return out;
}

/// Walk the input argument lists of `lhs` and `rhs` in parallel, skipping
/// launch args (DeviceContext / DeviceGraphBuilder) from either side, and
/// require pairwise compatibility. `lhsKind` / `rhsKind` are lowercase role
/// labels used in error messages (e.g. "execute" / "shape").
///
/// A pair is compatible when types match exactly, or when the lhs side has a
/// `MutableInputTensor` and the rhs side has a non-mutable tensor (shape /
/// build_graph cannot mutate their inputs, so a relaxed match is allowed).
LogicalResult verifyInputArgumentsAlign(KernelDeclOp decl,
                                        const KernelDeclFunctionAdaptor &lhs,
                                        llvm::StringRef lhsKind,
                                        const KernelDeclFunctionAdaptor &rhs,
                                        llvm::StringRef rhsKind) {
  auto skipLaunchArg =
      [](const llvm::SmallVector<KernelDeclOperandAdaptor> &inputs,
         uint64_t &idx) -> std::optional<KernelDeclOperandAdaptor> {
    while (idx < inputs.size()) {
      const auto &arg = inputs[idx];
      if (!arg.isLaunchArgument())
        return arg;
      ++idx;
    }
    return std::nullopt;
  };

  uint64_t lhsIndex = 0, rhsIndex = 0;
  while (true) {
    const auto lhsArgOpt = skipLaunchArg(lhs.inputArguments, lhsIndex);
    const auto rhsArgOpt = skipLaunchArg(rhs.inputArguments, rhsIndex);
    if (!lhsArgOpt.has_value() || !rhsArgOpt.has_value())
      break;

    auto isCompatibleArgs = [&]() {
      if (*lhsArgOpt == *rhsArgOpt)
        return true;
      return lhsArgOpt->isMutableTensorType() &&
             (rhsArgOpt->isTensorType() && !rhsArgOpt->isMutableTensorType());
    };

    if (!isCompatibleArgs()) {
      return decl->emitError()
             << capitalize(rhsKind) << " function and " << lhsKind
             << " function input argument types do not match. "
             << capitalize(lhsKind) << " function input at position "
             << lhsArgOpt->sourceDescriptor->position << " is a "
             << lhsArgOpt->underlyingType << " but corresponding " << rhsKind
             << " function argument at position " << rhsIndex << " is a "
             << rhsArgOpt->underlyingType;
    }
    ++lhsIndex;
    ++rhsIndex;
  }

  bool lhsEnd = lhsIndex == lhs.inputArguments.size();
  bool rhsEnd = rhsIndex == rhs.inputArguments.size();

  if (!lhsEnd || !rhsEnd) {
    std::stringstream errorMessage;
    errorMessage << capitalize(lhsKind) << " and " << rhsKind.str()
                 << " functions do not have the same input tensors. ";
    auto appendUnmatchedInputs =
        [&](const llvm::SmallVector<KernelDeclOperandAdaptor> &inputs,
            uint64_t idx) {
          while (skipLaunchArg(inputs, idx).has_value()) {
            errorMessage << inputs[idx].sourceDescriptor->position << ",";
            ++idx;
          }
        };
    if (lhsEnd) {
      errorMessage << capitalize(rhsKind) << " function inputs at positions (";
      appendUnmatchedInputs(rhs.inputArguments, rhsIndex);
      errorMessage << ") do not appear in the " << lhsKind.str() << " function";
    } else {
      errorMessage << capitalize(lhsKind) << " function inputs at positions (";
      appendUnmatchedInputs(lhs.inputArguments, lhsIndex);
      errorMessage << ") do not appear in the " << rhsKind.str() << " function";
    }
    return decl->emitError(errorMessage.str());
  }

  return success();
}

LogicalResult declVerifyShapeFunctionInputsMatchExecuteFunction(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor,
    const std::optional<KernelDeclFunctionAdaptor> &shapeFunction) {
  if (!shapeFunction.has_value())
    return success();

  if (adaptor.executeFunction.has_value() &&
      verifyInputArgumentsAlign(decl, *adaptor.executeFunction, "execute",
                                *shapeFunction, "shape")
          .failed())
    return failure();

  if (adaptor.buildGraphFunction.has_value() &&
      verifyInputArgumentsAlign(decl, *adaptor.buildGraphFunction,
                                "build_graph", *shapeFunction, "shape")
          .failed())
    return failure();

  return success();
}

/// build_graph must align with execute modulo their launch arguments
/// (DeviceContext vs DeviceGraphBuilder). Same tensor signature is required.
LogicalResult
declVerifyExecuteMatchesBuildGraph(KernelDeclOp decl,
                                   const KernelDeclAdaptor &adaptor) {
  if (!adaptor.executeFunction.has_value() ||
      !adaptor.buildGraphFunction.has_value())
    return success();
  return verifyInputArgumentsAlign(decl, *adaptor.executeFunction, "execute",
                                   *adaptor.buildGraphFunction, "build_graph");
}

/// A build_graph method records nodes against a DeviceGraphBuilder, so the
/// signature must explicitly accept one. There is no meaningful default for
/// this argument.
LogicalResult declVerifyBuildGraphHasBuilder(KernelDeclOp decl,
                                             const KernelDeclAdaptor &adaptor) {
  if (!adaptor.buildGraphFunction.has_value())
    return success();

  bool hasBuilder = llvm::any_of(
      adaptor.buildGraphFunction->inputArguments,
      [](const KernelDeclOperandAdaptor &arg) { return arg.isBuilderType(); });

  if (!hasBuilder) {
    decl->emitError()
        << "build_graph method must accept a DeviceGraphBuilder argument";
    return failure();
  }
  return success();
}

LogicalResult declVerifyExecuteFunctionDoesNotReturnATensor(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor) {
  auto doesNotReturnTensor =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value() || !func->outputResult.has_value())
      return success();
    if (func->outputResult->isTensorType()) {
      decl->emitError()
          << "Value semantics kernels (return a tensor) are not yet supported";
      return failure();
    }
    return success();
  };

  if (doesNotReturnTensor(adaptor.executeFunction).failed() ||
      doesNotReturnTensor(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

LogicalResult
declVerifyOutputsAppearBeforeInputs(KernelDeclOp decl,
                                    const KernelDeclAdaptor &adaptor) {
  auto outputsBeforeInputs =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value())
      return success();
    int64_t indexOfFirstInput = std::numeric_limits<int64_t>::max();
    for (const auto &arg : func->inputArguments) {
      indexOfFirstInput =
          std::min<int64_t>(indexOfFirstInput, arg.sourceDescriptor->position);
    }
    int64_t indexOfLastOutput = std::numeric_limits<int64_t>::min();
    for (const auto &arg : func->outputArguments) {
      indexOfLastOutput =
          std::max<int64_t>(indexOfLastOutput, arg.sourceDescriptor->position);
    }
    if (indexOfFirstInput < indexOfLastOutput) {
      decl->emitError()
          << "Outputs must appear before inputs in the function signature";
      return failure();
    }
    return success();
  };

  if (outputsBeforeInputs(adaptor.executeFunction).failed() ||
      outputsBeforeInputs(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

/// Each kernel function variant has at most one launch arg
/// (DeviceContext / DeviceContextArray on execute / shape, DeviceGraphBuilder
/// on build_graph) and the launch arg must come last in the signature.
LogicalResult declVerifyContextsAppearLastInSignature(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor,
    const std::optional<KernelDeclFunctionAdaptor> &shapeFunction) {
  const auto launchArgsAreLast =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func,
              llvm::StringRef argLower,
              llvm::StringRef argTitle) -> LogicalResult {
    if (!func.has_value())
      return success();
    if (llvm::count_if(func->inputArguments,
                       [](const KernelDeclOperandAdaptor &arg) {
                         return arg.isLaunchArgument();
                       }) > 1) {
      decl->emitError() << "There can be at most one " << argLower;
      return failure();
    }
    int64_t indexOfLastArgument = std::numeric_limits<int64_t>::min();
    int64_t indexOfLaunchArgument = std::numeric_limits<int64_t>::max();
    for (const auto &arg : func->inputArguments) {
      if (arg.isLaunchArgument())
        indexOfLaunchArgument = arg.sourceDescriptor->position;
      indexOfLastArgument = std::max<int64_t>(indexOfLastArgument,
                                              arg.sourceDescriptor->position);
    }
    if (indexOfLaunchArgument < indexOfLastArgument) {
      decl->emitError() << argTitle
                        << " argument must be the last argument "
                           "in the function signature";
      return failure();
    }
    return success();
  };

  if (launchArgsAreLast(adaptor.executeFunction, "context", "Context")
          .failed() ||
      launchArgsAreLast(shapeFunction, "context", "Context").failed() ||
      launchArgsAreLast(adaptor.buildGraphFunction, "graph builder",
                        "Graph builder")
          .failed())
    return failure();
  return success();
}

LogicalResult declVerifyOpaqueArgumentsArePassedByReference(
    KernelDeclOp decl, KernelDeclAdaptor &adaptor,
    std::optional<KernelDeclFunctionAdaptor> shapeFunction) {
  auto opaquesByReference =
      [&decl](std::optional<KernelDeclFunctionAdaptor> &func) -> LogicalResult {
    if (!func.has_value())
      return success();
    auto allArgs = llvm::concat<const KernelDeclOperandAdaptor>(
        func->outputArguments, func->inputArguments);
    for (const auto &arg : allArgs) {
      // Copy sig to call the non-const getArgConvention.
      auto sig = func->attr.getSignature();
      auto convention = sig.getArgConvention(arg.sourceDescriptor->position);
      if (arg.isOpaqueType() && (convention == KGEN::ArgConvention::OwnedMem ||
                                 convention == KGEN::ArgConvention::DeinitMem ||
                                 convention == KGEN::ArgConvention::OwnedReg)) {
        decl->emitError()
            << "Argument convention for argument at position "
            << arg.sourceDescriptor->position
            << " is invalid: Owned arguments not supported for custom types "
               "in kernels";
        return failure();
      }
      // Tensors must be passed by const reference (`read`/`read_mem`):
      // their mutability is encoded in the type itself via IOSpec, not in
      // the calling convention. Reject `mut`, `var`, owned, and `ref`.
      // Trait-bound tensor arguments now also classify as `isTensorType()`,
      // but their borrow rule is owned by the trait lane's
      // `declVerifyTensorArgsAreBorrowed` (which emits a trait-specific
      // diagnostic); skip them here so the legacy message stays scoped to the
      // legacy `ManagedTensorSlice`/`IOSpec` struct lane.
      if (arg.isTensorType() &&
          !getTensorArgTraitType(
              sig.getArgument(arg.sourceDescriptor->position)) &&
          convention != KGEN::ArgConvention::ImmMem &&
          convention != KGEN::ArgConvention::ImmReg) {
        decl->emitError()
            << "Only the borrowed argument (read) convention is supported "
               "for tensor arguments (argument at position "
            << arg.sourceDescriptor->position << " uses '"
            << KGEN::stringifyArgConvention(convention) << "')";
        return failure();
      }
    }
    return success();
  };

  if (opaquesByReference(adaptor.executeFunction).failed() ||
      opaquesByReference(shapeFunction).failed() ||
      opaquesByReference(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

// Trait-bound tensor arguments (the trait lane) carry their role and
// mutability in the distilled contract, not in the calling convention, so they
// must use the borrowed (`read`/`read_mem`) convention -- mirroring the legacy
// tensor rule in `declVerifyOpaqueArgumentsArePassedByReference`, which gates
// on the legacy `isTensorType()` predicate that trait args do not satisfy.
LogicalResult declVerifyTensorArgsAreBorrowed(
    KernelDeclOp decl, KernelDeclAdaptor &adaptor,
    std::optional<KernelDeclFunctionAdaptor> shapeFunction) {
  auto borrowed =
      [&decl](std::optional<KernelDeclFunctionAdaptor> &func) -> LogicalResult {
    if (!func.has_value())
      return success();
    // Copy sig to call the non-const getArgConvention.
    auto sig = func->attr.getSignature();
    for (size_t i = 0, n = sig.getNumArguments(); i < n; ++i) {
      if (!getTensorArgTraitType(sig.getArgument(i)))
        continue;
      auto convention = sig.getArgConvention(i);
      if (convention != KGEN::ArgConvention::ImmMem &&
          convention != KGEN::ArgConvention::ImmReg) {
        decl->emitError()
            << "Only the borrowed argument (read) convention is supported for "
               "trait-bound tensor arguments (argument at position "
            << i << " uses '" << KGEN::stringifyArgConvention(convention)
            << "')";
        return failure();
      }
    }
    return success();
  };

  if (borrowed(adaptor.executeFunction).failed() ||
      borrowed(shapeFunction).failed() ||
      borrowed(adaptor.updateViewFunction).failed() ||
      borrowed(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

// Per-function-kind policy for trait-bound tensor arguments on the
// KernelDeclOp-hosted entry points. Today only `execute` accepts them; the
// others reject them. As each kind grows its own contract this is where the
// rules live -- e.g. build_graph will forbid `TileTensorable`, and
// update_input_view will require a `ViewUpdatable`. (Shape functions and
// primitives are separate decl ops and are checked at import, as is universal
// well-formedness. This is the kind-specific layer for execute /
// update_input_view / build_graph / elementwise.)
LogicalResult declVerifyTraitArgFunctionKinds(KernelDeclOp decl,
                                              KernelDeclAdaptor &adaptor) {
  auto rejectTraitArgs = [&decl](std::optional<KernelDeclFunctionAdaptor> &func,
                                 StringRef kind) -> LogicalResult {
    if (!func.has_value())
      return success();
    auto sig = func->attr.getSignature();
    bool ok = true;
    for (size_t i = 0, n = sig.getNumArguments(); i < n; ++i) {
      if (getTensorArgTraitType(sig.getArgument(i))) {
        decl->emitError()
            << "trait-bound tensor arguments are not supported on " << kind
            << " functions yet (only `execute`)";
        ok = false;
      }
    }
    return ok ? success() : failure();
  };

  bool ok = true;
  if (rejectTraitArgs(adaptor.updateViewFunction, "update_input_view").failed())
    ok = false;
  if (rejectTraitArgs(adaptor.buildGraphFunction, "build_graph").failed())
    ok = false;
  if (rejectTraitArgs(adaptor.elementwiseFunction, "elementwise").failed())
    ok = false;
  if (rejectTraitArgs(adaptor.tileElementwiseFunction, "tile_elementwise")
          .failed())
    ok = false;
  return ok ? success() : failure();
}

LogicalResult declVerifyAtMostOneVariadicInputAndOutput(
    KernelDeclOp decl, const KernelDeclAdaptor &adaptor,
    const std::optional<KernelDeclFunctionAdaptor> &shapeFunction) {
  // TODO: (GEX-1994) Remove this.
  if (adaptor.registeredOpName.strref().starts_with("mo."))
    return success();

  const auto atMostOneVariadic =
      [&decl](const std::optional<KernelDeclFunctionAdaptor> &func)
      -> LogicalResult {
    if (!func.has_value())
      return success();
    if (llvm::count_if(func->inputArguments,
                       [](const KernelDeclOperandAdaptor &arg) {
                         return arg.isVariadicTensorType();
                       }) > 1) {
      decl->emitError()
          << "Kernel function can have at most one variadic input tensors";
      return failure();
    }
    if (llvm::count_if(func->outputArguments,
                       [](const KernelDeclOperandAdaptor &arg) {
                         return arg.isVariadicTensorType();
                       }) > 1) {
      decl->emitError()
          << "Kernel function can have at most one variadic output tensors";
      return failure();
    }
    return success();
  };

  if (atMostOneVariadic(adaptor.executeFunction).failed() ||
      atMostOneVariadic(shapeFunction).failed() ||
      atMostOneVariadic(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

LogicalResult
declVerifyExecuteNotReturningOpaqueAndThrows(KernelDeclOp decl,
                                             KernelDeclAdaptor &adaptor) {
  auto notOpaqueAndThrows =
      [&decl](std::optional<KernelDeclFunctionAdaptor> &func) -> LogicalResult {
    if (!func.has_value() || !func->outputResult.has_value())
      return success();
    auto sig = func->attr.getSignature();
    if (func->outputResult->isOpaqueType() && sig.isThrows()) {
      decl->emitError() << "Functions that raise and return a custom type are "
                           "not yet supported";
      return failure();
    }
    return success();
  };

  if (notOpaqueAndThrows(adaptor.executeFunction).failed() ||
      notOpaqueAndThrows(adaptor.buildGraphFunction).failed())
    return failure();
  return success();
}

LogicalResult declVerifyHasEntryPoint(KernelDeclOp decl,
                                      const KernelDeclAdaptor &adaptor) {
  if (!adaptor.executeFunction.has_value() &&
      !adaptor.elementwiseFunction.has_value()) {
    decl->emitError() << "The kernel must have an entry point named execute";
    return failure();
  }
  return success();
}

LogicalResult declVerifyElementwiseFunction(KernelDeclOp decl,
                                            const KernelDeclAdaptor &adaptor) {
  if (!adaptor.elementwiseFunction.has_value())
    return success();

  const KernelDeclFunctionAdaptor &elemwiseFn = *adaptor.elementwiseFunction;

  // Copy sig to call non-const isThrows.
  auto sig = elemwiseFn.attr.getSignature();
  if (sig.isThrows())
    return decl->emitError() << "Elementwise kernels can't throw";

  if (!elemwiseFn.outputArguments.empty())
    return decl->emitError() << "Elementwise kernels must return an SIMD value";
  if (!elemwiseFn.outputResult.has_value() ||
      !elemwiseFn.outputResult->isScalarType())
    return decl->emitError() << "Elementwise kernels must return an SIMD value";

  size_t operandsCount = 0;
  for (size_t i = 0, inputsCount = elemwiseFn.inputArguments.size();
       i < inputsCount; ++i) {
    // The last argument might be an IndexList. Skip it — it is an
    // elementwise-only concept.
    if (i + 1 == inputsCount &&
        elemwiseFn.inputArguments[i].isOpaqueWithTypeName(
            MOGG::kMojoInternalDpsIndexListTypeQualifiedName)) {
      continue;
    }
    ++operandsCount;
    if (!elemwiseFn.inputArguments[i].isScalarType())
      return decl->emitError() << "Elementwise kernels must have SIMD operands";
  }

  if (!operandsCount)
    return decl->emitError()
           << "Elementwise kernels must have at least one operand";

  return success();
}

LogicalResult
declVerifyTileElementwiseFunction(KernelDeclOp decl,
                                  const KernelDeclAdaptor &adaptor) {
  if (!adaptor.tileElementwiseFunction.has_value())
    return success();

  const KernelDeclFunctionAdaptor &fn = *adaptor.tileElementwiseFunction;

  auto sig = fn.attr.getSignature();
  if (sig.isThrows())
    return decl->emitError() << "Tile elementwise kernels can't throw";

  if (!fn.outputArguments.empty())
    return decl->emitError()
           << "Tile elementwise kernels must return a TileTensor value";
  if (!fn.outputResult.has_value() || !fn.outputResult->isTileType())
    return decl->emitError()
           << "Tile elementwise kernels must return a TileTensor value";

  size_t operandsCount = 0;
  for (const auto &arg : fn.inputArguments) {
    ++operandsCount;
    if (!arg.isTileType())
      return decl->emitError()
             << "Tile elementwise kernels must have TileTensor operands";
  }

  if (!operandsCount)
    return decl->emitError()
           << "Tile elementwise kernels must have at least one operand";

  return success();
}

//===----------------------------------------------------------------------===//
// Cross-kernel verifiers (consistency across multiple registrations of the
// same op name)
//===----------------------------------------------------------------------===//

enum class KernelTypeTag { Execute, Elementwise, View };

static KernelTypeTag getKernelType(const KernelDeclAdaptor &adaptor) {
  if (adaptor.isViewKernel())
    return KernelTypeTag::View;
  if (adaptor.isElementwise() || adaptor.isTileElementwise())
    return KernelTypeTag::Elementwise;
  if (adaptor.executeFunction.has_value())
    return KernelTypeTag::Execute;
  llvm_unreachable("kernel has no recognised entry point; "
                   "declVerifyHasEntryPoint should have caught this");
}

static llvm::StringRef kernelTypeName(KernelTypeTag t) {
  switch (t) {
  case KernelTypeTag::Execute:
    return "execute";
  case KernelTypeTag::Elementwise:
    return "elementwise";
  case KernelTypeTag::View:
    return "view";
  }
  llvm_unreachable("unknown KernelTypeTag");
}

static llvm::SmallVector<KernelDeclOperandAdaptor>
getNonLaunchInputs(const KernelDeclFunctionAdaptor &fn) {
  llvm::SmallVector<KernelDeclOperandAdaptor> result;
  for (const KernelDeclOperandAdaptor &arg : fn.inputArguments)
    if (!arg.isLaunchArgument())
      result.push_back(arg);
  return result;
}

// Return the semantic result of a kernel function, or nullopt if the function
// uses `raises` (in which case outputResult is a raises ABI artifact, not a
// real return value).
static std::optional<KernelDeclOperandAdaptor>
getSemanticResult(const KernelDeclFunctionAdaptor &fn) {
  if (fn.attr.getSignature().isThrows())
    return std::nullopt;
  return fn.outputResult;
}

/// Verify cross-kernel consistency for every op name with multiple
/// registrations. For each group, decls[0] is the reference (ImportPackages
/// inserts built-ins first, so it is the built-in when one exists).
///
/// Per registration, checks in order:
///   1. Kernel type must match (execute / elementwise / view).
///   2. If execute: output count, non-launch input count, and per-operand
///      types must all match (fusion variants are treated as equivalent).
LogicalResult crossVerifyKernels(ModuleOp moduleOp) {
  llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<KernelDeclOp>> byReg;
  for (KernelDeclOp decl : moduleOp.getOps<KernelDeclOp>())
    byReg[decl.getRegistrationAttr()].push_back(decl);

  bool failed = false;
  for (auto &[regName, decls] : byReg) {
    if (decls.size() < 2)
      continue;

    KernelDeclAdaptor refAdaptor(decls[0]);
    KernelTypeTag refTag = getKernelType(refAdaptor);
    bool refIsBuiltin = !decls[0].getIsUserDefined();

    // Pre-compute reference signature (used when refTag == Execute or
    // Elementwise).
    const llvm::SmallVector<KernelDeclOperandAdaptor> *refOutputs = nullptr;
    llvm::SmallVector<KernelDeclOperandAdaptor> refInputs;
    std::optional<KernelDeclOperandAdaptor> refOutputResult;
    if (refTag == KernelTypeTag::Execute) {
      refOutputs = &refAdaptor.executeFunction->outputArguments;
      refInputs = getNonLaunchInputs(*refAdaptor.executeFunction);
      refOutputResult = getSemanticResult(*refAdaptor.executeFunction);
    } else if (refTag == KernelTypeTag::Elementwise) {
      refOutputs = &refAdaptor.elementwiseFunction->outputArguments;
      refInputs = getNonLaunchInputs(*refAdaptor.elementwiseFunction);
      refOutputResult = getSemanticResult(*refAdaptor.elementwiseFunction);
    }

    for (size_t i = 1; i < decls.size(); ++i) {
      KernelDeclAdaptor adaptor(decls[i]);

      // 1. Kernel type check.
      KernelTypeTag tag = getKernelType(adaptor);
      if (tag != refTag) {
        if (refIsBuiltin) {
          decls[i]->emitError()
              << "Kernel \"" << regName.getValue()
              << "\" type mismatch: built-in uses " << kernelTypeName(refTag)
              << ", but this registration uses " << kernelTypeName(tag) << "; "
              << kernelTypeName(refTag) << " method expected";
        } else {
          decls[i]->emitError()
              << "Kernel \"" << regName.getValue()
              << "\" type mismatch: this registration uses "
              << kernelTypeName(tag) << " but a previous registration uses "
              << kernelTypeName(refTag) << "; " << kernelTypeName(refTag)
              << " method expected";
        }
        failed = true;
        continue;
      }

      // 2. Signature check (execute and elementwise; skip view / build-graph).
      if (refTag != KernelTypeTag::Execute &&
          refTag != KernelTypeTag::Elementwise)
        continue;

      llvm::StringRef sigKind;
      const llvm::SmallVector<KernelDeclOperandAdaptor> *outputs = nullptr;
      llvm::SmallVector<KernelDeclOperandAdaptor> inputs;
      std::optional<KernelDeclOperandAdaptor> outputResult;
      if (refTag == KernelTypeTag::Execute) {
        sigKind = "Execute";
        outputs = &adaptor.executeFunction->outputArguments;
        inputs = getNonLaunchInputs(*adaptor.executeFunction);
        outputResult = getSemanticResult(*adaptor.executeFunction);
      } else {
        sigKind = "Elementwise";
        outputs = &adaptor.elementwiseFunction->outputArguments;
        inputs = getNonLaunchInputs(*adaptor.elementwiseFunction);
        outputResult = getSemanticResult(*adaptor.elementwiseFunction);
      }

      // Count mismatch.
      if (outputs->size() != refOutputs->size() ||
          inputs.size() != refInputs.size() ||
          outputResult.has_value() != refOutputResult.has_value()) {
        if (refIsBuiltin) {
          decls[i]->emitError()
              << sigKind << " signature mismatch for \"" << regName.getValue()
              << "\": " << regName.getValue() << " expects "
              << refOutputs->size() << " output(s), " << refInputs.size()
              << " input(s), and " << (refOutputResult.has_value() ? 1 : 0)
              << " result(s), but this registration has " << outputs->size()
              << " output(s), " << inputs.size() << " input(s), and "
              << (outputResult.has_value() ? 1 : 0) << " result(s)";
        } else {
          decls[i]->emitError()
              << sigKind << " signature mismatch for \"" << regName.getValue()
              << "\": previous registration has " << refOutputs->size()
              << " output(s), " << refInputs.size() << " input(s), and "
              << (refOutputResult.has_value() ? 1 : 0)
              << " result(s), but this registration has " << outputs->size()
              << " output(s), " << inputs.size() << " input(s), and "
              << (outputResult.has_value() ? 1 : 0) << " result(s)";
        }
        failed = true;
        continue;
      }

      // Per-operand type check.
      auto emitOperandError = [&](size_t pos, llvm::StringRef kind,
                                  const KernelDeclOperandAdaptor &refOp,
                                  const KernelDeclOperandAdaptor &op) {
        std::string refStr, str;
        llvm::raw_string_ostream(refStr) << refOp;
        llvm::raw_string_ostream(str) << op;
        if (refIsBuiltin) {
          decls[i]->emitError()
              << sigKind << " signature mismatch for \"" << regName.getValue()
              << "\": " << kind << " " << pos
              << " type differs: " << regName.getValue() << " expects "
              << refStr << ", but this registration has " << str;
        } else {
          decls[i]->emitError()
              << sigKind << " signature mismatch for \"" << regName.getValue()
              << "\": " << kind << " " << pos
              << " type differs: previous registration has " << refStr
              << ", but this registration has " << str;
        }
        failed = true;
      };

      for (size_t j = 0; j < refOutputs->size(); ++j)
        if ((*refOutputs)[j] != (*outputs)[j])
          emitOperandError(j, "output", (*refOutputs)[j], (*outputs)[j]);
      for (size_t j = 0; j < refInputs.size(); ++j)
        if (refInputs[j] != inputs[j])
          emitOperandError(j, "input", refInputs[j], inputs[j]);
      if (refOutputResult.has_value() && outputResult.has_value() &&
          *refOutputResult != *outputResult)
        emitOperandError(0, "result", *refOutputResult, *outputResult);
    }
  }
  return failed ? failure() : success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct VerifyKernelsPass : public impl::VerifyKernelsBase<VerifyKernelsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MojoRegistry registry(moduleOp);

    bool failed = false;
    for (KernelDeclOp decl : moduleOp.getOps<KernelDeclOp>()) {
      std::optional<KernelDeclFunctionAdaptor> shapeFnAdaptor;
      if (registry.containsShapeFunction(decl.getRegistration()))
        shapeFnAdaptor = KernelDeclFunctionAdaptor(
            registry.lookupShapeFunction(decl.getRegistration()));
      KernelDeclAdaptor adaptor(decl);

      if (setFlagIfFailed(failed, declVerifyHasEntryPoint, decl, adaptor)
              .failed())
        continue;

      // Elementwise-only kernels: verify elementwise function(s).
      if (!adaptor.executeFunction.has_value()) {
        if (setFlagIfFailed(failed, declVerifyElementwiseFunction, decl,
                            adaptor)
                .failed())
          continue;
        if (setFlagIfFailed(failed, declVerifyTileElementwiseFunction, decl,
                            adaptor)
                .failed())
          continue;
        continue;
      }

      if (setFlagIfFailed(failed, declVerifyShapeFunctionHasNoDPSOutputs, decl,
                          shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed,
                          declVerifyShapeFunctionInputsMatchExecuteFunction,
                          decl, adaptor, shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyBuildGraphHasBuilder, decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyExecuteMatchesBuildGraph, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyExecuteFunctionDoesNotReturnATensor,
                          decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyOutputsAppearBeforeInputs, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyContextsAppearLastInSignature, decl,
                          adaptor, shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyOpaqueArgumentsArePassedByReference,
                          decl, adaptor, shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyTensorArgsAreBorrowed, decl,
                          adaptor, shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyTraitArgFunctionKinds, decl,
                          adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyAtMostOneVariadicInputAndOutput,
                          decl, adaptor, shapeFnAdaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyExecuteNotReturningOpaqueAndThrows,
                          decl, adaptor)
              .failed())
        continue;

      if (setFlagIfFailed(failed, declVerifyElementwiseFunction, decl, adaptor)
              .failed())
        continue;
    }

    // Cross-kernel consistency checks.
    (void)setFlagIfFailed(failed, crossVerifyKernels, moduleOp);

    if (failed)
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
