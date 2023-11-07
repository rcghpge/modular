//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringExtras.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "Support/Compiler/OperationUtils.h"

namespace {
#include MOGG_MOJO_BINDING_H
};

using namespace M;

namespace M::MOGG {

KGEN::GeneratorOp MojoLibraryAnalysis::getRegisteredMojoFunctionFor(
    StringRef name, llvm::StringMap<std::string> &mapping) {
  auto registeredFunction = mapping.find(name.str());
  if (registeredFunction == mapping.end())
    return nullptr;

  return cast_or_null<KGEN::GeneratorOp>(
      mlir::SymbolTable(*mojoBindings).lookup(registeredFunction->getValue()));
}

bool MojoLibraryAnalysis::kernelHasInputParameter(KGEN::GeneratorOp kernel,
                                                  StringRef parameterName) {
  return llvm::any_of(kernel.getInputParams(), [&](auto param) {
    return KGEN::LIT::demangleIfNeeded(param).getName() == parameterName;
  });
}

bool MojoLibraryAnalysis::kernelHasDecorator(KGEN::GeneratorOp kernel,
                                             StringRef decoratorName) {
  // Return true if the decorator exists on this generator.
  auto hasDecorator = [&](TypedAttr attr) {
    auto sym = dyn_cast<KGEN::SymbolConstantAttr>(attr);
    if (!sym)
      return false;

    /// FIXME: This will give a false positive when a decorator is a prefix of
    /// another one.
    return sym.getSymbol().getLeafReference().strref().startswith(
        decoratorName);
  };

  for (TypedAttr decorator : kernel.getDecorators()) {
    // The attribute could just be a symbol directly.
    if (hasDecorator(decorator))
      return true;

    // Otherwise look through any bind expressions.
    auto apply = dyn_cast<KGEN::ParamOperatorAttr>(decorator);
    if (apply && hasDecorator(apply.getOperand(0)))
      return true;
  }

  return false;
}

static ErrorOr<OwningOpRef<ModuleOp>> parseMojoPackage(MLIRContext *ctx,
                                                       StringRef path) {
  std::string errorMsg;
  auto result = mlir::openInputFile(path, &errorMsg);
  if (!result)
    return Error(errorMsg);

  Block b;
  if (failed(mlir::readBytecodeFile(
          *result, &b,
          mlir::ParserConfig(ctx,
                             /*verifyAfterParse=*/false)))) {
    return Error::getStaticString("failed to parse package");
  }
  if (!isa<KGEN::LIT::PackageOp>(b.front()))
    return Error::getStaticString("invalid package");
  auto packageOp = cast<KGEN::LIT::PackageOp>(b.front());

  std::optional<mlir::DenseResourceElementsAttr> preElaboratedModule =
      packageOp.getPreElaborationModule();
  if (!preElaboratedModule)
    return Error::getStaticString("invalid package");

  mlir::AsmResourceBlob *blob = preElaboratedModule->getRawHandle().getBlob();
  assert(blob && "invalid package");

  llvm::StringRef moduleBuffer(blob->getData().data(), blob->getData().size());

  auto kgenModule = mlir::parseSourceString<mlir::ModuleOp>(moduleBuffer, ctx);
  return kgenModule;
}

static ErrorOr<OwningOpRef<ModuleOp>> parseExtraLib(MLIRContext *ctx,
                                                    StringRef path) {
  if (path.ends_with(".mojopkg") || path.ends_with(".📦"))
    return parseMojoPackage(ctx, path);

  return mlir::parseSourceFile<mlir::ModuleOp>(path, ctx);
}

// Load a `;` delimited string list of kernels.
static ErrorOrSuccess
loadUserProvidedKernels(ModuleOp baseModule, const std::string &extraLibPaths) {
  if (extraLibPaths.empty())
    return success();

  SmallVector<StringRef> paths;
  llvm::SplitString(StringRef{extraLibPaths}, paths, ";");

  // The symbol table of the main IR file we should add user provided files to.
  mlir::SymbolTable mainSymTable{baseModule};

  // Load each of the extra user libraries and appending them to the main
  // module.
  for (StringRef path : paths) {
    auto userModuleOr = parseExtraLib(baseModule.getContext(), path);
    if (userModuleOr.isError())
      return userModuleOr.takeError();

    OwningOpRef<ModuleOp> userModule = userModuleOr.takeValue();
    if (!userModule)
      return Error("[MojoLibraryAnalysis] Could not load MOGG Mojo "
                   "auxillary binding library");
    mlir::SymbolTable userSymTable{*userModule};

    // Copy all the functions over into the new module. We will not check if
    // duplicates are identical and will assume there's no version issues.
    for (mlir::FunctionOpInterface op :
         userModule->getOps<mlir::FunctionOpInterface>()) {
      // Don't duplicate any functions.
      if (mainSymTable.lookup(op.getName()) != nullptr)
        continue;
      baseModule.getBody()->push_back(op.clone());
    }
  }

  return success();
}

// We scan the IR to collect all the candidates and then we evaluate them based
// on the metadata collected from the generator.
struct Candidate {
  Candidate(KGEN::GeneratorOp gen) : generator(gen) {}
  KGEN::GeneratorOp generator;
  int64_t priority = -1;
};

// MOGG intrinsics to their mojo implementation.
#define INTRINSIC_TO_MOJO(Intrin, MojoName)                                    \
  { Intrin, MojoName }

// Mapping of the mo op name to the mojo function which implements it.
#define MO_TO_MOJO(MOOp, MojoName)                                             \
  INTRINSIC_TO_MOJO(MOOp::getOperationName(), MojoName)

#define TYPE_TO_MOJO(Type, MojoName)                                           \
  INTRINSIC_TO_MOJO(Type::getMnemonic(), MojoName)

MojoLibraryAnalysis::MojoLibraryAnalysis(
    llvm::StringMap<std::string> &registeredMojoKernelSymbol,
    llvm::StringMap<std::string> &registeredShapeFunctionSymbol,
    OwningOpRef<mlir::ModuleOp> &mojoBindings)
    : mojoBindings(std::move(mojoBindings)),
      registeredMojoKernelSymbol(std::move(registeredMojoKernelSymbol)),
      registeredShapeFunctionSymbol(std::move(registeredShapeFunctionSymbol)) {}

ErrorOr<MojoLibraryAnalysis>
MojoLibraryAnalysis::get(MLIRContext *ctx, const std::string &extraLibPaths) {
  /// Define the mapping of MO operations and intrinsics to the mojo code.
  /// TODO(#23802): Get rid of this mapping (except for types).
  llvm::StringMap<StringLiteral> moToMojoKernels{
      MO_TO_MOJO(MO::AddOp, "$math::$math::add"),
      MO_TO_MOJO(MO::AndOp, "$math::$math::logical_and"),
      MO_TO_MOJO(MO::ArgMaxOp, "argmax"),
      MO_TO_MOJO(MO::ArgMinOp, "argmin"),
      MO_TO_MOJO(MO::AvgPoolOp, "$MOGG::avg_pool"),
      MO_TO_MOJO(MO::BottomKOp, "$MOGG::bottom_k"),
      MO_TO_MOJO(MO::BroadcastShapeOp, "$MOGG::broadcast_shape"),
      MO_TO_MOJO(MO::ConcatOp, "$MOGG::concat"),
      MO_TO_MOJO(MO::ConcatFromListOp, "$MOGG::concat_from_list"),
      MO_TO_MOJO(MO::CumSumOp, "$MOGG::cumsum"),
      MO_TO_MOJO(MO::ConvOp, "$MOGG::conv"),
      MO_TO_MOJO(MO::ConvTransposeOp, "$MOGG::conv_transpose"),
      MO_TO_MOJO(MO::CeilOp, "$math::$math::ceil"),
      MO_TO_MOJO(MO::CosOp, "$math::$math::cos"),
      MO_TO_MOJO(MO::DivOp, "$math::$math::div"),
      MO_TO_MOJO(MO::ErfOp, "$math::$math::erf"),
      MO_TO_MOJO(MO::ExpOp, "$math::$math::exp"),
      MO_TO_MOJO(MO::EqualOp, "$math::$math::equal"),
      MO_TO_MOJO(MO::FloorOp, "$math::$math::floor"),
      MO_TO_MOJO(MO::GatherOp, "$MOGG::gather"),
      MO_TO_MOJO(MO::GatherNdOp, "$MOGG::gather_nd"),
      MO_TO_MOJO(MO::GeluOp, "$Activations::gelu"),
      MO_TO_MOJO(MO::GreaterOp, "$math::$math::greater"),
      MO_TO_MOJO(MO::GreaterEqualOp, "$math::$math::greater_equal"),
      MO_TO_MOJO(MO::GeluOp, "$Activations::gelu"),
      MO_TO_MOJO(MO::IsInfOp, "$math::$limit::isinf"),
      MO_TO_MOJO(MO::IsNaNOp, "$math::$math::isnan"),
      MO_TO_MOJO(MO::LinAlg::SolveOp, "$MatrixSolve::matrix_solve"),
      MO_TO_MOJO(MO::LinAlg::BandPartOp, "$MatrixBandPart::matrix_band_part"),
      MO_TO_MOJO(MO::LogOp, "$math::$math::log"),
      MO_TO_MOJO(MO::Log1pOp, "$math::$math::log1p"),
      MO_TO_MOJO(MO::NotOp, "$math::$math::logical_not"),
      MO_TO_MOJO(MO::MeanOp, "$MOGG::mean"),
      MO_TO_MOJO(MO::MaxPoolOp, "$Pool::max_pool"),
      MO_TO_MOJO(MO::MatmulOp, "$MOGG::matmul"),
      MO_TO_MOJO(MO::BatchMatmulOp, "$MOGG::batched_matmul"),
      MO_TO_MOJO(MO::MulOp, "$math::$math::mul"),
      MO_TO_MOJO(MO::ModOp, "$math::$math::mod"),
      MO_TO_MOJO(MO::NegativeOp, "$MOGG::negative"),
      MO_TO_MOJO(MO::NonMaximumSuppressionOp, "$MOGG::non_maximum_suppression"),
      MO_TO_MOJO(MO::NotEqualOp, "$math::$math::not_equal"),
      MO_TO_MOJO(MO::PadOp, "$MOGG::pad"),
      MO_TO_MOJO(MO::ReduceAddOp, "$MOGG::reduce_add"),
      MO_TO_MOJO(MO::ReduceMaxOp, "$MOGG::reduce_max"),
      MO_TO_MOJO(MO::ReduceMinOp, "$MOGG::reduce_min"),
      MO_TO_MOJO(MO::ReduceMulOp, "$MOGG::reduce_mul"),
      MO_TO_MOJO(MO::ReluOp, "$Activations::relu"),
      MO_TO_MOJO(MO::ResizeLinearOp, "$MOGG::resize_linear"),
      MO_TO_MOJO(MO::ResizeNearestOp, "$MOGG::resize_nearest"),
      MO_TO_MOJO(MO::RoundOp, "$math::$math::round"),
      MO_TO_MOJO(MO::RoundEvenOp, "$math::$math::roundeven"),
      MO_TO_MOJO(MO::RsqrtOp, "$math::$math::rsqrt"),
      MO_TO_MOJO(MO::SelectOp, "$math::$math::select"),
      MO_TO_MOJO(MO::ScatterOp, "$MOGG::scatter"),
      MO_TO_MOJO(MO::ScatterAddOp, "$MOGG::scatter_add"),
      MO_TO_MOJO(MO::ScatterMaxOp, "$MOGG::scatter_max"),
      MO_TO_MOJO(MO::ScatterMinOp, "$MOGG::scatter_min"),
      MO_TO_MOJO(MO::ScatterMulOp, "$MOGG::scatter_mul"),
      MO_TO_MOJO(MO::ScatterNdOp, "$MOGG::scatter_nd"),
      MO_TO_MOJO(MO::ScatterNdAddOp, "$MOGG::scatter_nd_add"),
      MO_TO_MOJO(MO::ScatterNdMaxOp, "$MOGG::scatter_nd_max"),
      MO_TO_MOJO(MO::ScatterNdMinOp, "$MOGG::scatter_nd_min"),
      MO_TO_MOJO(MO::ScatterNdMulOp, "$MOGG::scatter_nd_mul"),
      MO_TO_MOJO(MO::SigmoidOp, "$Activations::sigmoid"),
      MO_TO_MOJO(MO::SinOp, "$math::$math::sin"),
      MO_TO_MOJO(MO::SoftmaxOp, "$MOGG::softmax"),
      MO_TO_MOJO(MO::SplitOp, "$MOGG::split"),
      MO_TO_MOJO(MO::SqueezeShapeOp, "$MOGG::calculate_squeeze_shape"),
      MO_TO_MOJO(MO::SubOp, "$math::$math::sub"),
      MO_TO_MOJO(MO::TanhOp, "$math::$math::tanh"),
      MO_TO_MOJO(MO::ArgNonzeroOp, "$ArgNonzero::arg_nonzero"),
      MO_TO_MOJO(MO::TruncOp, "$math::$math::trunc"),
      MO_TO_MOJO(MO::UnsqueezeShapeOp, "$MOGG::calculate_unsqueeze_shape"),
      MO_TO_MOJO(MO::XorOp, "$math::$math::logical_xor"),
      MO_TO_MOJO(MO::TopKOp, "$MOGG::top_k"),
      MO_TO_MOJO(MO::TileOp, "$Tile::tile"),

      // Shape functions.
      INTRINSIC_TO_MOJO("broadcast_to_shape", "$MOGG::broadcast_to_shape"),
      INTRINSIC_TO_MOJO("concat_shape", "$MOGG::concat_shape"),
      INTRINSIC_TO_MOJO("concat_from_list_shape", "$Concat::concat_shape"),
      INTRINSIC_TO_MOJO("conv_shape", "$Conv::conv_shape"),
      INTRINSIC_TO_MOJO("conv_transpose_shape",
                        "$ConvTranspose::conv_transpose_shape"),
      INTRINSIC_TO_MOJO("gather_shape", "$GatherScatter::gather_shape"),
      INTRINSIC_TO_MOJO("non_maximum_suppression_shape",
                        "$MOGG::non_maximum_suppression_shape_func"),
      INTRINSIC_TO_MOJO("pack_conv_filter_shape",
                        "$MOGG::pack_conv_filter_shape"),
      INTRINSIC_TO_MOJO("pack_matmul_b_shape_func",
                        "$Matmul::pack_matmul_b_shape_func"),
      INTRINSIC_TO_MOJO("pad_shape", "$Pad::pad_shape"),
      INTRINSIC_TO_MOJO("pool_shape", "$Pool::pool_shape"),
      INTRINSIC_TO_MOJO("range_shape", "$Arange::arange_shape"),
      INTRINSIC_TO_MOJO("reduce_shape", "$algorithm::$reduction::reduce_shape"),
      INTRINSIC_TO_MOJO("reshape_shape", "$MOGG::reshape_shape"),
      INTRINSIC_TO_MOJO("resize_shape", "$MOGG::resize_shape"),
      INTRINSIC_TO_MOJO("roi_align_shape", "$MOGG::roi_align_shape"),
      INTRINSIC_TO_MOJO("tile_shape", "$Tile::tile_shape"),

      INTRINSIC_TO_MOJO("scatter_shape",
                        "$GatherScatter::scatter_elements_shape"),
      INTRINSIC_TO_MOJO("slice_shape", "$Slice::slice_shape"),
      INTRINSIC_TO_MOJO("split_ith_output_shape",
                        "$MOGG::split_ith_output_shape"),
      INTRINSIC_TO_MOJO("transpose_shape", "$MOGG::transpose_shape"),
      INTRINSIC_TO_MOJO("arg_nonzero_shape", "$ArgNonzero::arg_nonzero_shape"),

      // Layout transformations
      INTRINSIC_TO_MOJO("layout_transform_KN_to_KNkni",
                        "$Matmul::pack_b_ndbuffer"),
      INTRINSIC_TO_MOJO("layout_transform_NK_to_KNkni",
                        "$Matmul::pack_transposed_b_ndbuffer"),
      INTRINSIC_TO_MOJO("layout_transform_RSCF_to_FRSCf",
                        "$MOGG::pack_conv_filter"),

      INTRINSIC_TO_MOJO("simd_target", "$MOGG::get_target_simd"),
      INTRINSIC_TO_MOJO("simd_target_to_int", "$MOGG::simd_width_to_int"),
      INTRINSIC_TO_MOJO("top_k_shape", "$TopK::top_k_shape"),
      INTRINSIC_TO_MOJO("to_buffer", "$MOGG::to_buffer"),
      INTRINSIC_TO_MOJO("to_buffer_list", "$MOGG::to_buffer_list"),
      INTRINSIC_TO_MOJO("destruct_buffer_list", "$MOGG::destruct_buffer_list"),
      INTRINSIC_TO_MOJO("to_shape", "$MOGG::to_shape"),
      INTRINSIC_TO_MOJO("buf_len", "$MOGG::bufferLength"),
      INTRINSIC_TO_MOJO("elementwise", "$MOGG::elementwise_wrapper"),
      INTRINSIC_TO_MOJO("get_int_from_shape", "$MOGG::get_int_from_shape"),
      INTRINSIC_TO_MOJO("tensor_to_shape", "$MOGG::tensor_to_shape"),

      // Add the types we will need to the registry as well. The types are
      // treated differently to the functions above. The functions above
      // _implement_ the operation they map onto. Instead these functions tell
      // us which type the mapped type matches by inspecting the argument of
      // the mojo function.
      TYPE_TO_MOJO(MOGG::IndicesType, "$MOGG::TensorIndicesTypeDef"),
      TYPE_TO_MOJO(MOGG::SimdType, "$MOGG::SimdTypeDef"),
  };

  llvm::StringMap<std::string> registeredMojoKernelSymbol;
  llvm::StringMap<std::string> registeredShapeFunctionSymbol;
  OwningOpRef<mlir::ModuleOp> mojoBindings;

  llvm::StringRef source{reinterpret_cast<const char *>(MojoMoggBindings::data),
                         MojoMoggBindings::size};
  mojoBindings = mlir::parseSourceString<mlir::ModuleOp>(source, ctx);
  assert(mojoBindings);
  assert(!loadUserProvidedKernels(*mojoBindings, extraLibPaths).isError());

  // Iterate over the avaliable generators and check if each matches a mo
  // operation.
  DenseMap<StringRef, SmallVector<Candidate>> candidates;
  for (auto generator : mojoBindings->getOps<KGEN::GeneratorOp>()) {
    Candidate candidate = {generator};

    // Check if it has a decorator.
    for (TypedAttr decorator : generator.getDecorators()) {
      // Decorators are expected to the the apply of a symbol.
      auto apply = dyn_cast<KGEN::ParamOperatorAttr>(decorator);
      if (!apply)
        continue;

      // The first operand is expected to be the symbol we are applying.
      auto sym = dyn_cast<KGEN::SymbolConstantAttr>(apply.getOperand(0));
      if (!sym)
        continue;

      StringRef decoratorName = sym.getSymbol().getLeafReference().strref();
      if (decoratorName.startswith(DECORATOR_REGISTER_OVERRIDE)) {
        auto asString = cast<StringAttr>(apply.getOperand(1)).strref();
        auto intPriority = cast<IntegerAttr>(apply.getOperand(2));
        candidate.priority = intPriority.getInt();
        candidates[asString].push_back(candidate);
      } else if (decoratorName.startswith(DECORATOR_REGISTER_SHAPE_FUNC)) {
        auto asString = cast<StringAttr>(apply.getOperand(1)).str();
        registeredShapeFunctionSymbol[asString] =
            candidate.generator.getSymName();
      } else if (decoratorName.startswith(DECORATOR_REGISTER_KERNEL)) {
        auto asString = cast<StringAttr>(apply.getOperand(1)).strref();
        candidates[asString].push_back(candidate);
      }
    }

    // The generator names are mangled, so we remove the parameter and
    // argument signature.
    StringRef baseName =
        generator.getSymName().split('(').first.split('[').first;
    for (const auto &[moName, mojoName] : moToMojoKernels) {
      if (baseName == mojoName)
        candidates[moName].push_back(candidate);
    }
  }

  // Look at all kernel implementation candidates and pick the implementation
  // with the highest priority.
  for (auto key : candidates) {
    StringRef moOpName = key.first;
    SmallVector<Candidate> &candidateFuncs = key.second;

    size_t indexOfHighestPriority = 0;
    int64_t priority = candidateFuncs[0].priority;

    DenseMap<int64_t, size_t> seenPriorites;
    seenPriorites[priority] = 0;

    // Pick the kernel with the highest priority.
    for (size_t i = 1; i < candidateFuncs.size(); ++i) {
      Candidate c = candidateFuncs[i];

      auto itr = seenPriorites.find(c.priority);
      if (itr != seenPriorites.end()) {
        auto diag = mlir::emitError(c.generator.getLoc(),
                                    "override of operation has multiple "
                                    "candidates with the same priority.");
        diag.attachNote(candidateFuncs[itr->second].generator.getLoc())
            << "override with priority " << c.priority;
        return Error("Operation overriding error");
      }

      seenPriorites[c.priority] = i;
      if (c.priority > priority) {
        indexOfHighestPriority = i;
        priority = c.priority;
      }
    }

    // The generator we should select.
    StringRef selection =
        candidateFuncs[indexOfHighestPriority].generator.getSymName();
    registeredMojoKernelSymbol[moOpName.str()] = selection.str();
  }

  return MojoLibraryAnalysis(registeredMojoKernelSymbol,
                             registeredShapeFunctionSymbol, mojoBindings);
}

} // namespace M::MOGG
