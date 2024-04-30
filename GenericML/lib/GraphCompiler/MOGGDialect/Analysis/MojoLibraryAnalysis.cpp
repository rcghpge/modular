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
#include "llvm/Support/SourceMgr.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/Support/Assert.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Context.h"
#include "Support/Filesystem/Paths.h"
#include "Support/MDialect/MDialect.h"

namespace {
#include MOGG_MOJO_BINDING_H
#include MOGG_PRIMITIVES_MOJO_BINDING_H
}; // namespace

using namespace M;

namespace M::MOGG {

namespace {

std::string getMOGGUniqueSymbolName(const std::string &baseName,
                                    llvm::StringMap<KGEN::GeneratorOp> &rawMap,
                                    unsigned &counter) {
  std::string uniqueName = baseName;
  while (rawMap.contains(uniqueName))
    uniqueName = (baseName + "_" + Twine(counter++)).str();
  return uniqueName;
}

std::string
getMOGGUniqueSymbolName(const std::string &baseName,
                        llvm::StringMap<KGEN::GeneratorOp> &rawMap) {
  unsigned counter = 0;
  return getMOGGUniqueSymbolName(baseName, rawMap, counter);
}

ErrorOr<OwningOpRef<ModuleOp>> parseMojoPackage(MLIRContext *ctx,
                                                StringRef path) {
  LLCL::Runtime &runtime = *loadContext(ctx)->get<LLCL::Runtime>();
  OwningOpRef<ModuleOp> ownedModule =
      KGEN::LIT::importStandaloneMojoBinaryPackage(
          runtime, std::make_shared<llvm::SourceMgr>(), ctx, path);
  if (!ownedModule)
    return Error("failed to load Mojo package: " + path);

  // Generate a library from the package, ensuring that dependencies are
  // materialized to make it self contained.
  if (auto err = KGEN::runLibraryGenerationPipeline(
          *ownedModule, KGEN::CompilationOptions(),
          /*materializeDependencies=*/true))
    return err.takeError();
  return ownedModule;
}

mlir::LogicalResult materialize(Operation *target,
                                mlir::BytecodeReader *reader) {
  auto walk = target->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (reader->isMaterializable(op))
      if (failed(reader->materialize(op)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (walk.wasInterrupted())
    return failure();

  return success();
}

void populateMetadata(Operation *module, bool eager,
                      llvm::SmallPtrSetImpl<Operation *> &finalizedMojoKernel,
                      llvm::StringMap<KGEN::GeneratorOp> &rawMojoKernel) {
  for (KGEN::GeneratorOp op :
       cast<mlir::ModuleOp>(module).getOps<KGEN::GeneratorOp>())
    rawMojoKernel[op.getSymName()] = op;

  if (eager)
    return;

  for (KGEN::GeneratorOp op :
       cast<mlir::ModuleOp>(module).getOps<KGEN::GeneratorOp>())
    finalizedMojoKernel.insert(op);
}

void populateRawMetadata(
    Operation *module, mlir::BytecodeReader *reader,
    llvm::DenseMap<Operation *, mlir::BytecodeReader *> &dedicatedReader,
    llvm::StringMap<KGEN::GeneratorOp> &rawMojoKernel) {

  for (auto gen : cast<mlir::ModuleOp>(module).getOps<KGEN::GeneratorOp>()) {
    rawMojoKernel[gen.getSymName()] = gen;
    dedicatedReader[gen] = reader;
  }
}

} // namespace

KGEN::GeneratorOp MojoLibraryAnalysis::getRegisteredMojoFunctionFor(
    StringRef name, llvm::StringMap<std::string> &mapping) {
  auto registeredFunction = mapping.find(name.str());
  if (registeredFunction == mapping.end())
    return nullptr;

  auto symbolName = registeredFunction->getValue();
  return getMojoFunctionNamed(symbolName);
}

KGEN::GeneratorOp MojoLibraryAnalysis::getMojoFunctionNamed(StringRef name) {
  if (rawMojoKernel.contains(name)) {
    KGEN::GeneratorOp gen = rawMojoKernel[name];
    if (failed(ready(gen)))
      return nullptr;
    return gen;
  }
  return nullptr;
}

bool MojoLibraryAnalysis::kernelHasInputParameter(KGEN::GeneratorOp kernel,
                                                  StringRef parameterName) {
  return llvm::any_of(kernel.getInputParams(), [&](auto param) {
    return KGEN::LIT::demangleIfNeeded(param).getName() == parameterName;
  });
}

// Load a list of kernels from the provided user libraries.
ErrorOrSuccess MojoLibraryAnalysis::loadUserProvidedKernels(
    ArrayRef<std::string> kernelLibraryPaths) {

  // Load each of the extra user libraries and update the book-keepings.
  for (StringRef path : kernelLibraryPaths) {
    llvm::StringMap<KGEN::GeneratorOp> userRawMojoKernel;
    Operation *userModule;

    if (Filesystem::isMojoBinaryPackagePath(path.str())) {
      // Load user library that is a mojo package file.
      auto userModuleOr = parseMojoPackage(context, path);
      if (userModuleOr.isError())
        return userModuleOr.takeError();

      userModule = userModuleOr.takeValue().release().getOperation();

      if (!userModule)
        return Error("[MojoLibraryAnalysis] Could not load MOGG Mojo "
                     "auxiliary binding library");

      populateMetadata(userModule, eager, finalizedMojoKernel,
                       userRawMojoKernel);
      // We read the library from `mojopkg` which means we don't need a lazy
      // reader for that. We insert this module op into the block because the
      // block is our centralized container for all module ops (some modules are
      // fully loaded from `mojopkg` files and some are partially loaded from
      // `mlirbc` files).
      block.push_back(userModule);
    } else {
      // Load user library that is a mojo mlibrc file.
      if (!eager) {
        // Lazy load of user library.
        auto bufOr = toModularErrorOr(llvm::MemoryBuffer::getFile(path));
        if (bufOr.isError())
          return bufOr.takeError();

        StringRef source = buffers.emplace_back(bufOr.takeValue())->getBuffer();
        llvm::MemoryBufferRef buffer(source, path);
        mlir::BytecodeReader *reader =
            readers.emplace_back(new mlir::BytecodeReader(buffer, config, true))
                .get();

        if (failed(reader->readTopLevel(&block, [&](Operation *op) {
              if (isa<mlir::ModuleOp>(op)) {
                return true;
              }
              return false;
            })))
          return Error(
              Twine("[MojoLibraryAnalysis] Could not load the mojo kernel "
                    "operators in: ") +
              path);

        if (!isa<mlir::ModuleOp>(block.back()))
          return Error("[MojoLibraryAnalysis] Top level op is not ModuleOp.");

        userModule = &block.back();
        populateRawMetadata(userModule, reader, dedicatedReader,
                            userRawMojoKernel);
      } else {
        // Eager load of user library.
        userModule =
            mlir::parseSourceFile<mlir::ModuleOp>(path, context).release();

        block.push_back(userModule);

        populateMetadata(userModule, eager, finalizedMojoKernel,
                         userRawMojoKernel);
      }
    }
    updatedLoadedKernelsForCollisions(userModule, userRawMojoKernel);
  }

  return success();
}

// Update the state in MJA for collisions etc.
void MojoLibraryAnalysis::updatedLoadedKernelsForCollisions(
    Operation *userModule,
    llvm::StringMap<KGEN::GeneratorOp> &userRawMojoKernel) {

  // Remap only exposed kernel functions if their name collides with an
  // existing function.
  symbolNameRemapping.insert({userModule, {}});
  for (KGEN::GeneratorOp op : toRawMojoKernelRange(userRawMojoKernel)) {
    // We will insert a slightly mangled version of the current name if it
    // already exists in the mlir file we intend to compile.
    std::string newSymName = op.getSymName().str();

    // If there's no collision we can copy directly obviously.
    if (!rawMojoKernel.contains(newSymName))
      continue;

    // We ignore duplicated functions which are not kernels or shape functions.
    if (!(KGEN::MOGGPreElab::isKernel(op) ||
          KGEN::MOGGPreElab::isV1ShapeFunc(op)))
      continue;

    // Create a new symbol which is unique in both files.
    while (rawMojoKernel.contains(newSymName) ||
           userRawMojoKernel.contains(newSymName)) {
      newSymName = getMOGGUniqueSymbolName(newSymName, rawMojoKernel);
      newSymName = getMOGGUniqueSymbolName(newSymName, userRawMojoKernel);
    }

    auto oldAttr = FlatSymbolRefAttr::get(context, op.getSymNameAttr());

    symbolNameRemapping[userModule].push_back({newSymName, oldAttr});

    // Update any references to the old symbol with the new symbol.
    mlir::AttrTypeReplacer symbolUpdater;
    symbolUpdater.addReplacement(
        [&](FlatSymbolRefAttr attr) -> std::optional<FlatSymbolRefAttr> {
          if (attr == oldAttr)
            return FlatSymbolRefAttr::get(context, newSymName);
          return std::nullopt;
        });
    symbolUpdater.recursivelyReplaceElementsIn(
        userModule, /*replaceAttrs=*/true, /*replaceLocs=*/false,
        /*replaceTypes=*/true);

    op.setSymName(newSymName);
  }

  if (eager) {
    // TODO(#37797): `rawMojoKernel` should already serve as the guard to avoid
    // the duplicated kernel. Somehow it isn't updated correctly while loading
    // multiple libraries. This SymbolTable is used as a short-term fix, but it
    // should be removed and guarded properly with `rawMojoKernel` instead.
    mlir::SymbolTable symtab(module);

    // Update the metadata and clone the operators.
    for (auto op :
         cast<mlir::ModuleOp>(userModule).getOps<mlir::SymbolOpInterface>()) {
      // Don't duplicate.
      if (rawMojoKernel.contains(op.getName()) || symtab.lookup(op.getName()))
        continue;
      Operation *duplicate = op.clone();
      module.getBody()->push_back(duplicate);
      if (auto gen = dyn_cast<KGEN::GeneratorOp>(duplicate))
        rawMojoKernel[gen.getSymName()] = gen;
    }
  } else {
    // Update the metadata only.
    for (KGEN::GeneratorOp gen : toRawMojoKernelRange(userRawMojoKernel)) {
      // Don't add already existing kernels.
      if (rawMojoKernel.contains(gen.getSymName()))
        continue;
      rawMojoKernel[gen.getSymName()] = gen;
    }
  }
}

// We scan the IR to collect all the candidates and then we evaluate them
// based on the metadata collected from the generator.
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

ErrorOrSuccess
MojoLibraryAnalysis::init(ArrayRef<std::string> kernelLibraryPaths) {
  /// Define the mapping of MO operations and intrinsics to the mojo code.
  /// TODO(#23802): Get rid of this mapping (except for types).
  static const llvm::StringMap<StringLiteral> moToMojoKernels{
      MO_TO_MOJO(MO::AddOp, "math::math::add"),
      MO_TO_MOJO(MO::AndOp, "math::math::logical_and"),
      MO_TO_MOJO(MO::CeilOp, "nn::activations::ceil"),
      MO_TO_MOJO(MO::CosOp, "math::math::cos"),
      MO_TO_MOJO(MO::DivOp, "math::math::div"),
      MO_TO_MOJO(MO::ErfOp, "math::math::erf"),
      MO_TO_MOJO(MO::ExpOp, "math::math::exp"),
      MO_TO_MOJO(MO::EqualOp, "math::math::equal"),
      MO_TO_MOJO(MO::FloorOp, "nn::activations::floor"),
      MO_TO_MOJO(MO::GeluOp, "nn::activations::gelu"),
      MO_TO_MOJO(MO::GreaterOp, "math::math::greater"),
      MO_TO_MOJO(MO::GreaterEqualOp, "math::math::greater_equal"),
      MO_TO_MOJO(MO::IsInfOp, "math::math::isinf"),
      MO_TO_MOJO(MO::IsNaNOp, "math::math::isnan"),
      MO_TO_MOJO(MO::LinAlg::BandPartOp,
                 "LinAlg::MatrixBandPart::matrix_band_part"),
      MO_TO_MOJO(MO::LogOp, "math::math::log"),
      MO_TO_MOJO(MO::Log1pOp, "math::math::log1p"),
      MO_TO_MOJO(MO::NotOp, "math::math::logical_not"),
      MO_TO_MOJO(MO::MulOp, "math::math::mul"),
      MO_TO_MOJO(MO::ModOp, "math::math::mod"),
      MO_TO_MOJO(MO::NotEqualOp, "math::math::not_equal"),
      MO_TO_MOJO(MO::ReluOp, "nn::activations::relu"),
      MO_TO_MOJO(MO::RoundOp, "math::math::round"),
      MO_TO_MOJO(MO::RoundEvenOp, "math::math::roundeven"),
      MO_TO_MOJO(MO::RsqrtOp, "math::math::rsqrt"),
      MO_TO_MOJO(MO::SelectOp, "math::math::select"),
      MO_TO_MOJO(MO::SigmoidOp, "nn::activations::sigmoid"),
      MO_TO_MOJO(MO::SinOp, "math::math::sin"),
      MO_TO_MOJO(MO::SubOp, "math::math::sub"),
      MO_TO_MOJO(MO::TanhOp, "nn::activations::tanh"),
      MO_TO_MOJO(MO::ArgNonzeroOp, "nn::arg_nonzero::arg_nonzero"),
      MO_TO_MOJO(MO::TruncOp, "math::math::trunc"),
      MO_TO_MOJO(MO::XorOp, "math::math::logical_xor"),
      MO_TO_MOJO(MO::TileOp, "nn::tile::tile"),

      // Shape functions.
      INTRINSIC_TO_MOJO("batch_matmul_shape",
                        "LinAlg::BatchedMatmul::batched_matmul_shape"),
      INTRINSIC_TO_MOJO("concat_from_list_shape", "nn::concat::concat_shape"),
      INTRINSIC_TO_MOJO("conv_shape", "nn::conv::conv_shape"),
      INTRINSIC_TO_MOJO("conv_transpose_shape",
                        "nn::conv_transpose::conv_transpose_shape"),
      INTRINSIC_TO_MOJO("gather_shape", "nn::gather_scatter::gather_shape"),
      INTRINSIC_TO_MOJO("gather_nd_shape",
                        "nn::gather_scatter::gather_nd_shape"),
      INTRINSIC_TO_MOJO("layer_norm_shape",
                        "nn::normalization::layer_norm_shape"),
      INTRINSIC_TO_MOJO("matrix_solve_shape",
                        "LinAlg::MatrixSolve::matrix_solve_shape"),
      INTRINSIC_TO_MOJO("pack_matmul_b_shape_func",
                        "LinAlg::MatmulPack::pack_matmul_b_shape_func"),
      INTRINSIC_TO_MOJO("pad_shape", "nn::pad::pad_shape"),
      INTRINSIC_TO_MOJO("pool_shape", "nn::pool::pool_shape"),
      INTRINSIC_TO_MOJO("pool_shape_ceil", "nn::pool::pool_shape_ceil"),
      INTRINSIC_TO_MOJO("range_shape", "nn::arange::arange_shape"),
      INTRINSIC_TO_MOJO("reduce_shape", "algorithm::reduction::reduce_shape"),
      INTRINSIC_TO_MOJO("reshape_shape", "nn::reshape::reshape_shape"),
      INTRINSIC_TO_MOJO("tile_shape", "nn::tile::tile_shape"),

      INTRINSIC_TO_MOJO("scatter_shape",
                        "nn::gather_scatter::scatter_elements_shape"),
      INTRINSIC_TO_MOJO("scatter_nd_shape",
                        "nn::gather_scatter::scatter_nd_shape"),
      INTRINSIC_TO_MOJO("slice_shape", "nn::slice::slice_shape"),
      INTRINSIC_TO_MOJO("arg_nonzero_shape",
                        "nn::arg_nonzero::arg_nonzero_shape"),

      // Layout transformations
      INTRINSIC_TO_MOJO("layout_transform_KN_to_KNkni",
                        "LinAlg::MatmulPack::pack_b_ndbuffer"),
      INTRINSIC_TO_MOJO("layout_transform_NK_to_KNkni",
                        "LinAlg::MatmulPack::pack_transposed_b_ndbuffer"),

      INTRINSIC_TO_MOJO("top_k_shape", "nn::topk::top_k_shape"),
  };

  llvm::ArrayRef<llvm::StringRef> sources_tag = {
      "mojo-mogg-bindings", "mojo-primitives-mogg-bindings"};
  llvm::ArrayRef<llvm::StringRef> sources = {
      llvm::StringRef{reinterpret_cast<const char *>(MojoMoggBindings::data),
                      MojoMoggBindings::size},
      llvm::StringRef{
          reinterpret_cast<const char *>(MojoPrimitivesMoggBindings::data),
          MojoPrimitivesMoggBindings::size}};

  llvm::StringRef source{reinterpret_cast<const char *>(MojoMoggBindings::data),
                         MojoMoggBindings::size};
  if (eager) {
    // Eager loading the main mojo bindings.
    module = mlir::parseSourceString<mlir::ModuleOp>(source, context).release();
    GENERICML_ASSERT(module.getOperation());
    block.push_back(module);
    populateMetadata(module, eager, finalizedMojoKernel, rawMojoKernel);
  } else {
    // Lazy loading the main mojo bindings.
    llvm::MemoryBufferRef buffer(source, "mojo-mogg-bindings");
    mlir::BytecodeReader *reader =
        readers.emplace_back(new mlir::BytecodeReader(buffer, config, true))
            .get();

    if (failed(reader->readTopLevel(&block, [&](Operation *op) {
          if (isa<mlir::ModuleOp>(op)) {
            return true;
          }
          return false;
        })))
      return Error("[MojoLibraryAnalysis] Could not load the mojo kernel "
                   "operators from the standard library.");

    module = dyn_cast<mlir::ModuleOp>(block.front());
    if (module == nullptr)
      return Error("[MojoLibraryAnalysis] Top level op is not ModuleOp.");

    symbolNameRemapping.insert({module, {}});
    populateRawMetadata(module, reader, dedicatedReader, rawMojoKernel);
  }

  llvm::StringRef primitive_source{
      reinterpret_cast<const char *>(MojoPrimitivesMoggBindings::data),
      MojoPrimitivesMoggBindings::size};
  if (eager) {
    // Eager loading the main mojo bindings.
    llvm::StringMap<KGEN::GeneratorOp> primitiveRawMojoKernel;
    auto primitiveModule =
        mlir::parseSourceString<mlir::ModuleOp>(primitive_source, context)
            .release();
    GENERICML_ASSERT(primitiveModule.getOperation());
    block.push_back(primitiveModule);
    populateMetadata(primitiveModule, eager, finalizedMojoKernel,
                     primitiveRawMojoKernel);
    updatedLoadedKernelsForCollisions(primitiveModule, primitiveRawMojoKernel);
  } else {
    // Lazy loading the main mojo bindings.
    llvm::MemoryBufferRef buffer(primitive_source,
                                 "mojo-primitives-mogg-bindings");
    mlir::BytecodeReader *reader =
        readers.emplace_back(new mlir::BytecodeReader(buffer, config, true))
            .get();

    if (failed(reader->readTopLevel(&block, [&](Operation *op) {
          if (isa<mlir::ModuleOp>(op)) {
            return true;
          }
          return false;
        })))
      return Error("[MojoLibraryAnalysis] Could not load the mojo kernel "
                   "operators from the standard library.");

    if (!isa<mlir::ModuleOp>(block.back()))
      return Error("[MojoLibraryAnalysis] Top level op is not ModuleOp.");

    auto primitiveModule = &block.back();
    llvm::StringMap<KGEN::GeneratorOp> primitiveRawMojoKernel;
    populateRawMetadata(primitiveModule, reader, dedicatedReader,
                        primitiveRawMojoKernel);
    updatedLoadedKernelsForCollisions(primitiveModule, primitiveRawMojoKernel);
  }

  auto loadError = loadUserProvidedKernels(kernelLibraryPaths);
  GENERICML_ASSERT(!loadError.isError()) << loadError.takeError();

  finalized = eager;

  // Iterate over the available generators and check if each matches a mo
  // operation.
  DenseMap<StringRef, SmallVector<Candidate>> candidates;
  for (auto generator : getRawMojoKernels()) {
    // Check if the generator has a the registration attribute directly
    // attached.
    if (auto regs = generator->getAttrOfType<ArrayAttr>(
            KGEN::MOGGPreElab::kernelRegistrationAttr)) {
      ArrayRef<Attribute> kernels = regs.getValue();
      // Each kernel is stored as a pair of name / priority.
      for (size_t i = 0; i < kernels.size(); i += 2) {
        Candidate candidate = {generator};
        auto asString = cast<StringAttr>(kernels[i]).strref();
        auto intPriority = cast<IntegerAttr>(kernels[i + 1]);
        candidate.priority = intPriority.getInt();
        candidates[asString].push_back(candidate);
      }
    }

    // Shape functions are just a list of operations they are the shape function
    // for.
    if (auto regs = generator->getAttrOfType<ArrayAttr>(
            KGEN::MOGGPreElab::shapeFuncRegistrationAttr)) {
      for (StringRef name : regs.getAsValueRange<StringAttr>())
        registeredShapeFunctionSymbol[name] = generator.getSymName();
    }

    // The generator names are mangled, so we remove the parameter and
    // argument signature.
    StringRef baseName =
        generator.getSymName().split('(').first.split('[').first;
    // Check if the kernel is in the C++ list of intrinsics.
    for (const auto &[moName, mojoName] : moToMojoKernels) {
      if (baseName == mojoName)
        candidates[moName].push_back(Candidate{generator});
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

  return success();
}

MojoLibraryAnalysis::MojoLibraryAnalysis(MLIRContext *ctx, bool eager)
    : eager(eager), context(ctx), config(ctx, false) {}

MojoLibraryAnalysis::~MojoLibraryAnalysis() {
  for (auto &rd : readers) {
    if (failed(rd->finalize([](Operation *op) { return false; })))
      llvm::report_fatal_error("failed to finalize bytecode reader.");
  }
}

mlir::LogicalResult MojoLibraryAnalysis::ready(KGEN::GeneratorOp gen) {
  if (finalized)
    return success();

  if (finalizedMojoKernel.contains(gen))
    return success();

  if (!dedicatedReader.contains(gen))
    return failure();

  mlir::BytecodeReader *reader = dedicatedReader[gen];
  if (succeeded(materialize(gen, reader))) {
    finalizedMojoKernel.insert(gen);
    return replaceSymbols(gen);
  }

  return failure();
}

mlir::LogicalResult MojoLibraryAnalysis::replaceSymbols(KGEN::GeneratorOp gen) {
  // Get the module that holds this op.
  Operation *scope = gen->getParentOp();

  // Each module must appear in the remapping.
  if (!symbolNameRemapping.contains(scope))
    return failure();

  ArrayRef<RenameEntry> newNames = symbolNameRemapping[scope];

  if (newNames.empty())
    return success();

  // Update any references to the old symbol with the new symbol.
  mlir::AttrTypeReplacer symbolUpdater;
  for (const RenameEntry &entry : newNames) {
    symbolUpdater.addReplacement(
        [&](FlatSymbolRefAttr attr) -> std::optional<FlatSymbolRefAttr> {
          if (attr == entry.second)
            return FlatSymbolRefAttr::get(context, entry.first);
          return std::nullopt;
        });
  }
  symbolUpdater.recursivelyReplaceElementsIn(scope,
                                             /*replaceAttrs=*/true,
                                             /*replaceLocs=*/false,
                                             /*replaceTypes=*/true);

  return success();
}

ErrorOr<mlir::ModuleOp> MojoLibraryAnalysis::getMojoBindings() {
  if (!finalized) {
    for (auto &rd : readers) {
      if (failed(rd->finalize()))
        return Error(
            "[MojoLibraryAnalysis] Could not finalize the bytecode readers.");
    }

    // Re-rename symbol names if there conflicts.
    for (auto userModule : block.getOps<mlir::ModuleOp>()) {
      for (auto gen : userModule.getOps<KGEN::GeneratorOp>())
        if (failed(replaceSymbols(gen)))
          return Error(Twine("[MojoLibraryAnalysis] Could not replace symbol "
                             "names in ") +
                       gen.getSymName());
    }

    // Clone all user-provided mojo kernels into the major container module.
    for (auto userModule : block.getOps<mlir::ModuleOp>()) {
      // Skip the first module, which is the one we clone into.
      if (module.getOperation() == userModule.getOperation())
        continue;

      mlir::SymbolTable symtab(module);
      // We clone all symbol ops including GeneratorOp and more (e.g.,
      // PackageLinkOp).
      for (auto userKernel : userModule.getOps<mlir::SymbolOpInterface>()) {
        // We don't clone the existing kernels.
        if (symtab.lookup(userKernel.getName()))
          continue;

        module.getBody()->push_back(userKernel.clone());
      }
    }

    // Update all related metadata.
    finalizedMojoKernel.clear();
    rawMojoKernel.clear();
    populateMetadata(module, eager, finalizedMojoKernel, rawMojoKernel);
    finalized = true;
  }

  return module;
}

Operation *MojoLibraryAnalysis::lookup(StringRef name) {
  if (rawMojoKernel.contains(name))
    return rawMojoKernel[name];
  return nullptr;
}

void MojoLibraryAnalysis::markAllFunctionsAsNotExported() {
  for (auto func : module.getOps<KGEN::FuncOp>())
    func.setNotExported();

  for (auto func : getRawMojoKernels())
    func.setNotExported();
}

} // namespace M::MOGG
