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

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Config.h"
#include "Support/Context.h"
#include "Support/Filesystem/Paths.h"
#include "Support/HashUtils.h"
#include "Support/MDialect/MDialect.h"

// Define the embedded MLIR bytecode arrays in other compilation units to speed
// up compile times.
namespace M::MOGG::Embeddings {
#ifdef MODULAR_MOGG_NO_LINK
#ifdef MODULAR_CMAKE_BUILD
extern llvm::StringRef sourceFilename;
extern llvm::StringRef primitiveSourceFilename;
#endif // MODULAR_CMAKE_BUILD
#else  // MODULAR_MOGG_NO_LINK
extern llvm::StringRef source;
extern llvm::StringRef primitiveSource;
#endif // MODULAR_MOGG_NO_LINK

} // namespace M::MOGG::Embeddings

using namespace M;
using namespace MOGG;

static bool areFunctionsTheSame(KGEN::GeneratorOp gen1,
                                KGEN::GeneratorOp gen2) {
  if (gen1.getSignature() == gen2.getSignature())
    return false;
  return areBlocksSame(*gen1.getBody(), *gen2.getBody());
}

static ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getFile(llvm::StringRef filepath) {
  auto memBufferOr = toModularErrorOr(
      llvm::MemoryBuffer::getFile(filepath, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false));
  if (memBufferOr.isError())
    return Error(memBufferOr.getError() + Twine(" [" + filepath + "]"));

  return memBufferOr;
}

static ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getEmbeddingsSourceBuffer(bool withDebuginfo) {
#ifdef MODULAR_MOGG_NO_LINK
#ifndef MODULAR_CMAKE_BUILD
  ErrorOr<KGEN::MojoConfig> config = KGEN::MojoConfig::open();
  if (failed(config))
    return config.takeError();

  if (withDebuginfo)
    return getFile(config->getMOGGLibDebuginfoPath());
  return getFile(config->getMOGGLibPath());
#else  // !MODULAR_CMAKE_BUILD
  return toModularErrorOr(
      llvm::MemoryBuffer::getFile(Embeddings::sourceFilename,
                                  /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false));
#endif // !MODULAR_CMAKE_BUILD
#else  // MODULAR_MOGG_NO_LINK
  return ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>(
      llvm::MemoryBuffer::getMemBuffer(Embeddings::source,
                                       "embedded MOGG bindings",
                                       /*RequiresNullTerminator=*/false));
#endif // MODULAR_MOGG_NO_LINK
}

static ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getEmbeddingsPrimitiveSourceBuffer(bool withDebuginfo) {
#ifdef MODULAR_MOGG_NO_LINK
#ifndef MODULAR_CMAKE_BUILD
  ErrorOr<KGEN::MojoConfig> config = KGEN::MojoConfig::open();
  if (failed(config))
    return config.takeError();

  if (withDebuginfo)
    return getFile(config->getMOGGPrimitivesLibDebuginfoPath());
  return getFile(config->getMOGGPrimitivesLibPath());
#else  // !MODULAR_CMAKE_BUILD
  return toModularErrorOr(
      llvm::MemoryBuffer::getFile(Embeddings::primitiveSourceFilename,
                                  /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false));
#endif // !MODULAR_CMAKE_BUILD
#else  // MODULAR_MOGG_NO_LINK
  return ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>(
      llvm::MemoryBuffer::getMemBuffer(Embeddings::primitiveSource,
                                       "embedded MOGG Primitives bindings",
                                       /*RequiresNullTerminator=*/false));
#endif // MODULAR_MOGG_NO_LINK
}

static std::string
getMOGGUniqueSymbolName(const std::string &baseName,
                        llvm::StringMap<KGEN::GeneratorOp> &rawMap,
                        unsigned &counter) {
  std::string uniqueName = baseName;
  while (rawMap.contains(uniqueName))
    uniqueName = (baseName + "_" + Twine(counter++)).str();
  return uniqueName;
}

static std::string
getMOGGUniqueSymbolName(const std::string &baseName,
                        llvm::StringMap<KGEN::GeneratorOp> &rawMap) {
  unsigned counter = 0;
  return getMOGGUniqueSymbolName(baseName, rawMap, counter);
}

static ErrorOr<OwningOpRef<ModuleOp>>
parseMojoPackage(MLIRContext *ctx, StringRef path,
                 const std::shared_ptr<llvm::SourceMgr> &sourceMgr) {
  AsyncRT::Runtime &runtime = *loadContext(ctx)->get<AsyncRT::Runtime>();
  OwningOpRef<ModuleOp> ownedModule =
      KGEN::LIT::importStandaloneMojoBinaryPackage(runtime, sourceMgr, ctx,
                                                   path);
  if (!ownedModule)
    return Error("failed to load Mojo package: " + path);

  // Generate a library from the package, ensuring that dependencies are
  // materialized to make it self contained.
  KGEN::KGENCompiler compiler(*(ownedModule->getContext()),
                              KGEN::CompilationOptions());

  if (auto err = compiler.runGenerateLibraryPipeline(*ownedModule))
    return err.takeError();
  return ownedModule;
}

static void
populateMetadata(Operation *module,
                 llvm::StringMap<KGEN::GeneratorOp> &rawMojoKernel) {
  for (KGEN::GeneratorOp op :
       cast<mlir::ModuleOp>(module).getOps<KGEN::GeneratorOp>())
    rawMojoKernel[op.getSymName()] = op;
}

KGEN::GeneratorOp MojoLibraryAnalysis::getRegisteredMojoFunctionFor(
    StringRef name, llvm::StringMap<std::string> &mapping) {
  auto registeredFunction = mapping.find(name.str());
  if (registeredFunction == mapping.end())
    return nullptr;

  auto symbolName = registeredFunction->getValue();
  return getMojoFunctionNamed(symbolName);
}

KGEN::GeneratorOp MojoLibraryAnalysis::getMojoFunctionNamed(StringRef name) {
  if (rawMojoKernel.contains(name))
    return rawMojoKernel.at(name);
  return nullptr;
}

llvm::MemoryBufferRef MojoLibraryAnalysis::internMLIRFile(
    std::unique_ptr<llvm::MemoryBuffer> buffer) {
  unsigned bufferId =
      sourceMgr->AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  return sourceMgr->getMemoryBuffer(bufferId)->getMemBufferRef();
}

ErrorOr<llvm::MemoryBufferRef>
MojoLibraryAnalysis::readAndInternMLIRFile(StringRef filename) {
  auto bufOr = toModularErrorOr(llvm::MemoryBuffer::getFile(filename));
  if (bufOr.isError())
    return bufOr.takeError();
  return internMLIRFile(bufOr.takeValue());
}

bool MojoLibraryAnalysis::kernelHasInputParameter(KGEN::GeneratorOp kernel,
                                                  StringRef parameterName) {
  return llvm::any_of(kernel.getInputParams(), [&](auto param) {
    return KGEN::LIT::demangleIfNeeded(param).getName() == parameterName;
  });
}

ErrorOr<Operation *> MojoLibraryAnalysis::loadMLIRKernels(
    llvm::MemoryBufferRef bufferRef,
    llvm::StringMap<KGEN::GeneratorOp> &kernelMap, const Twine &diag) {
  mlir::BytecodeReader &reader = *readers.emplace_back(new mlir::BytecodeReader(
      bufferRef, config, /*lazyLoad=*/false, sourceMgr));
  if (failed(reader.readTopLevel(
          &block, [&](Operation *op) { return isa<ModuleOp>(op); }))) {
    return Error(
        "[MojoLibraryAnalysis] Could not load the mojo kernel operators in " +
        diag);
  }

  Operation *module = &block.back();
  populateMetadata(module, kernelMap);
  return module;
}

// Load a list of kernels from the provided user libraries.
ErrorOrSuccess MojoLibraryAnalysis::loadUserProvidedKernels(
    ArrayRef<std::string> kernelLibraryPaths) {

  // Load each of the extra user libraries and update the book-keepings.
  for (StringRef pathRef : kernelLibraryPaths) {
    std::string path = pathRef.str();
    llvm::StringMap<KGEN::GeneratorOp> userRawMojoKernel;
    Operation *userModule;

    // Look to see if a debuginfo version of the library was built. It will have
    // the _debuginfo suffix.
    if (withDebuginfo) {
      std::filesystem::path p(path);
      auto debuginfoPath =
          (p.parent_path() /
           (p.stem().string() + "_debuginfo" + p.extension().string()));
      if (std::filesystem::exists(debuginfoPath))
        path = debuginfoPath.string();
    }

    if (Filesystem::isMojoBinaryPackagePath(path)) {
      // Load user library that is a mojo package file.
      auto userModuleOr = parseMojoPackage(context, path, sourceMgr);
      if (userModuleOr.isError())
        return userModuleOr.takeError();

      userModule = userModuleOr.takeValue().release().getOperation();

      if (!userModule)
        return Error("[MojoLibraryAnalysis] Could not load MOGG Mojo "
                     "auxiliary binding library");

      populateMetadata(userModule, userRawMojoKernel);
      // We read the library from `mojopkg` which means we don't need a lazy
      // reader for that. We insert this module op into the block because the
      // block is our centralized container for all module ops (some modules are
      // fully loaded from `mojopkg` files and some are partially loaded from
      // `mlirbc` files).
      block.push_back(userModule);
    } else {
      // Load user library that is a mojo mlibrc file.
      ErrorOr<llvm::MemoryBufferRef> bufOr = readAndInternMLIRFile(path);
      if (failed(bufOr))
        return bufOr.takeError();
      ErrorOr<Operation *> moduleOr =
          loadMLIRKernels(bufOr.takeValue(), userRawMojoKernel, path);
      if (failed(moduleOr))
        return moduleOr.takeError();
      userModule = moduleOr.takeValue();
    }
    updateLoadedKernelsForCollisions(userModule, userRawMojoKernel);
  }

  return success();
}

// Update the state in MJA for collisions etc.
void MojoLibraryAnalysis::updateLoadedKernelsForCollisions(
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
    auto itr = rawMojoKernel.find(newSymName);
    if (itr == rawMojoKernel.end())
      continue;

    // We ignore duplicated functions which are not kernels or shape functions
    // as long as they share the same signature.
    if (!(KGEN::MOGGPreElab::isKernel(op) ||
          KGEN::MOGGPreElab::isV1ShapeFunc(op) ||
          KGEN::MOGGPreElab::isExecuteFunc(op) ||
          KGEN::MOGGPreElab::isShapeFunc(op)) &&
        areFunctionsTheSame(op, itr->second))
      continue;

    bool hasSameSig = op.getSignature() == itr->second.getSignature();
    // Prevent the same kernel from colliding with itself.
    if (hasSameSig && op.getLoc() == rawMojoKernel[newSymName].getLoc() &&
        !isa<mlir::UnknownLoc>(op.getLoc()))
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
}

// We scan the IR to collect all the candidates and then we evaluate them
// based on the metadata collected from the generator.
struct Candidate {
  Candidate(KGEN::GeneratorOp gen) : generator(gen) {}
  KGEN::GeneratorOp generator;
  int64_t priority = -1;
};

ErrorOrSuccess
MojoLibraryAnalysis::init(ArrayRef<std::string> kernelLibraryPaths) {
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOr =
      getEmbeddingsSourceBuffer(withDebuginfo);
  if (failed(bufferOr))
    return Error(Twine("failed to load compiled MOGG.mojo: ") +
                 bufferOr.getError());

  llvm::MemoryBufferRef bufferRef = internMLIRFile(bufferOr.takeValue());
  ErrorOr<Operation *> moduleOr =
      loadMLIRKernels(bufferRef, rawMojoKernel, "the standard library");
  if (failed(moduleOr))
    return moduleOr.takeError();
  module = cast<ModuleOp>(moduleOr.takeValue());

  bufferOr = getEmbeddingsPrimitiveSourceBuffer(withDebuginfo);
  if (failed(bufferOr))
    return Error(Twine("failed to load compiled MOGGPrimitives.mojo: ") +
                 bufferOr.getError());

  bufferRef = internMLIRFile(bufferOr.takeValue());
  llvm::StringMap<KGEN::GeneratorOp> primitiveRawMojoKernel;
  moduleOr = loadMLIRKernels(bufferRef, primitiveRawMojoKernel,
                             "the standard library");
  if (failed(moduleOr))
    return moduleOr.takeError();
  Operation *primitiveModule = moduleOr.takeValue();
  updateLoadedKernelsForCollisions(primitiveModule, primitiveRawMojoKernel);

  if (auto loadError = loadUserProvidedKernels(kernelLibraryPaths))
    return Error(Twine("failed to load user-provided kernels: ") +
                 loadError.getError());

  // Iterate over the available generators and check if each matches a mo
  // operation.
  DenseMap<StringRef, SmallVector<Candidate>> candidates;
  for (auto generator : getRawMojoKernels()) {
    // Skipping functions that aren't exported.
    if (!generator.isExported())
      continue;

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
    } else if (auto kernelName = generator->getAttrOfType<StringAttr>(
                   KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel)) {
      Candidate candidate = {generator};
      candidate.priority = 0;
      candidates[kernelName.strref()].push_back(candidate);
    } else if (auto regs = generator->getAttrOfType<ArrayAttr>(
                   KGEN::MOGGPreElab::shapeFuncRegistrationAttr)) {
      for (StringRef name : regs.getAsValueRange<StringAttr>())
        registeredShapeFunctionSymbol[name] = generator.getSymName();
    } else if (auto name = generator->getAttrOfType<StringAttr>(
                   KGEN::MOGGPreElab::kMOGGShapeFunctionLabel)) {
      registeredShapeFunctionSymbol[name] = generator.getSymName();
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

MojoLibraryAnalysis::MojoLibraryAnalysis(MLIRContext *ctx, bool withDebuginfo)
    : context(ctx), withDebuginfo(withDebuginfo), config(ctx, false),
      sourceMgr(new llvm::SourceMgr()) {}

MojoLibraryAnalysis::~MojoLibraryAnalysis() {
  for (auto &rd : readers) {
    if (failed(rd->finalize([](Operation *op) { return false; })))
      llvm::report_fatal_error("failed to finalize bytecode reader.");
  }
}

LogicalResult MojoLibraryAnalysis::replaceSymbols(KGEN::GeneratorOp gen) {
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

Operation *MojoLibraryAnalysis::lookup(StringRef name) {
  if (rawMojoKernel.contains(name))
    return rawMojoKernel.at(name);
  return nullptr;
}
