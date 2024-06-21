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
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Config.h"
#include "Support/Context.h"
#include "Support/Filesystem/Paths.h"
#include "Support/MDialect/MDialect.h"

namespace {
#include MOGG_MOJO_BINDING_H
#include MOGG_PRIMITIVES_MOJO_BINDING_H
} // namespace

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

ErrorOr<OwningOpRef<ModuleOp>>
parseMojoPackage(MLIRContext *ctx, StringRef path,
                 const std::shared_ptr<llvm::SourceMgr> &sourceMgr) {
  LLCL::Runtime &runtime = *loadContext(ctx)->get<LLCL::Runtime>();
  OwningOpRef<ModuleOp> ownedModule =
      KGEN::LIT::importStandaloneMojoBinaryPackage(runtime, sourceMgr, ctx,
                                                   path);
  if (!ownedModule)
    return Error("failed to load Mojo package: " + path);

  // Generate a library from the package, ensuring that dependencies are
  // materialized to make it self contained.
  KGEN::KGENCompiler compiler(*(ownedModule->getContext()),
                              KGEN::CompilationOptions());

  if (auto err = compiler.runGenerateLibraryPipeline(
          *ownedModule, /*materializeDependencies=*/true))
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
    ArrayRef<std::string> kernelLibraryPaths, bool eager) {

  // Load each of the extra user libraries and update the book-keepings.
  for (StringRef path : kernelLibraryPaths) {
    llvm::StringMap<KGEN::GeneratorOp> userRawMojoKernel;
    Operation *userModule;

    if (Filesystem::isMojoBinaryPackagePath(path.str())) {
      // Load user library that is a mojo package file.
      auto userModuleOr = parseMojoPackage(context, path, sourceMgr);
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
            readers
                .emplace_back(
                    new mlir::BytecodeReader(buffer, config, true, sourceMgr))
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
    updateLoadedKernelsForCollisions(userModule, userRawMojoKernel, eager);
  }

  return success();
}

// Update the state in MJA for collisions etc.
void MojoLibraryAnalysis::updateLoadedKernelsForCollisions(
    Operation *userModule,
    llvm::StringMap<KGEN::GeneratorOp> &userRawMojoKernel, bool eager) {

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

    bool hasSameSig = op.getSignature() == itr->second.getSignature();
    // We ignore duplicated functions which are not kernels or shape functions
    // as long as they share the same signature.
    if (!(KGEN::MOGGPreElab::isKernel(op) ||
          KGEN::MOGGPreElab::isV1ShapeFunc(op)) &&
        hasSameSig)
      continue;

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

ErrorOrSuccess
MojoLibraryAnalysis::init(ArrayRef<std::string> kernelLibraryPaths,
                          bool eager) {

  llvm::StringRef source{reinterpret_cast<const char *>(MojoMoggBindings::data),
                         MojoMoggBindings::size};
  if (eager) {
    // Eager loading the main mojo bindings.
    module = mlir::parseSourceString<mlir::ModuleOp>(source, context).release();
    assert(module.getOperation());
    block.push_back(module);
    populateMetadata(module, eager, finalizedMojoKernel, rawMojoKernel);
  } else {
    // Lazy loading the main mojo bindings.
    llvm::MemoryBufferRef buffer(source, "mojo-mogg-bindings");
    mlir::BytecodeReader *reader = readers
                                       .emplace_back(new mlir::BytecodeReader(
                                           buffer, config, true, sourceMgr))
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
    assert(primitiveModule.getOperation());
    block.push_back(primitiveModule);
    populateMetadata(primitiveModule, eager, finalizedMojoKernel,
                     primitiveRawMojoKernel);
    updateLoadedKernelsForCollisions(primitiveModule, primitiveRawMojoKernel,
                                     eager);
  } else {
    // Lazy loading the main mojo bindings.
    llvm::MemoryBufferRef buffer(primitive_source,
                                 "mojo-primitives-mogg-bindings");
    mlir::BytecodeReader *reader = readers
                                       .emplace_back(new mlir::BytecodeReader(
                                           buffer, config, true, sourceMgr))
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
    updateLoadedKernelsForCollisions(primitiveModule, primitiveRawMojoKernel,
                                     eager);
  }

  auto loadError = loadUserProvidedKernels(kernelLibraryPaths, eager);
  ASSERT_STREAM(!loadError.isError(), << loadError.takeError());

  finalized = eager;

  // Iterate over the available generators and check if each matches a mo
  // operation.
  DenseMap<StringRef, SmallVector<Candidate>> candidates;
  for (auto generator : getRawMojoKernels()) {
    // Don't scan internal functions which aren't exported.
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
    }

    // Shape functions are just a list of operations they are the shape function
    // for.
    if (auto regs = generator->getAttrOfType<ArrayAttr>(
            KGEN::MOGGPreElab::shapeFuncRegistrationAttr)) {
      for (StringRef name : regs.getAsValueRange<StringAttr>())
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

MojoLibraryAnalysis::MojoLibraryAnalysis(MLIRContext *ctx)
    : context(ctx), config(ctx, false), sourceMgr(new llvm::SourceMgr()) {}

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

    // Re-rename symbol names if there are conflicts.
    for (auto userModule : block.getOps<mlir::ModuleOp>()) {
      for (auto gen : userModule.getOps<KGEN::GeneratorOp>()) {
        if (failed(replaceSymbols(gen))) {
          return Error(Twine("[MojoLibraryAnalysis] Could not replace symbol "
                             "names in ") +
                       gen.getSymName());
        }
      }
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
    populateMetadata(module, true, finalizedMojoKernel, rawMojoKernel);
    finalized = true;
  }

  return module;
}

Operation *MojoLibraryAnalysis::lookup(StringRef name) {
  if (rawMojoKernel.contains(name))
    return rawMojoKernel[name];
  return nullptr;
}

} // namespace M::MOGG
