//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Kernels/mojo/Mogg/MOGGLoader.h"
#include "Support/Context.h"
#include "Support/Filesystem/Paths.h"
#include "Support/HashUtils.h"

using namespace M;
using namespace MOGG;

static bool areFunctionsTheSame(KGEN::GeneratorOp gen1,
                                KGEN::GeneratorOp gen2) {
  if (gen1.getFuncTypeGenerator() != gen2.getFuncTypeGenerator())
    return false;
  return areBlocksSame(*gen1.getBody(), *gen2.getBody());
}

[[maybe_unused]] static ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getFile(llvm::StringRef filepath) {
  auto memBufferOr = toModularErrorOr(
      llvm::MemoryBuffer::getFile(filepath, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false));
  if (memBufferOr.isError())
    return Error(memBufferOr.getError() + Twine(" [" + filepath + "]"));

  return memBufferOr;
}

static ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getEmbeddingsNewAPISourceBuffer() {
  return ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>(
      llvm::MemoryBuffer::getMemBuffer(Embeddings::moggKernelSource(),
                                       "embedded MOGG bindings",
                                       /*RequiresNullTerminator=*/false));
}

static ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getEmbeddingsPrimitiveSourceBuffer() {
  return ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>(
      llvm::MemoryBuffer::getMemBuffer(Embeddings::primitiveSource(),
                                       "embedded MOGG Primitives bindings",
                                       /*RequiresNullTerminator=*/false));
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

KGEN::GeneratorOp
MojoLibraryAnalysis::getRegisteredMojoFunctionFor(StringRef name,
                                                  StringRef functionLabel) {
  auto registeredFunctions = registeredKernelFunctionSymbols.find(name);
  if (registeredFunctions == registeredKernelFunctionSymbols.end())
    return {};

  auto registeredFunction = registeredFunctions->second.find(functionLabel);
  if (registeredFunction == registeredFunctions->second.end())
    return {};

  return registeredFunction->getValue();
}

KGEN::GeneratorOp MojoLibraryAnalysis::getMojoFunctionNamed(StringRef name) {
  if (rawMojoKernel.contains(name))
    return rawMojoKernel.at(name);
  return nullptr;
}

ErrorOr<KGEN::SymbolConstantAttr>
MojoLibraryAnalysis::getSpecializedKernel(StringRef name,
                                          ArrayRef<TypedAttr> params) {
  KGEN::GeneratorOp generator = getRegisteredMojoKernelFor(name);
  if (!generator)
    return Error("Cannot find mojo kernel for " + name);

  auto emitError = [&]() -> mlir::InFlightDiagnostic {
    return generator->emitError("couldn't not retrieve specialized signature");
  };

  KGEN::FuncTypeGeneratorType signature =
      generator.getFuncTypeGenerator().getSpecializedGenerator(params,
                                                               emitError);
  if (!signature)
    return Error("unable to specialize signature for " + name);

  return KGEN::SymbolConstantAttr::get(generator, signature, params);
}

ErrorOr<TypedAttr>
MojoLibraryAnalysis::buildParameterCall(llvm::StringLiteral callee,
                                        ArrayRef<TypedAttr> parameters,
                                        ArrayRef<TypedAttr> operands) {
  ErrorOr<KGEN::SymbolConstantAttr> maybeSymbol =
      getSpecializedKernel(callee, parameters);

  if (failed(maybeSymbol))
    return maybeSymbol.takeError();

  SmallVector<TypedAttr> allOperands;
  allOperands.push_back(*maybeSymbol);
  allOperands.append(operands.begin(), operands.end());

  auto resultTy = maybeSymbol->getType().getBody().getResults()[0];
  return KGEN::ParamOperatorAttr::get(KGEN::POC::Apply, allOperands, resultTy);
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
ErrorOrSuccess MojoLibraryAnalysis::addKernelLibraryPath(
    const std::string &kernelLibraryPath) {

  // Load the extra user library and update the book-keepings.
  llvm::StringMap<KGEN::GeneratorOp> userRawMojoKernel;
  Operation *userModule;

  if (Filesystem::isMojoBinaryPackagePath(kernelLibraryPath)) {
    // Load user library that is a mojo package file.
    auto userModuleOr = parseMojoPackage(context, kernelLibraryPath, sourceMgr);
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
    ErrorOr<llvm::MemoryBufferRef> bufOr =
        readAndInternMLIRFile(kernelLibraryPath);
    if (failed(bufOr))
      return bufOr.takeError();
    ErrorOr<Operation *> moduleOr = loadMLIRKernels(
        bufOr.takeValue(), userRawMojoKernel, kernelLibraryPath);
    if (failed(moduleOr))
      return moduleOr.takeError();
    userModule = moduleOr.takeValue();
  }
  if (auto updateErr =
          updateLoadedKernelsForCollisions(userModule, userRawMojoKernel))
    return Error(Twine("failed to load kernel library ") + kernelLibraryPath +
                 ":" + updateErr.getError());

  return success();
}

// Update the state in MJA for collisions etc.
ErrorOrSuccess MojoLibraryAnalysis::updateLoadedKernelsForCollisions(
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

    // We ignore duplicated functions.
    // TODO (GEX-1925): Correctly check if the functions are identical.
    if (areFunctionsTheSame(op, itr->second))
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
    if (auto gen = dyn_cast<KGEN::GeneratorOp>(duplicate)) {
      rawMojoKernel[gen.getSymName()] = gen;
      if (auto updateErr = updateRegisteredKernelsMap(gen))
        return updateErr;
    }
  }

  return success();
}

ErrorOrSuccess
MojoLibraryAnalysis::updateRegisteredKernelsMap(KGEN::GeneratorOp generator) {
  auto registerKernelFunction = [&](StringRef name, StringRef label,
                                    bool override = true) {
    if (!registeredKernelFunctionSymbols.contains(name))
      registeredKernelFunctionSymbols[name] = {};

    if (override || !registeredKernelFunctionSymbols[name].contains(label))
      registeredKernelFunctionSymbols[name][label] = generator;
  };

  auto registerKernelCandidate = [&](StringRef name,
                                     int64_t priority) -> ErrorOrSuccess {
    auto it = registeredMojoKernelSymbol.find(name);
    if (it == registeredMojoKernelSymbol.end()) {
      // First time we find the kernel, just register it.
      registeredMojoKernelSymbol.insert(
          std::make_pair(name, KernelCandidate(generator, priority)));
      return success();
    }

    // Returns an error if two kernels have the same priority.
    int64_t oldPriority = it->getValue().priority;
    if (oldPriority == priority) {
      auto diag = mlir::emitError(generator.getLoc(),
                                  "override of operation has multiple "
                                  "candidates with the same priority.");
      diag.attachNote(it->getValue().generator.getLoc())
          << "override with priority " << priority;
      return Error("Operation overriding error");
    }

    // Replace it if the new kernel has an higher priority.
    if (priority > oldPriority)
      it->getValue().priority = priority;
    return success();
  };

  // Skipping functions that aren't exported.
  if (!generator.isExported())
    return success();

  // Check if the generator has a the registration attribute directly
  // attached.
  if (auto regs = generator->getAttrOfType<ArrayAttr>(
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
  if (auto kernelName = generator->getAttrOfType<StringAttr>(
          KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel)) {
    if (auto registerErr = registerKernelCandidate(kernelName, /*priority=*/0))
      return registerErr;
    registerKernelFunction(kernelName,
                           KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel);
  }
  if (auto name = generator->getAttrOfType<StringAttr>(
          KGEN::MOGGPreElab::kMOGGShapeFunctionLabel)) {
    registerKernelFunction(name, KGEN::MOGGPreElab::kMOGGShapeFunctionLabel);
  }
  if (auto name = generator->getAttrOfType<StringAttr>(
          KGEN::MOGGPreElab::kMOGGPyTorchFallbackFunctionLabel)) {
    registerKernelFunction(
        name, KGEN::MOGGPreElab::kMOGGPyTorchFallbackFunctionLabel);
  }

  return success();
}

ErrorOrSuccess
MojoLibraryAnalysis::init(ArrayRef<std::string> kernelLibraryPaths) {
  // Load MOGGKernelAPI.mojo [new API] / MOGG.mojo [old API]
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOr =
      getEmbeddingsNewAPISourceBuffer();
  if (failed(bufferOr))
    return Error(Twine("failed to load compiled main module: ") +
                 bufferOr.getError());
  llvm::MemoryBufferRef bufferRef = internMLIRFile(bufferOr.takeValue());
  ErrorOr<Operation *> moduleOr =
      loadMLIRKernels(bufferRef, rawMojoKernel, "the standard library");
  if (failed(moduleOr))
    return moduleOr.takeError();
  module = cast<ModuleOp>(moduleOr.takeValue());
  for (auto generator : getRawMojoKernels())
    if (auto updateErr = updateRegisteredKernelsMap(generator))
      return updateErr;

  // Load MOGGPrimitives.mojo
  bufferOr = getEmbeddingsPrimitiveSourceBuffer();
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
  if (auto updateErr = updateLoadedKernelsForCollisions(primitiveModule,
                                                        primitiveRawMojoKernel))
    return Error(Twine("failed to load compiled MOGGPrimitives.mojo: ") +
                 updateErr.getError());

  // Load the extra user kernel libraries.
  for (const std::string &libraryPath : kernelLibraryPaths)
    if (auto loadErr = addKernelLibraryPath(libraryPath))
      return loadErr;

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
  auto it = rawMojoKernel.find(name);
  if (it == rawMojoKernel.end())
    return nullptr;

  return it->second;
}

KGEN::GeneratorOp
MojoLibraryAnalysis::getRegisteredMojoKernelFor(StringRef name) {
  auto registeredFunction = registeredMojoKernelSymbol.find(name);
  if (registeredFunction == registeredMojoKernelSymbol.end())
    return nullptr;

  return registeredFunction->getValue().generator;
}

KGEN::GeneratorOp
MojoLibraryAnalysis::getRegisteredMojoShapeFunctionFor(StringRef name) {
  return getRegisteredMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGShapeFunctionLabel);
}
