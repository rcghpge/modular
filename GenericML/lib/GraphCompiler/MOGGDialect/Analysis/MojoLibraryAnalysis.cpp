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

#include "AsyncRT/CompilerSupport/Context.h"
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
#include "Kernels/mojo/Mogg/MOGGLoader.h"
#include "Support/AssertStream.h"
#include "Support/Context.h"
#include "Support/Filesystem/Paths.h"
#include "Support/HashUtils.h"

using namespace M;
using namespace MOGG;

// TODO(GEX-2488): The following are Mojo module names hardcoded for
// convenience. We should determine them programmatically instead.
constexpr StringLiteral MOGGKernelAPIModuleName = "MOGGKernelAPI";
constexpr StringLiteral MOGGPrimitivesModuleName = "MOGGPrimitives";

static bool areFunctionsTheSame(KGEN::GeneratorOp gen1,
                                KGEN::GeneratorOp gen2) {
  if (gen1.getFuncTypeGenerator() != gen2.getFuncTypeGenerator())
    return false;

  return mlir::OperationEquivalence::isRegionEquivalentTo(
      &gen1.getBodyRegion(), &gen2.getBodyRegion(),
      /*flags=*/mlir::OperationEquivalence::IgnoreLocations);
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

KGEN::GeneratorOp MojoLibraryAnalysis::getRegisteredMojoFunctionFor(
    StringRef name, StringRef functionLabel) const {
  auto registeredFunctions = registeredKernelFunctionSymbols.find(name);
  if (registeredFunctions == registeredKernelFunctionSymbols.end())
    return {};

  auto registeredFunction = registeredFunctions->second.find(functionLabel);
  if (registeredFunction == registeredFunctions->second.end())
    return {};

  return registeredFunction->getValue();
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

KGEN::GeneratorOp
MojoLibraryAnalysis::getMojoFunctionNamed(StringRef name) const {
  return rawMojoKernelTable->lookup<KGEN::GeneratorOp>(name);
}

ErrorOr<KGEN::SymbolConstantAttr>
MojoLibraryAnalysis::getSpecializedKernel(StringRef name,
                                          ArrayRef<TypedAttr> params) const {
  KGEN::GeneratorOp generator = getRegisteredMojoKernelFor(name);
  if (!generator)
    return Error("Cannot find mojo kernel for " + name);

  auto emitError = [&]() -> mlir::InFlightDiagnostic {
    return generator->emitError("couldn't not retrieve specialized signature");
  };

  KGEN::FuncTypeGeneratorType signature =
      generator.getFuncTypeGenerator().getSpecializedGenerator(
          params,
          /*evaluationContext=*/nullptr, emitError);
  if (!signature)
    return Error("unable to specialize signature for " + name);

  return KGEN::SymbolConstantAttr::get(generator, signature, params);
}

ErrorOr<TypedAttr>
MojoLibraryAnalysis::buildParameterCall(llvm::StringLiteral callee,
                                        ArrayRef<TypedAttr> parameters,
                                        ArrayRef<TypedAttr> operands) const {
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

bool MojoLibraryAnalysis::kernelHasInputParameter(
    KGEN::GeneratorOp kernel, StringRef parameterName) const {
  return llvm::any_of(kernel.getInputParams(), [&](auto param) {
    return KGEN::LIT::demangleParameterName(param.getName()) == parameterName;
  });
}

ErrorOr<OwningOpRef<ModuleOp>> MojoLibraryAnalysis::loadMLIRKernels(
    std::unique_ptr<llvm::MemoryBuffer> bufferRef, const Twine &diag) {
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  unsigned bufferId =
      sourceMgr->AddNewSourceBuffer(std::move(bufferRef), llvm::SMLoc());

  auto internedBuffer = sourceMgr->getMemoryBuffer(bufferId);

  mlir::BytecodeReader reader(*internedBuffer, config, /*lazyLoad=*/false,
                              sourceMgr);

  Block block;
  if (failed(reader.readTopLevel(&block))) {
    return Error(
        "[MojoLibraryAnalysis] Could not load the mojo kernel operators in " +
        diag);
  }

  auto module = cast<ModuleOp>(&block.back());
  module->remove();
  return module;
}

ErrorOr<SmallVector<OwningOpRef<ModuleOp>>>
MojoLibraryAnalysis::loadAllMLIRKernels(ArrayRef<std::string> paths) {
  SmallVector<OwningOpRef<ModuleOp>> loadedModules;
  loadedModules.reserve(paths.size() + 2);

  // Load MOGGKernelAPI.mojo
  if (!useExperimentalMojoPath()) {
    auto buffer = getEmbeddingsNewAPISourceBuffer();
    if (buffer.isError())
      return buffer.takeError();

    auto module = loadMLIRKernels(buffer.takeValue(), "the standard library");
    if (module.isError())
      return module.takeError();

    loadedModules.push_back(module.takeValue());
  }

  { // Load MOGGPrimitives.mojo
    auto buffer = getEmbeddingsPrimitiveSourceBuffer();
    if (buffer.isError())
      return buffer.takeError();

    auto module = loadMLIRKernels(buffer.takeValue(), "the standard library");
    if (module.isError())
      return module.takeError();

    loadedModules.push_back(module.takeValue());
  }

  // Load all user provided library paths.
  for (auto index : llvm::seq(paths.size())) {
    if (auto buffer = loadMojoPackageOrBytecode(paths[index]))
      return buffer.takeError();
    else
      loadedModules.push_back(buffer.takeValue());
  }

  return loadedModules;
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

// Load a list of kernels from the provided user libraries.
ErrorOr<OwningOpRef<ModuleOp>>
MojoLibraryAnalysis::loadMojoPackageOrBytecode(StringRef kernelLibraryPath) {

  // Load the extra user library and update the book-keepings.
  OwningOpRef<ModuleOp> userModule;

  if (Filesystem::isMojoBinaryPackagePath(kernelLibraryPath.str())) {
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();

    // Load user library that is a mojo package file.
    ErrorOr<OwningOpRef<ModuleOp>> userModuleOr = parseMojoPackage(
        getContext(), kernelLibraryPath, sourceMgr, /*generateKGENLib=*/true);
    if (userModuleOr.isError())
      return userModuleOr.takeError();

    OwningOpRef<ModuleOp> parsedModule = userModuleOr.takeValue();

    if (!parsedModule)
      return Error("[MojoLibraryAnalysis] Could not load MOGG Mojo "
                   "auxiliary binding library");

    userModule = std::move(parsedModule);
  } else {
    // Load user library that is a mojo mlibrc file.
    ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufOr =
        getFile(kernelLibraryPath);
    if (bufOr.isError())
      return bufOr.takeError();

    auto moduleOr = loadMLIRKernels(bufOr.takeValue(), kernelLibraryPath);
    if (failed(moduleOr))
      return moduleOr.takeError();

    userModule = moduleOr.takeValue();
  }

  return userModule;
}

ErrorOrSuccess
MojoLibraryAnalysis::addKernelLibraryPath(StringRef kernelLibraryPath) {
  auto moduleOr = loadMojoPackageOrBytecode(kernelLibraryPath);

  if (moduleOr.isError())
    return moduleOr.takeError();

  auto module = moduleOr.takeValue();
  return updateLoadedKernelsForCollisions(*module);
}

// Update the state in MojoLibraryAnalysis for collisions etc.
ErrorOrSuccess
MojoLibraryAnalysis::updateLoadedKernelsForCollisions(ModuleOp userModule) {

  SymbolTable userModuleTable(userModule);
  SymbolTable &rawMojoTable = *rawMojoKernelTable;

  // Generate a new name that is unique to both userModuleTable and
  // rawMojoTable.
  auto newUniqueName = [&](StringAttr oldName) {
    MLIRContext *context = oldName.getContext();
    SmallString<64> prefix = oldName.getValue();
    int uniqueId = 0;
    prefix.push_back('_');

    StringAttr newName;
    do
      newName = StringAttr::get(context, prefix + Twine(uniqueId++));
    while (userModuleTable.lookup(newName) || rawMojoTable.lookup(newName));

    return newName;
  };

  llvm::DenseMap<Attribute, Attribute> renaming;

  // Remap only exposed kernel functions if their name collides with an
  // existing function.
  for (auto op : userModule.getOps<KGEN::GeneratorOp>()) {
    // We will insert a slightly mangled version of the current name if it
    // already exists in the mlir file we intend to compile.
    auto oldSymName = op.getSymNameAttr();

    // If there's no collision we can copy directly obviously.
    auto existingKernel = rawMojoTable.lookup<KGEN::GeneratorOp>(oldSymName);
    if (!existingKernel)
      continue;

    // We ignore duplicated functions.
    // TODO (GEX-1925): Correctly check if the functions are identical.
    if (areFunctionsTheSame(op, existingKernel))
      continue;

    // Create a new symbol which is unique in both files.
    auto newSymName = newUniqueName(op.getSymNameAttr());
    op.setSymNameAttr(newSymName);

    auto oldSymbolRef = FlatSymbolRefAttr::get(oldSymName);
    auto newSymbolRef = FlatSymbolRefAttr::get(newSymName);
    renaming[oldSymbolRef] = newSymbolRef;
  }

  if (!renaming.empty()) {
    // The symbol table infrastructure does not descend into types when renaming
    // a symbol. Use the replacer to perform renaming in type positions.
    mlir::AttrTypeReplacer symbolReplacer;
    symbolReplacer.addReplacement(
        [renaming = std::move(renaming)](
            FlatSymbolRefAttr ref) -> std::optional<Attribute> {
          if (auto it = renaming.find(ref); it != renaming.end())
            return it->second;
          return std::nullopt;
        });

    symbolReplacer.recursivelyReplaceElementsIn(
        userModule, /*replaceAttrs=*/true, /*replaceLocs=*/false,
        /*replaceTypes=*/true);
  }

  // Update the metadata and clone the operators.
  for (auto op : llvm::make_early_inc_range(
           userModule.getOps<mlir::SymbolOpInterface>())) {
    StringAttr oldName = op.getNameAttr();

    // Don't duplicate.
    if (rawMojoTable.lookup(oldName))
      continue;

    // Move the operation to the end of the module and insert into the symbol
    // table. The operation has to be unlinked from its parent before symbol
    // table insertion.
    op->moveBefore(module->getBody(), module->getBody()->end());

    [[maybe_unused]] StringAttr newName = rawMojoTable.insert(op);
    ASSERT_STREAM(oldName == newName, "no renaming should occur at this point");

    if (auto gen = dyn_cast<KGEN::GeneratorOp>(op.getOperation())) {
      if (auto updateErr = updateRegisteredKernelsMap(
              gen, /*isGeneratorFromStandardLibrary*/ false))
        return updateErr;
    }
  }

  return success();
}

// If the generator is from the standard library and it has a default priority
// level, then we want to give it a higher priority, because it should be used
// instead of a pre-built kernel library. This allows the user to modify the
// open source standard library code and have it used by the graph compiler.
static bool shouldIncreasePriority(bool isStandardLibrary, int priority) {
  return isStandardLibrary && priority == -1;
}

ErrorOrSuccess MojoLibraryAnalysis::updateRegisteredKernelsMap(
    KGEN::GeneratorOp generator, bool isGeneratorFromStandardLibrary) {
  auto registerKernelFunction = [&](StringRef name, StringRef label,
                                    bool override = true) {
    auto [it, _] = registeredKernelFunctionSymbols.try_emplace(
        name, llvm::StringMap<KGEN::GeneratorOp>{});

    auto [it2, inserted] = it->second.try_emplace(label, generator);
    if (!inserted && override)
      it2->second = generator;
  };

  auto registerKernelCandidate = [&](StringRef name,
                                     int64_t priority) -> ErrorOrSuccess {
    if (shouldIncreasePriority(isGeneratorFromStandardLibrary, priority))
      priority = 1;

    auto [it, inserted] = registeredMojoKernelSymbol.try_emplace(
        name, KernelCandidate(generator, priority));
    if (inserted) {
      // First time we find the kernel, just register it.
      return success();
    }

    // Returns an error if two kernels have the same priority.
    int64_t oldPriority = it->getValue().priority;
    if (oldPriority == priority) {
      auto diag = generator->emitError() << "override of operation named "
                                         << name << " has multiple candidates";
      diag.attachNote(it->getValue().generator.getLoc())
          << "override with priority " << priority;
      return Error("Operation overriding error");
    }

    // Replace it if the new kernel has an higher priority.
    if (priority > oldPriority)
      it->getValue() = KernelCandidate(generator, priority);
    return success();
  };

  // Detect foreach kernel, treat it specially.
  if (generator->hasAttr(KGEN::MOGGPreElab::MOGG_INTRINSIC_ELEMWISE_FOR_EACH)) {
    if (!foreachKernel) {
      foreachKernel = generator;
    } else if (foreachKernel.getLoc() == generator.getLoc()) {
      return Error(
          llvm::formatv("Multiple elementwise foreach kernels detected.\n"
                        "  Previous: {0} at {1}\n"
                        "  New: {2} at {3}",
                        foreachKernel.getSymName(), foreachKernel.getLoc(),
                        generator.getSymName(), generator.getLoc())
              .str());
    } else if (isGeneratorFromStandardLibrary) {
      // There is an existing kernel, but this one is a different location.
      // Favor the one from the standard library then.
      foreachKernel = generator;
    }
  }

  if (generator->hasAttr(KGEN::MOGGPreElab::MOGG_INTRINSIC_VIEW_MATERIALIZE)) {
    if (!viewMaterializeKernel) {
      viewMaterializeKernel = generator;
    } else if (viewMaterializeKernel.getLoc() == generator.getLoc()) {
      return Error(llvm::formatv("Multiple materialize view kernels detected.\n"
                                 "  Previous: {0} at {1}\n"
                                 "  New: {2} at {3}",
                                 viewMaterializeKernel.getSymName(),
                                 viewMaterializeKernel.getLoc(),
                                 generator.getSymName(), generator.getLoc())
                       .str());
    } else if (isGeneratorFromStandardLibrary) {
      // There is an existing kernel, but this one is a different location.
      // Favor the one from the standard library then.
      viewMaterializeKernel = generator;
    }
  }

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
          KGEN::MOGGPreElab::kMOGGUpdateViewFunctionLabel)) {
    registerKernelFunction(name,
                           KGEN::MOGGPreElab::kMOGGUpdateViewFunctionLabel);
  }
  if (auto name = generator->getAttrOfType<StringAttr>(
          KGEN::MOGGPreElab::kMOGGElementwiseFunctionLabel)) {
    registerKernelFunction(name,
                           KGEN::MOGGPreElab::kMOGGElementwiseFunctionLabel);
  }
  return success();
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
  auto loadedModulesOr = loadAllMLIRKernels(kernelLibraryPaths);

  if (loadedModulesOr.isError())
    return loadedModulesOr.takeError();

  SmallVector<OwningOpRef<ModuleOp>> loadedModules =
      loadedModulesOr.takeValue();

  // TODO: Currently For EmitMojo, MOGGKernelAPI is loaded using LIT, but
  // MOGGPrimitives using KGEN. Switch to load everything using LIT.

  bool loadLITModules = useExperimentalMojoPath();

  // Standard library is loaded with KGEN only if emit mojo is disabled.
  ASSERT_STREAM(loadLITModules || loadedModules.size() >= 2,
                "Must contains standard library and MOGG primitives");

  // Pull out the loaded standard library.
  module = std::move(loadedModules[0]);
  loadedModules.erase(loadedModules.begin());

  // Initialize the symbol table over the standard library module.
  rawMojoKernelTable.emplace(*module);

  for (auto generator : getRawMojoKernels()) {
    // Mark standard library generators
    if (auto updateErr = updateRegisteredKernelsMap(
            generator, /*isGeneratorFromStandardLibrary*/ !loadLITModules))
      return updateErr;
  }

  // Process all remaining modules.
  for (auto [index, loadedModule] : llvm::enumerate(loadedModules)) {
    if (auto updateErr = updateLoadedKernelsForCollisions(*loadedModule))
      return Error(Twine("failed to load kernel library: ") +
                   updateErr.getError());
  }

  // Take care of loading all LIT kernels if needed
  if (loadLITModules) {
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
  }

  return success();
}

MojoLibraryAnalysis::MojoLibraryAnalysis(MLIRContext *ctx)
    : config(ctx, /*verifyAfterParse=*/false) {}

MojoLibraryAnalysis::~MojoLibraryAnalysis() = default;

KGEN::GeneratorOp
MojoLibraryAnalysis::getRegisteredMojoKernelFor(StringRef name) const {
  auto registeredFunction = registeredMojoKernelSymbol.find(name);
  if (registeredFunction == registeredMojoKernelSymbol.end())
    return nullptr;

  return registeredFunction->getValue().generator;
}

KGEN::LIT::FnOp
MojoLibraryAnalysis::getRegisteredLITMojoKernelFor(StringRef name) const {
  auto registeredFunction = registeredLITMojoKernelSymbol.find(name);
  if (registeredFunction == registeredLITMojoKernelSymbol.end())
    return nullptr;

  return registeredFunction->getValue().fun;
}

KGEN::GeneratorOp
MojoLibraryAnalysis::getRegisteredMojoShapeFunctionFor(StringRef name) const {
  return getRegisteredMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGShapeFunctionLabel);
}

KGEN::LIT::FnOp MojoLibraryAnalysis::getRegisteredLITMojoShapeFunctionFor(
    StringRef name) const {
  return getRegisteredLITMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGShapeFunctionLabel);
}

KGEN::GeneratorOp MojoLibraryAnalysis::getRegisteredMojoUpdateViewFunctionFor(
    StringRef name) const {
  return getRegisteredMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGUpdateViewFunctionLabel);
}

KGEN::LIT::FnOp MojoLibraryAnalysis::getRegisteredLITMojoUpdateViewFunctionFor(
    StringRef name) const {
  return getRegisteredLITMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGUpdateViewFunctionLabel);
}

KGEN::GeneratorOp
MojoLibraryAnalysis::getRegisteredMojoExecuteFunctionFor(StringRef name) const {
  return getRegisteredMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel);
}

KGEN::LIT::FnOp MojoLibraryAnalysis::getRegisteredLITMojoExecuteFunctionFor(
    StringRef name) const {
  return getRegisteredLITMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel);
}

KGEN::GeneratorOp MojoLibraryAnalysis::getRegisteredMojoElementwiseFunctionFor(
    StringRef name) const {
  return getRegisteredMojoFunctionFor(
      name, KGEN::MOGGPreElab::kMOGGElementwiseFunctionLabel);
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

using KernelAdaptor = KGEN::MOGGPreElab::MojoKernelAdaptor<KGEN::GeneratorOp>;

ErrorOrSuccess
MojoLibraryAnalysis::validateCustomOpParameters(MO::CustomOp customOp) const {
  // TODO: Reimplement the validation for LIT.
  if (useExperimentalMojoPath())
    return success();

  // Find the kernel from the library.
  auto maybeAdaptor = getMojoKernelAdaptor<KGEN::GeneratorOp>(*this, customOp);
  if (maybeAdaptor.isError())
    return maybeAdaptor.takeError();

  auto adaptor = maybeAdaptor.takeValue();
  auto mojoCode = adaptor.executeFunction->mojoCode;

  auto parameterTypeNames = mojoCode->getAttrOfType<ArrayAttr>(
      KGEN::MOGGPreElab::MOGG_INPUT_PARAM_TYPES);

  struct MojoParameterAdaptor {
    KGEN::ParamDeclAttr attr;
    StringAttr typeName;
  };
  llvm::DenseMap<StringAttr, MojoParameterAdaptor> paramDecls;

  for (auto [param, value] :
       llvm::zip(mojoCode.getInputParams(), parameterTypeNames))
    paramDecls[param.getName()] = {param, dyn_cast<StringAttr>(value)};

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
      [[maybe_unused]] auto asString = dyn_cast<StringAttr>(mlirParamAttr);
      ASSERT_STREAM(asString != nullptr,
                    "failed to convert custom op attribute "
                        << mlirParamAttr
                        << " to expected mojo kernel param type " << kgenParamTy
                        << ": expected string type");
    } else if (isa<KGEN::DTypeType>(kgenParamTy)) {
      auto asType = dyn_cast<TypeAttr>(mlirParamAttr);
      if (asType == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected a DType",
            customOp.getSymbol(), customParamDecl.getName(), mlirParamAttr));
      }
    } else if (typeName &&
               typeName ==
                   KGEN::MOGGPreElab::MOJO_INTERNAL_STATIC_STRING_TYPE_NAME) {
      [[maybe_unused]] auto asString = dyn_cast<StringAttr>(mlirParamAttr);
      if (asString == nullptr) {
        return Error(llvm::formatv(
            "failed to convert custom op '{}' attribute '{}' with value '{}' "
            "to the proper Mojo parameter type: expected string type",
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

template <class FuncT>
static ErrorOrSuccess
verifyMOOpMatchesKernelImpl(const MojoLibraryAnalysis &libraryAnalysis,
                            Operation *op) {
  // Find the kernel from the library.
  auto maybeAdaptor = getMojoKernelAdaptor<FuncT>(libraryAnalysis, op);
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
  if (useExperimentalMojoPath())
    return verifyMOOpMatchesKernelImpl<KGEN::LIT::FnOp>(*this, op);
  else
    return verifyMOOpMatchesKernelImpl<KGEN::GeneratorOp>(*this, op);
}

ErrorOr<std::shared_ptr<MojoLibraryAnalysis>>
M::MOGG::MojoLibraryAnalysis::fromPaths(MLIRContext *ctx,
                                        ArrayRef<std::string> paths) {
  auto result = std::make_shared<MOGG::MojoLibraryAnalysis>(ctx);
  if (auto status = result->init(paths))
    return status.takeError();
  return result;
}
