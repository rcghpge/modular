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
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
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
  if (auto it = rawMojoKernel.find(name); it != rawMojoKernel.end())
    return it->second;
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
    auto [it, _] = registeredKernelFunctionSymbols.try_emplace(
        name, llvm::StringMap<KGEN::GeneratorOp>{});

    auto [it2, inserted] = it->second.try_emplace(label, generator);
    if (!inserted && override)
      it2->second = generator;
  };

  auto registerKernelCandidate = [&](StringRef name,
                                     int64_t priority) -> ErrorOrSuccess {
    auto [it, inserted] = registeredMojoKernelSymbol.try_emplace(
        name, KernelCandidate(generator, priority));
    if (inserted) {
      // First time we find the kernel, just register it.
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
  auto it = symbolNameRemapping.find(scope);
  if (it == symbolNameRemapping.end())
    return failure();

  ArrayRef<RenameEntry> newNames = it->second;

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

//===----------------------------------------------------------------------===//
// MojoKernelInfos
//===----------------------------------------------------------------------===//

static bool kernelHasExtraOutputShapeArgument(KGEN::GeneratorOp mojoCode) {
  auto kernelName = mojoCode->getAttrOfType<StringAttr>(
      KGEN::MOGGPreElab::kMOGGExecuteFunctionLabel);
  if (!kernelName)
    return false;

  // only reshape / broadcast_to requires an extra argument.
  return kernelName == MO::Static::ReshapeOp::getOperationName() ||
         kernelName == MO::Static::BroadcastToOp::getOperationName();
}

ErrorOr<MojoKernelInfos> MojoKernelInfos::create(KGEN::GeneratorOp mojoCode,
                                                 bool supportMultiVariadics) {
  MojoKernelInfos res;
  res.argumentTypesNames = dyn_cast_or_null<ArrayAttr>(
      mojoCode->getAttr(KGEN::MOGGPreElab::MOGG_ARG_TYPE_NAMES));
  if (!res.argumentTypesNames) {
    return Error("Missing mogg.arg_types_name decorator on the mojo kernel");
  }
  res.resultTypeName = dyn_cast_or_null<StringAttr>(
      mojoCode->getAttr(KGEN::MOGGPreElab::MOGG_RESULT_TYPE_NAME));

  res.mojoCode = mojoCode;
  res.canThrow = mojoCode.getFuncTypeGenerator().getBody().isThrows();
  size_t numOperandsWithoutThrowValues =
      res.argumentTypesNames.size() - res.numErrorHandleResults();
  res.numDPSStyleOutputs = getNumDPSStyleOutputs(mojoCode);
  res.hasExtraOutputShapeArgument = kernelHasExtraOutputShapeArgument(mojoCode);

  // Go through all operands to figure out their I/O kind.
  res.numKernelInputs = 0;
  res.numOpaqueDPSOutputs = 0;
  res.hasVariadicInputs = false;
  res.hasVariadicOutputs = false;

  for (size_t i = 0; i < res.numDPSStyleOutputs; ++i) {
    auto argTypeName = cast<StringAttr>(res.argumentTypesNames[i]);
    if (argTypeName == KGEN::MOGGPreElab::MOJO_VARIADIC_TENSORS_NAME) {
      if (res.hasVariadicOutputs && !supportMultiVariadics)
        return Error("Kernel can't have multiple VariadicTensor outputs");
      res.hasVariadicOutputs = true;
    }
  }

  for (size_t i = res.numDPSStyleOutputs; i < numOperandsWithoutThrowValues;
       ++i) {
    KGEN::ArgConvention argConvention =
        mojoCode.getFuncTypeGenerator().getBody().getArgConvention(i);
    auto argTypeName = cast<StringAttr>(res.argumentTypesNames[i]);
    // Detect MojoCallContextPtr or Tuple[MojoCallContextPtr].

    if (argTypeName ==
        KGEN::MOGGPreElab::
            MOJO_EXTENSIBILITY_API_DEVICE_CONTEXT_PTR_TYPE_NAME) {
      if (res.deviceContextArgIndex.has_value()) {
        return Error("Kernel can't have multiple DevicesContextPtr arguments");
      }
      res.deviceContextArgIndex = i;
      continue;
    }

    if (argTypeName ==
        KGEN::MOGGPreElab::
            MOJO_EXTENSIBILITY_API_DEVICE_CONTEXT_PTR_LIST_TYPE_NAME) {
      if (res.deviceContextListArgIndex.has_value()) {
        return Error(
            "Kernel can't have multiple DevicesContextPtrList arguments");
      }
      res.deviceContextListArgIndex = i;
      continue;
    }

    if (argConvention == KGEN::ArgConvention::ByRefResult) {
      // We use this for result memory opaque type.
      // eg: execute(...) -> MyOpaqueType.
      // Has the follow KGEN signature:
      // execute(..., !kgen.pointer<struct<...> byref_result) -> !kgen.none>.
      if (argTypeName == KGEN::MOGGPreElab::MOJO_DPS_TENSOR_TYPE_NAME ||
          argTypeName == KGEN::MOGGPreElab::MOJO_VARIADIC_TENSORS_NAME) {
        // Tensor types can't be used by return value.
        return Error("The output tensor must be DPS argument, and not a "
                     "function return value");
      }
      res.numOpaqueDPSOutputs += 1;
    } else {
      res.numKernelInputs += 1;
      if (argTypeName == KGEN::MOGGPreElab::MOJO_VARIADIC_TENSORS_NAME) {
        if (res.hasVariadicInputs && !supportMultiVariadics)
          return Error("Kernel can't have multiple VariadicTensor inputs");
        res.hasVariadicInputs = true;
      }
    }
  }

  // Verify the kernel output.
  res.numOpaqueKernelOutputs = 0;
  if (res.resultTypeName) {
    if (res.resultTypeName == KGEN::MOGGPreElab::MOJO_DPS_TENSOR_TYPE_NAME ||
        res.resultTypeName == KGEN::MOGGPreElab::MOJO_VARIADIC_TENSORS_NAME) {
      // Tensor types can't be used by return value.
      return Error("The output tensor must be DPS argument, and not a function "
                   "return value");
    }
    // If it's not a tensor, it must be an opaque result.
    res.numOpaqueKernelOutputs += 1;
  }

  // Verify the position of the device context arguments.
  if (res.deviceContextArgIndex.has_value()) {
    // The DeviceContext argument must be last.
    size_t expectedArgPos = numOperandsWithoutThrowValues - 1;
    if (expectedArgPos != *res.deviceContextArgIndex) {
      return Error(Twine("The DeviceContextPtr argument must be last (#") +
                   Twine(expectedArgPos) + ") but is at position #" +
                   Twine(*res.deviceContextArgIndex));
    }
  }

  if (res.deviceContextListArgIndex.has_value()) {
    // The DeviceContextList argument must be last or before the DeviceContext
    // arg.
    size_t expectedArgPos = numOperandsWithoutThrowValues -
                            (res.deviceContextArgIndex.has_value() ? 2 : 1);
    if (expectedArgPos != *res.deviceContextListArgIndex) {
      return Error(Twine("The DeviceContextListPtr argument must be last or "
                         "before the DeviceContextPtr argument (#") +
                   Twine(expectedArgPos) + ") but is at position #" +
                   Twine(*res.deviceContextListArgIndex));
    }
  }

  return res;
}

void MojoKernelInfos::dump() {
  llvm::errs() << "MojoKernelInfos for " << mojoCode.getSymName() << ":\n";
  llvm::errs() << "canThrow = " << canThrow << "\n";
  llvm::errs() << "deviceContextArgIndex = " << deviceContextArgIndex << "\n";
  llvm::errs() << "deviceContextListArgIndex = " << deviceContextListArgIndex
               << "\n";
  llvm::errs() << "numDPSStyleOutputs = " << numDPSStyleOutputs << "\n";
  llvm::errs() << "numOpaqueDPSOutputs = " << numOpaqueDPSOutputs << "\n";
  llvm::errs() << "numOpaqueKernelOutputs = " << numOpaqueKernelOutputs << "\n";
  llvm::errs() << "numKernelOutputs = " << numKernelOutputs() << "\n";
  llvm::errs() << "numKernelInputs = " << numKernelInputs << "\n";
  llvm::errs() << "hasVariadicInputs = " << hasVariadicInputs << "\n";
  llvm::errs() << "hasVariadicOutputs = " << hasVariadicOutputs << "\n";
  llvm::errs() << "hasExtraOutputShapeArgument = "
               << hasExtraOutputShapeArgument << "\n";
  llvm::errs() << "numErrorHandleResults = " << numErrorHandleResults() << "\n";
  llvm::errs() << mojoCode << "\n\n";
}

//===----------------------------------------------------------------------===//
// KernelCall Verification
//===----------------------------------------------------------------------===//

static ErrorOrSuccess verifyKernelArgsConvention(Operation *op,
                                                 KGEN::GeneratorOp mojoKernel) {
  // Check the arg conventions.
  // !mo.opaque isn't supported for owned arguments.
  for (auto [operandTy, argConv] : llvm::zip(
           op->getOperandTypes(),
           mojoKernel.getFuncTypeGenerator().getBody().getArgConventions())) {
    if (isa<MO::OpaqueType>(operandTy) &&
        (argConv == KGEN::ArgConvention::OwnedMem ||
         argConv == KGEN::ArgConvention::OwnedReg)) {
      return Error(Twine(
          "Owned arguments not supported for opaque types in the kernel " +
          mojoKernel.getSymName()));
    }
  }
  return success();
}

static ErrorOr<size_t> getOperandsCountWithoutChains(TypeRange types,
                                                     bool isInput) {
  size_t res = 0;
  bool hasChain = false;
  for (Type ty : types) {
    if (isa<MO::ChainType>(ty)) {
      if (hasChain)
        return Error(Twine("There can't be more than one ") +
                     (isInput ? "input" : "output") + " chain");
      hasChain = true;
    } else {
      res += 1;
    }
  }
  return res;
}

ErrorOrSuccess
MojoLibraryAnalysis::verifyKernelCallOp(Operation *kernelCallOp) {
  // Find the kernel from the library.
  auto maybeKernel = getMojoKernelTarget(*this, kernelCallOp);
  if (maybeKernel.isError()) {
    return maybeKernel.takeError();
  }
  KGEN::GeneratorOp mojoKernel = maybeKernel.takeValue();

  // Verify the arguments convention of the kernel.
  if (auto err = verifyKernelArgsConvention(kernelCallOp, mojoKernel))
    return err.takeError();

  // Some MO ops (eg AllReduce) supports multi variadic operands.
  // But for custom ops we don't know how the operands are divided between the
  // variadics, so we disable it.
  bool supportMultiVariadics = !isa<MO::CustomOp>(kernelCallOp);

  // Build the kernel informations.
  auto maybeKernelInfos =
      MojoKernelInfos::create(mojoKernel, supportMultiVariadics);
  if (maybeKernelInfos.isError()) {
    return maybeKernelInfos.takeError();
  }
  MojoKernelInfos kernelInfos = maybeKernelInfos.takeValue();

  // Verify the number of inputs.
  // Ignore input chains.
  ErrorOr<size_t> opInputsCountOrErr = getOperandsCountWithoutChains(
      kernelCallOp->getOperandTypes(), /*isInput=*/true);
  if (opInputsCountOrErr.isError())
    return opInputsCountOrErr.takeError();
  size_t opInputsCount = opInputsCountOrErr.takeValue();
  if (kernelInfos.hasExtraOutputShapeArgument)
    opInputsCount += 1;
  if (kernelInfos.hasVariadicInputs) {
    // The number of inputs is a lower bound (variadic size must be >= 1)
    if (opInputsCount < kernelInfos.numKernelInputs) {
      return Error(Twine("Kernel with variadics expects at least ") +
                   Twine(kernelInfos.numKernelInputs) +
                   " inputs but MO op has " + Twine(opInputsCount) +
                   " inputs. You might have some missing inputs and / or an "
                   "empty varidic");
    }
  } else {
    if (opInputsCount != kernelInfos.numKernelInputs) {
      return Error(Twine("Kernel expects ") +
                   Twine(kernelInfos.numKernelInputs) +
                   " inputs but MO op has " + Twine(opInputsCount) + " inputs");
    }
  }

  // Verify the number of outputs.
  // Ignore output chains.
  ErrorOr<size_t> opOutputsCountOrErr = getOperandsCountWithoutChains(
      kernelCallOp->getResultTypes(), /*isInput=*/false);
  if (opOutputsCountOrErr.isError())
    return opOutputsCountOrErr.takeError();
  size_t opOutputsCount = opOutputsCountOrErr.takeValue();
  if (kernelInfos.hasVariadicOutputs) {
    // The number of outputs is a lower bound (variadic size must be >= 1)
    if (opOutputsCount < kernelInfos.numKernelOutputs()) {
      return Error(Twine("Kernel with variadics expects at least ") +
                   Twine(kernelInfos.numKernelOutputs()) +
                   " outputs but MO op has " + Twine(opOutputsCount) +
                   " outputs. You might have some missing outputs and / or an "
                   "empty varidic");
    }
  } else {
    if (opOutputsCount != kernelInfos.numKernelOutputs()) {
      return Error(
          Twine("Kernel expects ") + Twine(kernelInfos.numKernelOutputs()) +
          " outputs but MO op has " + Twine(opOutputsCount) + " outputs");
    }
  }

  return success();
}
