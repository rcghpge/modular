//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/CompiledKernelsContext.h"
#include "GenericML/Support/List.h"
#include "KGEN/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "Support/AssertStream.h"
#include "Support/FileSystemExtras.h"
#include "Support/Preprocessor.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "compiled-kernels-context"

using namespace M;
using namespace M::GML;

//===----------------------------------------------------------------------===//
// CompiledKernelCall
//===----------------------------------------------------------------------===//

ErrorOrSuccess CompiledKernelABI::CompiledKernelCall::pushObject(
    AnyAsyncValueRef input, char inOutType, StringRef summary,
    const EncodedLocation &loc) {
  if (input.isType<TensorBufferRef>()) {
    const void *ptr = input.get<TensorBufferRef>().getBuffer();
    // Forget the const here since the ABI mixes input and output.
    inputPointers.push_back(const_cast<void *>(ptr));

  } else if (input.isType<GML::List>()) {
    auto &runtimeList = input.get<GML::List>();
    auto &abiTensors = tensorsMemory.emplace_back();

    // Convert individual elements to their corresponding ABI structures.
    // NOTE: This is currently not closed under all argument types and cannot
    //       be extended by subclasses.
    for (GML::Variant &ref : runtimeList.getElements()) {
      if (ref.getValue().isType<Tensor>()) {
        auto &runtimeTensor = ref.getValue().get<Tensor>();
        const TensorShape &shape = runtimeTensor.getShape();
        ssize_t *dims;
        // If the Tensor is a scalar, need to upscale to 1D to mojo
        // can reason about it
        if (shape.getRank() == 0)
          dims = dimsMemory.emplace_back(1, 1).data();
        else
          dims = dimsMemory.emplace_back(shape.begin(), shape.end()).data();
        void *data = const_cast<void *>(runtimeTensor.getBuffer());
        abiTensors.emplace_back(dims, data);
      } else {
        return Error(Twine("Invalid list element type: ") +
                     ref.getValue().getPointer()->getTypeID().getTypeName());
      }
    }

    // Convert an entire list to its ABI structure.
    auto &abiListPtr = listMemory.emplace_back(std::make_unique<ABIList>(
        runtimeList.getElements().size(), abiTensors.data()));
    inputPointers.push_back(abiListPtr.get());

  } else {
    return Error(Twine("Invalid input type: ") +
                 input.getPointer()->getTypeID().getTypeName());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CompiledKernelsContext
//===----------------------------------------------------------------------===//

ErrorOrSuccess CompiledKernelsContext::load(ArrayRef<uint8_t> binary,
                                            ArrayRef<StringRef> kernelNames,
                                            size_t kernelIndex) {
  // We have two problems here:
  //  - Compiled kernels from different models may share the same names.
  //  - The same model, with the same compiled kernels, may be set up more
  //    than one.
  // We deal with these by loading into a dylib with a name derived from
  // the archive hash, and using the engine's atomic 'addIfAbsent' to
  // ensure that the dylib is created exactly once.
  std::string dylibName = llvm::utohexstr(llvm::xxh3_64bits(binary),
                                          /*LowerCase=*/true, /*Width=*/16);
  auto buf = Buffer::get(
      StringRef(reinterpret_cast<const char *>(binary.data()), binary.size()));

  // If kernels are built with debuginfo then we write them out to a shared
  // library and load it back in. Kernels are object files, so they need to be
  // linked to create a shared library. This makes the symbols visible to
  // profilers after this program has run.
  if (loadDebuginfo) {
    auto dylibOr = engine->getExecutionSession().createJITDylib(dylibName);
    if (!dylibOr)
      return Error("Could not create JIT dylib");
    llvm::orc::JITDylib &dylib = *dylibOr;

    auto objectBinaryOr =
        toModularErrorOr(llvm::object::createBinary(buf->getMemBufferRef()));
    if (objectBinaryOr.isError())
      return objectBinaryOr.takeError();
    llvm::Error err = llvm::Error::success();
    std::unique_ptr<llvm::object::Binary> objectBinary =
        objectBinaryOr.takeValue();

    std::string libName =
        std::filesystem::path(dylibName).filename().string() + "_" +
        std::filesystem::path(objectBinary->getFileName().str())
            .stem()
            .string();

    auto objectBuffer = objectBinary->getMemoryBufferRef();
    auto tmpFile = writeTempFile(libName + "-%%%%.o", objectBuffer.getBuffer());

    if (tmpFile.isError())
      return M::Error(std::string(tmpFile.takeError().get()) + ": " + libName);
    // On MacOS debuginfo is stored in object files and referenced by shared
    // libraries. Keep the object files around so the debuginfo can be found.
    tmpFile->keep();
    std::string objectFile = tmpFile->getPath();
    tmpFile->close();

    if (err)
      return toModularError(std::move(err));

    SmallString<0> outPath;
    std::error_code ec = llvm::sys::fs::createUniqueFile(
        (std::filesystem::temp_directory_path() / dylibName).string() +
            "-%%%%.so",
        outPath);
    if (ec)
      return Error("Could not create tmp file: " + ec.message());
    // Linker command is generated by cmake. The only library referenced by
    // the kernels is compiler_rt.a. We use dynamic lookup for those symbols
    // to avoid linking to it directly.
    // TODO: run linker programmatically
    std::string cmd;
    llvm::raw_string_ostream cmdS(cmd);
    cmdS << std::string(GENERICML_LINKER_CMD) << " -shared ";
    cmdS << objectFile;
    cmdS << " -o " << outPath;
    LLVM_DEBUG(llvm::dbgs() << "Linked shared library: " << cmd << "\n");
    int r = system(cmd.c_str());
    if (r != 0)
      return Error("Could not link shared object");
    LLVM_DEBUG(llvm::dbgs() << "Loading shared library " << outPath << "\n");
    auto gen = llvm::orc::DynamicLibrarySearchGenerator::Load(
        outPath.c_str(), engine->getDataLayout().getGlobalPrefix());
    if (!gen)
      return M::Error(toModularError(gen.takeError()));
    dylib.addGenerator(std::move(gen.get()));
    if (auto err = engine->addToSearchOrder(dylibName, &dylib))
      return err.takeError();
  } else {
    if (auto err = engine->addIfAbsent<KGEN::StaticArchiveLayer>(
            dylibName, std::move(buf)))
      return err.takeError();
  }

  // Load the compiled kernels into our table.
  for (StringRef kernelName : kernelNames) {
    auto symOr = engine->lookup(dylibName, kernelName);
    if (symOr.isError())
      return symOr.takeError();
    addKernel(kernelName, kernelIndex, symOr.takeValue().getFunctionPointer());
    kernelIndex++;
  }

  return success();
}

void CompiledKernelsContext::addKernel(StringRef name, size_t index,
                                       void *kernel) {
  ASSERT_STREAM(
      kernels.size() == index,
      << "All kernels must be added to symbol table at the same time.");
  kernels.push_back(kernel);
  symbolTable.push_back(name);
}
