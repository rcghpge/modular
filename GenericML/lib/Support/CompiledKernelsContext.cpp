//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/CompiledKernelsContext.h"
#include "ArraySupport/Tensor.h"
#include "ArraySupport/TensorBufferRef.h"
#include "GenericML/Support/Assert.h"
#include "GenericML/Support/Definitions.h"
#include "GenericML/Support/List.h"
#include "KGEN/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "Support/FileSystemExtras.h"
#include "Support/Preprocessor.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/Object/Archive.h"

#define DEBUG_TYPE "compiled-kernels-context"

using namespace M;
using namespace M::GML;

//===----------------------------------------------------------------------===//
// CompiledKernelCall
//===----------------------------------------------------------------------===//

ErrorOrSuccess CompiledKernelABI::CompiledKernelCall::pushObject(
    AnyAsyncValueRef input, char inOutType, StringRef summary,
    const EncodedLocation &loc
#if MODULAR_PARANOID
    ,
    SmallVectorImpl<LLCL::ResourceUse> &uses
#endif
) {
#if MODULAR_PARANOID
  switch (inOutType) {
  case 'i': {
    if (input.isType<TensorBufferRef>()) {
      auto &buffer = input.get<TensorBufferRef>();
      // Record that we'll be reading this buffer while the kernel is
      // in-flight.
      buffer.collectAccessUses(
          uses, (Twine("kernel(line=") + Twine(loc.decode().line) + ")").str(),
          LLCL::kReadingResourceUse);

    } else if (input.isType<List>()) {
      auto &list = input.get<List>();
      for (Variant &v : list.getElements()) {
        const AnyAsyncValueRef &ref = v.getValue();
        // Record that we'll be reading the buffer of this tensor while the
        // kernel is in-flight.
        if (ref.isType<Tensor>()) {
          auto &tensor = ref.get<Tensor>();
          tensor.getBufferRef().collectAccessUses(
              uses,
              (Twine("kernel(line=") + Twine(loc.decode().line) + ")").str(),
              LLCL::kReadingResourceUse);
        }
      }
    }
    break;
  }
  case 'o': {
    if (input.isType<TensorBufferRef>()) {
      auto &buffer = input.get<TensorBufferRef>();
      // Record that we'll be writing this buffers while the kernel is
      // in-flight.
      buffer.collectAccessUses(
          uses, (Twine("kernel(line=") + Twine(loc.decode().line) + ")").str(),
          LLCL::kWritingResourceUse);
    }
    break;
  }
  default:
    llvm::llvm_unreachable_internal("invalid in/out signature");
  }
#endif // MODULAR_PARANOID

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
  //  - The same model, with the same complied kernels, may be setup more
  //    than one.
  // We deal with these by loading into a dylib with a name derived from
  // the archive hash, and using the engine's atomic 'addIfAbsent' to
  // ensure that the dylib is created exactly once.
  auto hash = llvm::BLAKE3::hash(binary);
  std::string dylibName = llvm::toHex(hash);
  auto buf = Buffer::get(
      StringRef(reinterpret_cast<const char *>(binary.data()), binary.size()));

  // If kernels are built with debuginfo then we write them out to a shared
  // library and load it back in. Kernels are object files, so they need to be
  // linked to create a shared library. This makes the symbols visible to
  // profilers after this program has run.
  if (loadDebuginfo) {
#ifndef GENERICML_LINKER_CMD
    return M::Error("Cannot load debug libraries if no compiler command is "
                    "given. You are on an unsupported platform.");
#else
    auto dylibOr = engine->getExecutionSession().createJITDylib(dylibName);
    if (!dylibOr)
      return Error("Could not create JIT dylib");
    llvm::orc::JITDylib &dylib = *dylibOr;
    auto archiveBinary =
        toModularErrorOr(llvm::object::Archive::create(buf->getMemBufferRef()));
    if (archiveBinary.isError())
      return archiveBinary.takeError();
    llvm::Error err = llvm::Error::success();
    llvm::SmallVector<std::string> objectFiles;
    for (auto &child : (*archiveBinary)->children(err)) {
      if (err)
        return toModularError(std::move(err));
      auto libNameOr = child.getName();
      if (!libNameOr)
        return M::Error(toString(libNameOr.takeError()));
      std::string libName =
          std::filesystem::path(dylibName).filename().string() + "_" +
          std::filesystem::path(libNameOr.get().str()).stem().string();
      auto childBufferOr = child.getMemoryBufferRef();
      if (!childBufferOr)
        return M::Error(toString(childBufferOr.takeError()));
      auto tmpFile =
          writeTempFile(libName + "-%%%%.o", childBufferOr.get().getBuffer());
      if (tmpFile.isError())
        return M::Error(std::string(tmpFile.takeError().get()) + ": " +
                        libName);
      // On MacOS debuginfo is stored in object files and referenced by shared
      // libraries. Keep the object files around so the debuginfo can be found.
      tmpFile->keep();
      std::string path = tmpFile->getPath();
      tmpFile->close();
      objectFiles.push_back(path);
    }
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
    llvm::raw_string_ostream cmd_s(cmd);
    cmd_s << std::string(GENERICML_LINKER_CMD) << " -shared ";
    llvm::interleave(objectFiles, cmd_s, " ");
    cmd_s << " -o " << outPath;
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
#endif
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
  GENERICML_ASSERT(kernels.size() == index)
      << "All kernels must be added to symbol table at the same time.";
  kernels.push_back(kernel);
  symbolTable.push_back(name);
}
