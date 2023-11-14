//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/CompiledKernelsContext.h"
#include "GenericML/Support/BufferRef.h"
#include "GenericML/Support/List.h"
#include "GenericML/Support/Tensor.h"

using namespace M;
using namespace M::GML;

//===----------------------------------------------------------------------===//
// CompiledKernelCall
//===----------------------------------------------------------------------===//

ErrorOrSuccess CompiledKernelABI::CompiledKernelCall::pushObject(
    AnyAsyncValueRef input, char inOutType, KGEN::OutputChain &outputChain,
    StringRef kernelName, const EncodedLocation &loc) {
#if MODULAR_PARANOID
  switch (inOutType) {
  case 'i': {
    if (input.isType<GML::BufferRef>()) {
      auto &buffer = input.get<GML::BufferRef>();
      // Record that we'll be reading this buffer while the kernel is
      // in-flight.
      buffer.collectAccessUses(
          outputChain.uses,
          (Twine("kernel(line=") + Twine(loc.decode().line) + ")").str(),
          LLCL::kReadingResourceUse);

    } else if (input.isType<List>()) {
      auto &list = input.get<List>();
      for (AnyAsyncValueRef &ref : list.elements) {
        // Record that we'll be reading the buffer of this tensor while the
        // kernel is in-flight.
        if (ref.isType<GML::Tensor>()) {
          auto &tensor = ref.get<GML::Tensor>();
          tensor.getBufferRef().collectAccessUses(
              outputChain.uses,
              (Twine("kernel(line=") + Twine(loc.decode().line) + ")").str(),
              LLCL::kReadingResourceUse);
        }
      }
    }
    break;
  }
  case 'o': {
    if (input.isType<GML::BufferRef>()) {
      auto &buffer = input.get<GML::BufferRef>();
      // Record that we'll be writing this buffers while the kernel is
      // in-flight.
      buffer.collectAccessUses(
          outputChain.uses,
          (Twine("kernel(line=") + Twine(loc.decode().line) + ")").str(),
          LLCL::kWritingResourceUse);
    }
    break;
  }
  default:
    llvm::llvm_unreachable_internal("invalid in/out signature");
  }
#endif // MODULAR_PARANOID

  if (input.isType<GML::BufferRef>()) {
    const void *ptr = input.get<GML::BufferRef>().getBuffer();
    // Forget the const here since the ABI mixes input and output.
    inputPointers.push_back(const_cast<void *>(ptr));

  } else if (input.isType<GML::List>()) {
    auto &runtimeList = input.get<GML::List>();
    auto &abiTensors = tensorsMemory.emplace_back();

    // Convert individual elements to their corresponding ABI structures.
    // NOTE: This is currently not closed under all argument types and cannot
    //       be extended by subclasses.
    for (AnyAsyncValueRef &ref : runtimeList.elements) {
      if (ref.isType<GML::Tensor>()) {
        auto &runtimeTensor = ref.get<GML::Tensor>();
        ssize_t *dims =
            dimsMemory.emplace_back(runtimeTensor.getSpec().getDimsCopy())
                .data();
        void *data = const_cast<void *>(runtimeTensor.getBuffer());
        abiTensors.emplace_back(ABITensor{dims, data});
      } else {
        return Error(Twine("Invalid list element type: ") +
                     ref.getPointer()->getTypeID().getTypeName());
      }
    }

    // Convert an entire list to its ABI structure.
    auto &abiList = listMemory.emplace_back(
        ABIList{runtimeList.elements.size(), abiTensors.data()});
    inputPointers.push_back((void *)&abiList);

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
                                            ArrayRef<StringRef> kernelNames) {
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
  if (auto err = engine->addIfAbsent<KGEN::StaticArchiveLayer>(dylibName,
                                                               std::move(buf)))
    return err.takeError();

  // Load the compiled kernels into our table.
  size_t index = 0;
  for (StringRef kernelName : kernelNames) {
    auto symOr = engine->lookup(dylibName, kernelName);
    if (symOr.isError())
      return symOr.takeError();
    addKernel(kernelName, index, symOr.takeValue().getFunctionPointer());
    index++;
  }

  return success();
}

void CompiledKernelsContext::addKernel(StringRef name, size_t index,
                                       void *kernel) {
  assert(kernels.size() == index &&
         "All kernels must be added to symbol table at the same time.");
  kernels.push_back(reinterpret_cast<CompiledKernelFn *>(kernel));
  symbolTable.push_back(name);
}
