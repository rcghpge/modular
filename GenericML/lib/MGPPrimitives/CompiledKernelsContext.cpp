//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "./CompiledKernelsContext.h"

using namespace M;
using namespace M::GML;

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
  auto buf = Cache::Buffer::get(
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
