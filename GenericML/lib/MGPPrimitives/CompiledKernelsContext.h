//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPPRIMITIVES_COMPILEDKERNELSCONTEXT_H
#define GENERICML_MGPPRIMITIVES_COMPILEDKERNELSCONTEXT_H

#include "KGEN/Compiler/ExecutionEngine.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/CompilerRT/OutputChain.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"

namespace M::GML {

//
// This is the ABI for calling a compiled kernel:
//   I = number of input tensor buffers
//   O = number of output tensor buffers
//   N = number of input dimension values (for dynamic dims only)
//   M = number of output dimension values (for compiled shape functions only)
//   void* buffers[I + O], ssize_t dims[N + M], OutputChain*
//
// Both buffers and dims take inputs followed by outputs. Dimensions known
// statically are already accounted for in the compiled kernel and do not need
// to be passed in dims.
//
// We currently only support static ranks in compiled kernels, so no ranks
// are passed at runtime.
//
using CompiledKernelFn = void(void **, ssize_t *, KGEN::OutputChain *);

// Holds all the loaded compiled kernels. Will be held within a state slot
// of the mgp.context.
class CompiledKernelsContext {
public:
  CompiledKernelsContext(KGEN::ExecutionEngine *engine) : engine(engine) {}

  ~CompiledKernelsContext() = default;

  /// Load the given binary which supplies the given kernel names into the
  /// context. The kernels can be retrieved by index of their kernel name
  /// within kernelNames.
  ErrorOrSuccess load(ArrayRef<uint8_t> binary,
                      ArrayRef<StringRef> kernelNames);

  /// Returns the compiled kernel at the given index.
  CompiledKernelFn *getKernel(size_t index) const {
    assert(index <= kernels.size() && "invalid compiled kernel index");
    return kernels[index];
  }

  /// Returns the symbol name for the compiled kernel at the given index.
  StringRef getKernelName(size_t index) const {
    assert(index <= kernels.size() && "invalid compiled kernel index");
    return symbolTable[index];
  }

private:
  void addKernel(StringRef name, size_t index, void *kernel);

  /// The execution engine, retrieved from the runtime and held here as a
  /// mutable borrow for convenience.
  ///
  /// Note that the execution engine must outlive the MGP Context holding
  /// this object since LLCL tasks involving loaded code can't be guaranteed
  /// to have completed until the Runtime's WorkQueue has been shutdown.
  KGEN::ExecutionEngine *engine = nullptr;

  /// The loaded kernels.
  SmallVector<CompiledKernelFn *, 16> kernels;

  // Kernel name for each loaded kernel. For debugging and profiling only.
  SmallVector<StringRef, 16> symbolTable;
};

} // namespace M::GML

#endif // GENERICML_MGPPRIMITIVES_COMPILEDKERNELSCONTEXT_H
