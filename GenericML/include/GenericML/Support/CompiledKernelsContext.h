//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_COMPILEDKERNELSCONTEXT_H
#define GENERICML_SUPPORT_COMPILEDKERNELSCONTEXT_H

#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/ExecutionEngine/ExecutionEngine.h"
#include "Runtime/MojoCallContext.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"

namespace M::GML {

//===----------------------------------------------------------------------===//
// CompiledKernelCall
//===----------------------------------------------------------------------===//

// Data structures supporting the MOGG/MGP ABI
namespace CompiledKernelABI {

// ABI version of GML::List.
// NOTE the layout must match the mojo `ABI_List`
struct ABIList {
  size_t numElems;
  void *elements; // Currently always points to ABITensors

  /// Constructor
  ABIList(size_t numElems, void *elements)
      : numElems(numElems), elements(elements) {}
};

// ABI version of Tensor.
// NOTE the layout must match the mojo `ABI_Tensor`
struct ABITensor {
  ssize_t *dims; // Shape, with implied rank
  void *dataPtr; // The underlying buffer

  /// Constructor
  ABITensor(ssize_t *dims, void *dataPtr) : dims(dims), dataPtr(dataPtr) {}
};

/// Holds marshalled data for the duration of an ABI call. This instance lives
/// only until the ABI call returns, and it is the responsibility of the called
/// kernel to unmarshal this data before returning, even if the kernel is
/// asynchronous.
///
/// TODO: Remove this once we have a unified representation of values across
///       the ABI.
struct CompiledKernelCall {
  /// The accumulated pointer arguments.
  SmallVector<void *, 8> inputPointers;

  // NOTE The follow memory vectors contain object that are either
  // 1. Referenced via `.get()` by `inputPointers`.
  // 2. Referenced via `.data()` by objects in other memory vector.
  //
  // Both requires the pointer reference to remain the same after the memory
  // vector expansion; this requires the underlying memory to be allocated on
  // the heap, and moved during vector expansion. Thus we use `std::vector<T>`
  // and `std::unique_ptr<T>` instead of `SmallVector<T>` and `T`.

  /// ABI Shapes, in the form of arrays of ssize_ts.
  SmallVector<std::vector<ssize_t>> dimsMemory;
  /// ABI Tensors, which combine raw buffers with shapes.
  SmallVector<std::vector<ABITensor>> tensorsMemory;
  /// ABI Lists, which hold pointers to arbitrary marshalled objects.
  SmallVector<std::unique_ptr<ABIList>> listMemory;

  CompiledKernelCall(size_t numPointers) { inputPointers.reserve(numPointers); }

  /// Pushes the marshalled form of the given AnyAsyncValueRef onto the
  /// given pointers vector, possibly pushing additional intermediate
  /// datastructures onto the above fields. Returns error if input is
  /// not supported.
  ErrorOrSuccess pushObject(AnyAsyncValueRef input, char inOutType,
                            StringRef summary, const EncodedLocation &loc
#if MODULAR_PARANOID
                            ,
                            SmallVectorImpl<LLCL::ResourceUse> &uses
#endif
  );
};

} // namespace CompiledKernelABI

//
// This is the ABI for calling a compiled kernel:
//   I = number of 'in' arguments
//   O = number of 'out' tensor buffers
//   N = number of 'in' dimension values (for dynamic dims only)
//   M = number of 'out' dimension values (for compiled shape functions only)
//   void* args[I + O], ssize_t dims[N + M], MojoCallContext*
//
// Both args and dims take 'in' args (ie read only) followed by 'out' args
// (ie write only). Arguments representing tensors are passed as their
// underlying buffers (in args), combined with their dynamic dimensions (in
// dims). Statically known dimensions are not passed since they have already
// been accounted for in the compilation of the kernel. That is, rather than
// passing `Tensor`s as `CompiledKernelABI::Tensor` with a runtime
// representation of the entire shape, we instead passed
// `TensorBufferRefs` as their underlying buffer pointers, and a
// dimension for each dynamic dimension parameter from the original `!mo.tensor`
// type.
//
// We currently only support static ranks in compiled kernels, so no ranks
// are passed at runtime, and no dynamic rank parameters are supported.
//
// Each void* arg can be:
//  - The underlying buffer of a TensorBufferRef (for
//  `!mgp.buffer`s).
//    The lifetime is managed by the C++ runtime, possibly my moving
//    referencs into a sub-task closure.
//  - A `CompiledKernelABI::ABIList*` (for `!mgp.list`s). The lifetime is
//    managed by the `CompiledKernelCall`, and the called kernel shim is
//    responsible for unmarshalling before returning.
//
// If the called kernel is synchronous then it may signal error using
// MojoCallcontext::setToError. Otherwise, it is free to just return.
//
// If the called kernel is asynchronous then the C++ runtime will
// keep all operands and the MojoCallContext alive until the kernel's coroutine
// completes. That is signalled by the kernel eventually calling back to
// MojoCallContext::{complete, setToError}. Note that the kernel must unmarshal
// all other arguments before returning to the caller after hitting a
// suspension point.
using CompiledKernelFn = void(void **, ssize_t *, MojoCallContext *);

// A function pointer representing an alternate ABI which takes in a list of
// async value pointers as the first value and a runtime, and callContext as
// third. The async values will be unpacked and manipulated directly within the
// kernel.
using CompiledKernelAsyncABIFn = void(void **, void *, void *);
//===----------------------------------------------------------------------===//
// CompiledKernelsContext
//===----------------------------------------------------------------------===//

// Holds all the loaded compiled kernels. Will be held within a state slot
// of the mgp.context.
class CompiledKernelsContext : public ReferenceCounted<CompiledKernelsContext> {
public:
  CompiledKernelsContext(KGEN::ExecutionEngine *engine, bool loadDebuginfo)
      : engine(engine), loadDebuginfo(loadDebuginfo) {}

  ~CompiledKernelsContext() = default;

  /// Load the given binary which supplies the given kernel names into the
  /// context. The kernels can be retrieved by index of their kernel name
  /// within kernelNames.
  ErrorOrSuccess load(ArrayRef<uint8_t> binary, ArrayRef<StringRef> kernelNames,
                      size_t kernelIndex = 0);

  /// Returns the compiled kernel at the given index.
  CompiledKernelFn *getKernel(size_t kernelIndex) const {
    ASSERT_STREAM(kernelIndex < kernels.size())
        << "invalid kernel index " << kernelIndex;
    return reinterpret_cast<CompiledKernelFn *>(kernels[kernelIndex]);
  }

  CompiledKernelAsyncABIFn *getKernelAsyncABI(size_t kernelIndex) const {
    ASSERT_STREAM(kernelIndex < kernels.size())
        << "invalid kernel index " << kernelIndex;
    return reinterpret_cast<CompiledKernelAsyncABIFn *>(kernels[kernelIndex]);
  }

  /// Returns the symbol name for the compiled kernel at the given index.
  StringRef getKernelName(size_t kernelIndex) const {
    ASSERT_STREAM(kernelIndex < kernels.size())
        << "invalid kernel index " << kernelIndex;
    return symbolTable[kernelIndex];
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

  /// Should compiled kernels be loaded with debug info
  bool loadDebuginfo;

  /// The loaded kernels.
  SmallVector<void *, 16> kernels;

  // Kernel name for each loaded kernel. For debugging and profiling only.
  SmallVector<StringRef, 16> symbolTable;
};

} // namespace M::GML

#endif // GENERICML_SUPPORT_COMPILEDKERNELSCONTEXT_H
