//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ARRAY_SUPPORT_TENSOR_H
#define ARRAY_SUPPORT_TENSOR_H

#include "ArraySupport/TensorBufferRef.h"
#include "AsyncRT/Runtime/CompactRuntimePtr.h"
#include "Support/ML/TensorBase.h"
#include "Support/ML/TensorSpec.h"

namespace M {

//===----------------------------------------------------------------------===//
// Tensor Printing
//===----------------------------------------------------------------------===//

enum class ResultOutputStyle {
  kCompact, // Display a few elements at the start and end of each dimension
  kFull,    // Display all the elements of result
  kBinary,  // Save results to .bin
  kNone,    // Do not display any tensor details
};

/// This prints the specified tensor data to stdout.
ErrorOrSuccess printTensor(raw_ostream &os, const void *srcPtr,
                           const TensorSpec &spec, ResultOutputStyle style,
                           unsigned precision = 6);

/// Tensors in CPU memory using a dense, row-major element order.
///
/// Tensors are just a pair of a TensorBufferRef and a TensorSpec. Tensors are
/// movable and explicitly deep copyable.
///
/// CAUTION: Though movable, this is 48 bytes on 64-bit systems so best not
///          to consider it a simple handle.
class Tensor : public TensorBase<TensorBufferRef> {
public:
  using TensorBase::TensorBase;

  //===--------------------------------------------------------------------===//
  // Creation methods.
  //===--------------------------------------------------------------------===//

  /// Creates a tensor whose buffer is allocated from the allocator in
  /// runtime with size appropriate for spec.
  ///
  /// TODO(#5799): We're requiring all callers to be explicit about alignment
  /// (or lack thereof) to help track down 'forgotten alignment' problems.
  static Tensor createOwned(TensorSpec spec, LLCL::CompactRuntimePtr runtime,
                            std::optional<size_t> alignment);

  /// Creates an owned tensor whose buffer is from a taken void* pointer -- This
  /// taken pointer is allocated by caller and taken to construct the underlying
  /// bufferRef which has control over the lifetime. There is an assumption that
  /// the takeFrom pointer is allocated by caller via runtime->getAllocator().
  /// Lifetime of the allocated memory follows any Owned Tensor (via bufferRef)
  static Tensor takeOwned(TensorSpec spec, LLCL::CompactRuntimePtr runtime,
                          void *takeFrom);

  /// Creates a tensor whose buffer is at the given pointer. It is assumed the
  /// pointer has size appropriate for spec and lifetime exceeding that of
  /// the result tensor. The alignment will be used should the tensor need
  /// to be deep copied. The given lifetime should be a sound approximation of
  /// the lifetime of the buffer.
  static Tensor createBorrowed(void *buffer, TensorSpec spec,
                               std::optional<size_t> alignment,
                               TensorBufferRef::Lifetime lifetime);
  static Tensor createBorrowed(const void *buffer, TensorSpec spec,
                               std::optional<size_t> alignment,
                               TensorBufferRef::Lifetime lifetime);

  //===--------------------------------------------------------------------===//
  // Core APIs for working with a Tensor.
  //===--------------------------------------------------------------------===//

  /// Returns a pointer to the underlying buffer.
  void *getMutableBuffer() { return bufferRef.getBuffer(); }
  const void *getBuffer() const { return bufferRef.getBuffer(); }
  template <typename T>
  T *getMutableBuffer() {
    return static_cast<T *>(getMutableBuffer());
  }
  template <typename T>
  const T *getBuffer() const {
    return static_cast<const T *>(getBuffer());
  }

  /// Release the ownership of the underlying buffer to the returned reference.
  /// This should be only happen when the tensor will be immediately destroyed.
  TensorBufferRef releaseBufferRef() { return std::move(bufferRef); }

  /// Simple accessor to the buffer ref. To maintain the value semantics of this
  /// tensor it should not be modified.
  TensorBufferRef &getBufferRef() { return bufferRef; }

  /// Returns a shallow copy of this tensor, with the underlying buffer borrowed
  /// from this tensor. This tensor may owning or itself be borrowing the
  /// buffer, and the underlying buffer must outlive the returned tensor. The
  /// result will have the given borrowLifetime.
  ///
  /// Assert fails if asking to borrow from a borrow whose lifetime is
  /// not greater or equal to borrowLifetime.
  Tensor borrow(TensorBufferRef::Lifetime borrowLifetime) const;

  /// This prints the specified tensor data to stdout.
  ErrorOrSuccess print(ResultOutputStyle style, unsigned precision = 6) const;

  /// This prints the specified tensor data to stream.
  ErrorOrSuccess print(raw_ostream &os, ResultOutputStyle style,
                       unsigned precision = 6) const;

  /// This prints a debug representation of the specified tensor data to stream.
  void printDebug(raw_ostream &os) const;
};

inline raw_ostream &operator<<(raw_ostream &os, const Tensor &value) {
  (void)value.print(os, ResultOutputStyle::kFull);
  return os;
}

} // namespace M

#endif // ARRAY_SUPPORT_TENSOR_H
