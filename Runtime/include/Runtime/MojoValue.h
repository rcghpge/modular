//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_MOJOVALUE_H
#define RUNTIME_MOJOVALUE_H

#include <cstddef>

namespace M {

/// `MojoValue` represents a runtime mojo object.
/// It owns the object (and release it when the MojoValue is
/// destroyed). This class is used to move mojo objects between the C++ /
/// Mojo world. We can't do much with the object in C++ (besides moving it to
/// mojo land or releasing it).
///
/// Warning: The MojoValue has a pointer to the mojo destructor function, and
/// the user must ensure that the function pointer is still valid. (for example
/// if the mojo library is dynamically loaded).
class MojoValue {
public:
  using DestructorFn = void (*)(void *);

  /// Create a new mojo object.
  /// MojoValue owns the object and call `destructor` when it get destroyed.
  /// The underlying object isn't initialized.
  /// It's the user responsibility to initialize it properly.
  /// eg: `initialize_pointee_move(mojoValue.getData(), value^)`.
  MojoValue(size_t size, size_t align, DestructorFn destructor);

  /// Create a new mojo object.
  /// `data` points to the mojo struct.
  /// MojoValue owns the object and call `destructor` when it get destroyed.
  /// `data` is also owned and released by MojoValue (it must be allocated with
  /// malloc).
  MojoValue(void *data, DestructorFn destructor)
      : data(data), destructor(destructor) {}

  ~MojoValue();

  MojoValue(MojoValue &&rhs);
  MojoValue &operator=(MojoValue &&rhs);

  /// Allocate the heap buffer needed for the mojo value.
  /// This is exposed to be able to call it from the mojo API.
  static void *allocateBuffer(size_t size, size_t align);

  /// Free the heap buffer needed for the mojo value.
  /// This is exposed to be able to call it from the mojo API.
  static void freeBuffer(void *ptr);

  /// Returns a pointer to the mojo object.
  void *getData() const { return data; }

  /// Returns the destructor to the mojo object.
  DestructorFn getDestructor() const { return destructor; }

  /// Take ownership of the mojo object.
  /// The caller is responsible for freeing the memory.
  void *take();

  /// Swap `*this` and `rhs` objects.
  /// Required for the AsyncValue API.
  void swap(MojoValue &rhs);

private:
  void *data = nullptr;
  DestructorFn destructor = nullptr;
};

} // namespace M

#endif // RUNTIME_MOJOVALUE_H
