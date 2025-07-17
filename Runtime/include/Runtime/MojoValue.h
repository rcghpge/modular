//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_MOJOVALUE_H
#define RUNTIME_MOJOVALUE_H

#include <cstddef>
#include <cstdint>

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

  // Tag value for MojoValues containing a PythonObject.

public:
  enum Tag : uint8_t {
    kDefault = 0,
    kPython = 1,
  };

  using DestructorFn = void (*)(void *);

  /// Create a new mojo object.
  /// MojoValue owns the object and call `destructor` when it get destroyed.
  /// The underlying object isn't initialized.
  /// It's the user responsibility to initialize it properly.
  /// eg: `initialize_pointee_move(mojoValue.getData(), value^)`.
  MojoValue(size_t size, size_t align, DestructorFn destructor,
            Tag tag = Tag::kDefault);

  /// Create a new mojo object.
  /// `data` points to the mojo struct.
  /// MojoValue owns the object and call `destructor` when it get destroyed.
  /// `data` is also owned and released by MojoValue (it must be allocated with
  /// malloc).
  MojoValue(void *data, DestructorFn destructor, Tag tag = Tag::kDefault)
      : data(data), destructor(destructor), owning(true), shouldFree(true),
        tag(tag) {}

  /// Create a reference to an existing Mojo object. The object must outlive the
  /// lifetime of this MojoValue.
  static MojoValue createRef(void *data, Tag tag = Tag::kDefault) {
    return MojoValue(data, nullptr, false, false, tag);
  }

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

  /// Take ownership of the mojo object.
  /// The caller is responsible for freeing the memory.
  void *take();

  bool isOwned() const { return owning; }

  Tag getTag() const { return tag; }

  /// Swap `*this` and `rhs` objects.
  /// Required for the AsyncValue API.
  void swap(MojoValue &rhs);

private:
  MojoValue(void *data, DestructorFn destructor, bool owning, bool shouldFree,
            Tag tag)
      : data(data), destructor(destructor), owning(owning),
        shouldFree(shouldFree), tag(tag) {}

  void *data = nullptr;
  DestructorFn destructor = nullptr;
  /// Does the mojo value own this value. Data not freed and destructor not run
  /// if not owned.
  bool owning = false;
  /// Should the mojovalue free its pointer.
  bool shouldFree = false;
  /// Type tag.
  Tag tag;
};

} // namespace M

#endif // RUNTIME_MOJOVALUE_H
