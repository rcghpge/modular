//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Runtime/MojoValue.h"

#include "AsyncRT/Runtime/Runtime.h"
#include <algorithm>
#include <cstdlib>

using namespace M;

MojoValue::MojoValue(size_t size, size_t align, DestructorFn destructor)
    : MojoValue(allocateBuffer(size, align), destructor) {}

MojoValue::~MojoValue() {
  if (data) {
    destructor(data);
    freeBuffer(data);
  }
}

MojoValue::MojoValue(MojoValue &&rhs) { swap(rhs); }

MojoValue &MojoValue::operator=(MojoValue &&rhs) {
  swap(rhs);
  return *this;
}

void *MojoValue::allocateBuffer(size_t size, size_t align) {
  return alignedAlloc(align, size);
}

void MojoValue::freeBuffer(void *ptr) { alignedFree(ptr); }

void MojoValue::swap(MojoValue &rhs) {
  std::swap(data, rhs.data);
  std::swap(destructor, rhs.destructor);
}

void *MojoValue::take() {
  void *ptr = data;
  data = nullptr;
  destructor = nullptr;
  return ptr;
}
