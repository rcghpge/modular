//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Runtime/MojoValue.h"

#include "LLCL/Runtime/Runtime.h"
#include <algorithm>
#include <cstdlib>

using namespace M;

MojoValue::MojoValue(size_t size, size_t align, DestructorFn destructor)
    : MojoValue(Runtime::getCurrentAllocator()->allocateBytes(size, align),
                destructor) {}

MojoValue::~MojoValue() {
  if (data) {
    destructor(data);
    Runtime::getCurrentAllocator()->deallocateBytes(data);
  }
}

MojoValue::MojoValue(MojoValue &&rhs) { swap(rhs); }

MojoValue &MojoValue::operator=(MojoValue &&rhs) {
  swap(rhs);
  return *this;
}

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
