//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Core/MojoValue.h"
#include "Support/AlignedAlloc.h"

#include <algorithm>
#include <cstdlib>

using namespace M;

MojoValue::MojoValue(size_t size, size_t align, DestructorFn destructor)
    : MojoValue(allocateBuffer(size, align), destructor, /*owning=*/true,
                /*shouldFree=*/true) {}

MojoValue::~MojoValue() {
  if (owning && data) {
    destructor(data);
    if (shouldFree)
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
  std::swap(owning, rhs.owning);
  std::swap(shouldFree, rhs.shouldFree);
}
