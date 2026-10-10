//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Support/BumpAllocator.h"

#include "gtest/gtest.h"
#include <cstdint>
#include <utility>

namespace {

struct alignas(32) OverAligned {
  int value;
};

TEST(BumpAllocator, allocationsHonorAlignment) {
  M::BumpAllocator allocator;

  auto *integer = allocator.allocate<int>();
  auto *overAligned = allocator.allocate<OverAligned>();

  EXPECT_EQ(reinterpret_cast<uintptr_t>(integer) % alignof(int), 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(overAligned) % alignof(OverAligned),
            0u);
  EXPECT_EQ(allocator.getNumSlabs(), 1u);
}

TEST(BumpAllocator, createsObjectsAndCopiesArrays) {
  M::BumpAllocator allocator;

  auto *created = allocator.create<std::pair<int, int>>(3, 5);
  const int source[] = {2, 4, 6};
  llvm::ArrayRef<int> copy = allocator.copyArray<int>(source);

  EXPECT_EQ(created->first, 3);
  EXPECT_EQ(created->second, 5);
  EXPECT_EQ(copy, llvm::ArrayRef(source));
  EXPECT_NE(copy.data(), source);
}

TEST(BumpAllocator, largeAllocationsUseDedicatedSlabs) {
  M::BumpAllocator allocator;

  void *small = allocator.allocate(16, alignof(std::max_align_t));
  void *large =
      allocator.allocate(M::BumpAllocator::sizeThreshold + 1, alignof(char));

  EXPECT_EQ(allocator.getNumSlabs(), 2u);
  EXPECT_GE(allocator.getTotalMemory(),
            M::BumpAllocator::slabSize + M::BumpAllocator::sizeThreshold + 1);
  EXPECT_GE(allocator.identifyKnownObject(small), 0);
  EXPECT_LT(allocator.identifyKnownObject(large), 0);
}

TEST(BumpAllocator, resetRetainsOnlyFirstRegularSlab) {
  M::BumpAllocator allocator;

  allocator.allocate(16, alignof(std::max_align_t));
  allocator.allocate(M::BumpAllocator::sizeThreshold + 1, alignof(char));
  allocator.reset();

  EXPECT_EQ(allocator.getNumSlabs(), 1u);
  EXPECT_EQ(allocator.getTotalMemory(), M::BumpAllocator::slabSize);
  EXPECT_NE(allocator.allocate(0, alignof(char)), nullptr);
}

TEST(BumpAllocator, moveTransfersOwnership) {
  M::BumpAllocator source;
  void *allocation = source.allocate(16, alignof(std::max_align_t));
  int64_t identifier = source.identifyKnownObject(allocation);

  M::BumpAllocator destination(std::move(source));

  EXPECT_EQ(source.getNumSlabs(), 0u);
  EXPECT_EQ(destination.identifyKnownObject(allocation), identifier);
}

TEST(BumpAllocator, checkpointRollsBackWithinCurrentSlab) {
  M::BumpAllocator allocator;
  allocator.allocate(16, alignof(std::max_align_t));

  void *discarded;
  {
    auto checkpoint = allocator.checkpoint();
    discarded = allocator.allocate(32, alignof(std::max_align_t));
  }

  EXPECT_EQ(allocator.allocate(32, alignof(std::max_align_t)), discarded);
}

TEST(BumpAllocator, checkpointRetainsAndReusesLaterSlabs) {
  M::BumpAllocator allocator;
  allocator.allocate(3000, alignof(std::max_align_t));

  {
    auto checkpoint = allocator.checkpoint();
    allocator.allocate(2000, alignof(std::max_align_t));
    EXPECT_EQ(allocator.getNumSlabs(), 2u);
  }

  EXPECT_EQ(allocator.getNumSlabs(), 2u);
  allocator.allocate(2000, alignof(std::max_align_t));
  EXPECT_EQ(allocator.getNumSlabs(), 2u);
}

TEST(BumpAllocator, checkpointRetainsAndReusesDedicatedSlabs) {
  M::BumpAllocator allocator;
  void *discarded;

  {
    auto checkpoint = allocator.checkpoint();
    discarded =
        allocator.allocate(M::BumpAllocator::sizeThreshold + 1, alignof(char));
  }

  EXPECT_EQ(allocator.getNumSlabs(), 1u);
  EXPECT_EQ(
      allocator.allocate(M::BumpAllocator::sizeThreshold + 1, alignof(char)),
      discarded);
  EXPECT_EQ(allocator.getNumSlabs(), 1u);
}

} // namespace
