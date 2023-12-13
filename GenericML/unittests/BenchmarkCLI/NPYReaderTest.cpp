//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/BenchmarkCLI/NPYReader.h"
#include "GenericML/BenchmarkCLI/Tensor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include <filesystem>

using namespace M;
using namespace M::BenchmarkCLI;

// We do not expose any CLI tool to test this with llvm-lit instead, so dig
// around inside the source tree so we can do this from a unit test.  LLVM has
// a utility to do this, getInputFileDirectory, but its use is discouraged (and
// requires a different CMake function to be used to add the unit test
// instead), so we dig around like this instead (sorry).
static std::string getTestInputPath(StringRef filename) {
  return (std::filesystem::path(std::getenv("MODULAR_PATH")) / "GenericML" /
          "unittests" / "BenchmarkCLI" / "Inputs" / std::string_view(filename))
      .string();
}

TEST(readNPYFile, basic) {
  auto bufferOr = llvm::MemoryBuffer::getFile(getTestInputPath("basic.npy"),
                                              /*IsText=*/false,
                                              /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(bufferOr) << bufferOr.getError().message();
  auto tensorOr = readNPYFile(std::move(*bufferOr));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  Tensor tensor = tensorOr.takeValue();
  EXPECT_EQ(TensorSpec({1, 2, 3}, DType::si64), tensor.getSpec());
  auto data = static_cast<const int64_t *>(tensor.getBuffer());
  EXPECT_EQ(0, data[0]);
  EXPECT_EQ(1, data[1]);
  EXPECT_EQ(2, data[2]);
  EXPECT_EQ(3, data[3]);
  EXPECT_EQ(4, data[4]);
  EXPECT_EQ(5, data[5]);
}

TEST(hasNPYFileMagic, npy) {
  auto bufferOr = llvm::MemoryBuffer::getFile(getTestInputPath("basic.npy"),
                                              /*IsText=*/false,
                                              /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(bufferOr) << bufferOr.getError().message();
  EXPECT_TRUE(hasNPYFileMagic(**bufferOr));
}

TEST(hasNPYFileMagic, empty) {
  auto bufferOr = llvm::MemoryBuffer::getFile(getTestInputPath("empty.dat"),
                                              /*IsText=*/false,
                                              /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(bufferOr) << bufferOr.getError().message();
  EXPECT_FALSE(hasNPYFileMagic(**bufferOr));
}
