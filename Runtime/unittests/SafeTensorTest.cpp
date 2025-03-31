//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Runtime/IO/SafeTensor.h"
#include "Support/Buffer.h"
#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

using namespace M;

template <size_t N>
BufferRef getDeserializedBuffer(const char (&data)[N]) {
  return Buffer::get(StringRef(data, N));
}

static std::string toString(const llvm::json::Value &value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  value.print(os);
  return str;
}

static std::string toString(const SafeTensor::TensorInfo &info) {
  return toString(info.toJSON());
}

TEST(SafeTensor, CanReadMetadata) {
  BufferRef buffer = getDeserializedBuffer(
      "\x59\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":["
      "2,2],\"data_offsets\":[0,16]},\"__metadata__\":{\"foo\":\"bar\"}}"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");

  auto safeTensorOr = SafeTensor::get(buffer);
  ASSERT_FALSE(safeTensorOr.isError()) << safeTensorOr.getError();

  ASSERT_EQ(toString(safeTensorOr->getMetadata()), "{\"foo\":\"bar\"}");
}

TEST(SafeTensor, CanReadTensorInfo) {
  BufferRef buffer = getDeserializedBuffer(
      "<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,"
      "2],\"data_offsets\":[0,16]}}"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");

  auto safeTensorOr = SafeTensor::get(buffer);
  ASSERT_FALSE(safeTensorOr.isError()) << safeTensorOr.getError();

  // llvm::outs() << "metadata = " << safeTensor.getMetadata() << "\n";
  auto tensorInfo = safeTensorOr->getTensorInfo("test");
  ASSERT_FALSE(tensorInfo.isError()) << tensorInfo.getError();
  ASSERT_EQ(toString(std::move(*tensorInfo)),
            "{\"data_offsets\":[0,16],\"dtype\":\"I32\",\"shape\":[2,2]}");

  ASSERT_EQ(tensorInfo->dtype, DType::si32);
  ASSERT_EQ(tensorInfo->shape, SmallVector<size_t>({2, 2}));
  ASSERT_EQ(tensorInfo->offsets.first, 0);
  ASSERT_EQ(tensorInfo->offsets.second, 16);

  ASSERT_EQ(tensorInfo->getSizeInBytes(), 16);
}

TEST(SafeTensor, HeaderTooSmall) {
  BufferRef buffer = getDeserializedBuffer("");

  auto safeTensorOr = SafeTensor::get(buffer);
  ASSERT_TRUE(safeTensorOr.isError()) << safeTensorOr.getError();
  ASSERT_STREQ(safeTensorOr.getError(), "file is too small to be a safetensor");
}

TEST(SafeTensor, HeaderTooLarge) {
  BufferRef buffer = getDeserializedBuffer(
      "<\x00\x00\x00\x00\xff\xff\xff{\"test\":{\"dtype\":\"I32\",\"shape\":[2,"
      "2],\"data_offsets\":[0,16]}}"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");

  auto safeTensorOr = SafeTensor::get(buffer);
  ASSERT_TRUE(safeTensorOr.isError()) << safeTensorOr.getError();
  ASSERT_STREQ(safeTensorOr.getError(),
               "header length is greater than the max header size");
}
