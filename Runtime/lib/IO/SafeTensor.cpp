//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Runtime/IO/SafeTensor.h"
#include "Support/Preprocessor.h"

using namespace M;

static constexpr size_t kHeaderBinarySize = 8;
static constexpr size_t kHeaderMaxSize = 100'000'000;

#define DEBUG_TYPE "safetensor"

#define TYPE_MAPPING(X)                                                        \
  X(kBool, BOOL)                                                               \
  X(ui8, U8)                                                                   \
  X(ui16, U16)                                                                 \
  X(ui32, U32)                                                                 \
  X(ui64, U64)                                                                 \
  X(si8, I8)                                                                   \
  X(si16, I16)                                                                 \
  X(si32, I32)                                                                 \
  X(si64, I64)                                                                 \
  X(f8e4m3, F8_E4M3)                                                           \
  X(f8e5m2, F8_E5M2)                                                           \
  X(f16, F16)                                                                  \
  X(f32, F32)                                                                  \
  X(f64, F64)                                                                  \
  X(bf16, BF16)

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static ErrorOr<DType> getDType(std::optional<StringRef> nameOr) {
  if (!nameOr)
    return Error("unknown dtype");

  StringRef name = *nameOr;

#define DISPATCH_DTYPE(DTYPE, NAME)                                            \
  if (name == STRINGIFY(NAME))                                                 \
    return DType::DTYPE;

  TYPE_MAPPING(DISPATCH_DTYPE)
#undef DISPATCH_DTYPE

  return Error(Twine("unknown dtype: ") + name);
}

static ErrorOr<std::string> fromDType(DType dtype) {
  switch (dtype.getValue()) {
  default:
    return Error(Twine("cannot convert dtype: ") + dtype.getAsString());

#define DISPATCH_DTYPE(DTYPE, NAME)                                            \
  case DType::DTYPE:                                                           \
    return STRINGIFY(NAME);

    TYPE_MAPPING(DISPATCH_DTYPE)
#undef DISPATCH_DTYPE
  }

  llvm_unreachable("unhandled dtype");
  return "";
}

//===----------------------------------------------------------------------===//
// SafeTensorInfo
//===----------------------------------------------------------------------===//

ErrorOr<SafeTensorInfo>
SafeTensorInfo::get(const llvm::json::Object &tensorInfo) {
  // Get the dtype.
  auto dtypeData = tensorInfo.find("dtype");
  if (dtypeData == tensorInfo.end())
    return Error("dtype is not present in the tensor info");
  ErrorOr<DType> dtype = getDType(dtypeData->second.getAsString());
  if (dtype.isError())
    return dtype.takeError();

  // Get the shape.
  auto shapeData = tensorInfo.find("shape");
  if (shapeData == tensorInfo.end())
    return Error("shape is not present in the tensor info");
  auto shapeArray = shapeData->second.getAsArray();
  if (!shapeArray)
    return Error("shape is not an array");

  SmallVector<size_t> shape;
  shape.reserve(shapeArray->size());
  for (auto &shapeValue : *shapeArray) {
    std::optional<int64_t> shapeValueAsInteger = shapeValue.getAsInteger();
    assert(shapeValueAsInteger && "shape value is not an integer");
    shape.push_back(*shapeValueAsInteger);
  }

  // Get the offsets.
  auto dataOffsets = tensorInfo.find("data_offsets");
  if (dataOffsets == tensorInfo.end())
    return Error("data_offsets is not present in the tensor info");
  auto dataOffsetsArrayOr = dataOffsets->second.getAsArray();
  if (!dataOffsetsArrayOr)
    return Error("data_offsets is not an array");

  llvm::json::Array dataOffsetsArray = *dataOffsetsArrayOr;
  if (dataOffsetsArray.size() != 2)
    return Error("data_offsets must have exactly two elements");

  std::optional<int64_t> dataOffsets0 = dataOffsetsArray[0].getAsInteger();
  std::optional<int64_t> dataOffsets1 = dataOffsetsArray[1].getAsInteger();
  assert(dataOffsets0 && "data_offsets[0] is not an integer");
  assert(dataOffsets1 && "data_offsets[1] is not an integer");

  return SafeTensorInfo(*dtype, shape, {*dataOffsets0, *dataOffsets1});
}

size_t SafeTensorInfo::getSizeInBytes() const {
  return offsets.second - offsets.first;
}

llvm::json::Value SafeTensorInfo::toJSON() const {
  llvm::json::Object json;
  json["dtype"] = *fromDType(dtype);
  json["shape"] = llvm::json::Array(shape);
  json["data_offsets"] = llvm::json::Array({offsets.first, offsets.second});
  return json;
}

//===----------------------------------------------------------------------===//
// SafeTensor
//===----------------------------------------------------------------------===//

/// Load and parse a SafeTensor file from the given data.
ErrorOr<SafeTensor> SafeTensor::get(BufferRef buffer) {
  // Get the buffer size.
  size_t bufferSize = buffer->getBufferSize();

  // Check if the buffer size is smaller than the header binary size (8
  // bytes).
  if (bufferSize < kHeaderBinarySize)
    return Error("file is too small to be a safetensor");

  // Read the header length from the beginning of the buffer (little-endian,
  // unaligned).
  uint64_t headerLength =
      llvm::support::endian::read<uint64_t, llvm::endianness::little,
                                  llvm::support::unaligned>(
          buffer->getBufferStart());

  LLVM_DEBUG(llvm::outs() << "headerLength = " << headerLength << "\n");

  // Check if the header length exceeds the maximum allowed header size.
  if (headerLength > kHeaderMaxSize)
    return Error("header length is greater than the max header size");

  // Check if the header length plus the header binary size exceeds the total
  // buffer size.
  if (headerLength + kHeaderBinarySize > bufferSize)
    return Error("header length is greater than the file size");

  // Extract the JSON string from the buffer, starting after the header binary
  // size, with the length of headerLength.
  StringRef jsonString(buffer->getBufferStart() + kHeaderBinarySize,
                       headerLength);

  // If the buffer does not start with `{` then it's not a valid json.
  if (!jsonString.starts_with("{"))
    return Error("the header is not a valid json object");

  LLVM_DEBUG(llvm::outs() << "headerLength = " << headerLength << "\n");
  LLVM_DEBUG(llvm::outs() << "jsonString = " << jsonString << "\n");

  // Parse the JSON string into a JSON value.
  llvm::Expected<llvm::json::Value> jsonValue = llvm::json::parse(jsonString);
  if (!jsonValue)
    return Error("failed to deserialize safetensor");

  // Get the JSON value as a JSON object.
  llvm::json::Object *jsonValueAsObject = jsonValue->getAsObject();
  if (!jsonValueAsObject)
    return Error("failed to convert JSON value to object");

  llvm::json::Object metadata;
  llvm::DenseMap<std::string, llvm::json::Object> tensorInfo;
  for (auto &[key, value] : *jsonValueAsObject) {
    // If the key is "__metadata__", store the value as metadata.
    if (key == "__metadata__") {
      if (!metadata.empty())
        return Error("duplicate metadata while deserializing safetensor");
      assert(value.getAsObject() && "metadata is not a JSON object");
      metadata = *value.getAsObject();
      continue;
    }

    // Otherwise, treat it as tensor info and try to emplace it into
    // tensorInfo map. Check for duplicate tensor names.
    auto [_, seen] = tensorInfo.try_emplace(key.str(), *value.getAsObject());
    if (!seen)
      return Error(Twine("duplicate tensor name: ") + key.str());
  }

  // Construct and return a SafeTensor object with the parsed metadata and
  // tensor info.
  return SafeTensor(buffer, std::move(metadata), std::move(tensorInfo));
}

/// Load and parse a SafeTensor file from the given path.
ErrorOr<SafeTensor> SafeTensor::get(const std::filesystem::path &path) {
  // Check if the file exists at the given path.
  std::error_code ec;
  if (!std::filesystem::exists(path, ec))
    return Error::fromErrorCode(ec, "file does not exist");

  // Load the file into a buffer.
  ErrorOr<BufferRef> buffer = Buffer::getFile(path);
  if (buffer.isError())
    return buffer.takeError();

  return get(*buffer);
}

/// Retrieve metadata for a specific tensor by name.
ErrorOr<SafeTensorInfo> SafeTensor::getTensorInfo(StringRef name) const {
  auto it = tensorInfo.find(name.str());
  if (it == tensorInfo.end()) {
    return Error(Twine("the tensor named '") + name +
                 "' was not found in the safetensor");
  }
  return SafeTensorInfo::get(it->second);
}
