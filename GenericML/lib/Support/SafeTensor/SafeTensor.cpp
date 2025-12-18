//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/SafeTensor/SafeTensor.h"
#include "GenericML/DeviceDriver/Driver.h"
#include "MLRT/AsyncRT/DeviceContext/DeviceContext.h"
#include "Support/Preprocessor.h"

using namespace M;

static constexpr size_t kHeaderBinarySize = 8;
static constexpr size_t kHeaderMaxSize = 100'000'000;

#define DEBUG_TYPE "safetensor"

// f8e4m3 follows HF's `safetensor` implementation,
// this maps f8e4m3 to f8e4m3fn intentionally.
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
  X(f4e2m1fn, F4)                                                              \
  X(f8e8m0fnu, F8_E8M0)                                                        \
  X(f8e4m3fn, F8_E4M3)                                                         \
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
// SafeTensor::TensorInfo
//===----------------------------------------------------------------------===//

ErrorOr<SafeTensor::TensorInfo>
SafeTensor::TensorInfo::get(const llvm::json::Object &tensorInfo) {
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
  for (const auto &shapeValue : *shapeArray) {
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

  return SafeTensor::TensorInfo(*dtype, shape, {*dataOffsets0, *dataOffsets1});
}

size_t SafeTensor::TensorInfo::getSizeInBytes() const {
  return offsets.second - offsets.first;
}

llvm::json::Value SafeTensor::TensorInfo::toJSON() const {
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
ErrorOr<SafeTensor>
SafeTensor::get(RCRef<Driver::Device> device,
                AsyncRT::DeviceBufferRef parentBuffer,
                std::optional<llvm::sys::fs::mapped_file_region> mmap) {
  // Get the buffer size.
  size_t bufferSize = parentBuffer->byteSize();

  // Check if the buffer size is smaller than the header binary size (8
  // bytes).
  if (bufferSize < kHeaderBinarySize)
    return Error("file is too small to be a safetensor");

  // Read the header length from the beginning of the buffer (little-endian,
  // unaligned).
  uint64_t headerLength =
      llvm::support::endian::read<uint64_t, llvm::support::unaligned>(
          parentBuffer->ptr(), llvm::endianness::little);

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
  StringRef jsonString(reinterpret_cast<const char *>(parentBuffer->ptr()) +
                           kHeaderBinarySize,
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
  auto *jsonValueAsObject = jsonValue->getAsObject();
  if (!jsonValueAsObject)
    return Error("failed to convert JSON value to object");

  llvm::json::Object metadata;
  llvm::DenseMap<std::string, TensorInfo> tensorInfo;
  for (const auto &[key, value] : *jsonValueAsObject) {
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
    auto tensorInfoOr = TensorInfo::get(*value.getAsObject());
    if (tensorInfoOr.isError())
      return tensorInfoOr.takeError();
    auto [_, seen] = tensorInfo.try_emplace(key.str(), *tensorInfoOr);
    if (!seen)
      return Error(Twine("duplicate tensor name: ") + key.str());
  }

  SafeTensor result(headerLength, std::move(device), std::move(parentBuffer),
                    std::move(metadata), std::move(tensorInfo));
  if (mmap)
    result.mmap =
        std::make_shared<llvm::sys::fs::mapped_file_region>(std::move(*mmap));

  return result;
}

/// Load and parse a SafeTensor file from the given path.
ErrorOr<SafeTensor> SafeTensor::get(const std::filesystem::path &path,
                                    RCRef<Driver::Device> device) {
  // TODO: On the CPU path, can read just once.
  // On the GPU path, read the file twice: first for the metadata (tensor
  // names, etc) and second to send the data to the GPU.

  auto result = device->memmapFile(path);
  if (result)
    return result.takeError();

  return get(std::move(device), std::move(result->first),
             std::move(result->second));
}

ErrorOr<SafeTensor::TensorInfo>
SafeTensor::getTensorInfo(StringRef name) const {
  auto it = tensorInfo.find(name.str());
  if (it == tensorInfo.end()) {
    return Error(Twine("the tensor named '") + name +
                 "' was not found in the safetensor");
  }
  return it->second;
}

/// Gets a tensor from given info.
std::shared_ptr<Driver::DeviceMemory>
SafeTensor::getTensorMemory(const TensorInfo &info) const {
  auto subBuffer = parentBuffer->createSubBuffer(
      8 + headerLength + info.offsets.first, info.getSizeInBytes());
  assert(!subBuffer.isError() &&
         "sub-buffer dimensions should already have been validated");
  auto result = std::make_shared<Driver::DeviceMemory>(
      device, std::move(*subBuffer), TensorSpec{info.shape, info.dtype});

  // If the memory is memmap'ed, shared ownership of the RAII memmap object
  // with the new tensor so it can outlive this object.
  if (mmap)
    result->setDestructorFn(
        [mmap = mmap](void *, const TensorSpec &) { (void)mmap; });

  return result;
}

//===----------------------------------------------------------------------===//
// SafeTensorBundle
//===----------------------------------------------------------------------===//

ErrorOr<SafeTensorBundle>
SafeTensorBundle::get(ArrayRef<std::filesystem::path> paths,
                      RCRef<Driver::Device> ctx) {
  DenseMap<std::string, size_t> tensorNameToIndex;
  SmallVector<SafeTensor> files;
  files.reserve(files.size());

  for (const auto &[i, path] : llvm::enumerate(paths)) {
    auto safeTensorOr = SafeTensor::get(path, ctx);
    if (safeTensorOr.isError())
      return safeTensorOr.takeError();
    for (const auto &[name, tensorInfo] : safeTensorOr->getTensorInfo())
      tensorNameToIndex[name] = i;
    files.push_back(std::move(*safeTensorOr));
  }

  return SafeTensorBundle(std::move(tensorNameToIndex), std::move(files));
}

ErrorOr<std::shared_ptr<Driver::DeviceMemory>>
SafeTensorBundle::getTensorMemory(StringRef name) const {
  auto it = tensorNameToIndex.find(name.str());
  if (it == tensorNameToIndex.end())
    return Error(Twine("the tensor named '") + name +
                 "' was not found in the safetensor bundle");

  const SafeTensor &file = files[it->second];

  // Minor todo: see if we can avoid this .str()
  auto it2 = file.getTensorInfo().find(name.str());

  assert(it2 != file.getTensorInfo().end());

  return file.getTensorMemory(it2->second);
}
