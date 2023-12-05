//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/BenchmarkCLI/NPYReader.h"
#include "GenericML/BenchmarkCLI/NPYMetadata.h"
#include "GenericML/BenchmarkCLI/Tensor.h"
#include "Support/ErrorOr.h"
#include "Support/ML/TensorSpec.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/JSON.h"

using namespace M;
using namespace M::BenchmarkCLI;

static constexpr uint8_t kSignature[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

namespace {

struct Header {
  std::string descr;
  bool fortranOrder;
  std::vector<int> shape;
};

bool fromJSON(const llvm::json::Value &value, Header &header,
              llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  return mapper && mapper.map("descr", header.descr) &&
         mapper.map("fortran_order", header.fortranOrder) &&
         mapper.map("shape", header.shape);
}

} // namespace

static DType descrToDType(StringRef descr) {
  return llvm::StringSwitch<DType>(descr)
      .Case("|i1", DType::si8)
      .Case("<i2", DType::si16)
      .Case("<i4", DType::si32)
      .Case("<i8", DType::si64)
      .Case("|u1", DType::ui8)
      .Case("<u2", DType::ui16)
      .Case("<u4", DType::ui32)
      .Case("<u8", DType::ui64)
      .Case("<f2", DType::f16)
      .Case("<f4", DType::f32)
      .Case("<f8", DType::f64)
      .Case("<c8", DType::getComplex(DType::f32))
      .Case("<c16", DType::getComplex(DType::f64))
      .Case("|b1", DType::kBool)
      .Default(DType::invalid);
}

ErrorOr<Tensor> M::BenchmarkCLI::readNPYFile(
    std::unique_ptr<llvm::MemoryBuffer> npyFileBuffer) {
  llvm::MemoryBufferByteStream stream(std::move(npyFileBuffer),
                                      llvm::endianness::little);
  llvm::BinaryStreamReader reader(stream);
  {
    ArrayRef<uint8_t> signature;
    llvm::Error error = reader.readBytes(signature, sizeof(kSignature));
    if (error)
      return toModularError(std::move(error));
    if (signature != ArrayRef(kSignature))
      return Error("NPY file signature not present");
  }
  uint8_t majorVersion, minorVersion;
  llvm::Error error = reader.readInteger(majorVersion);
  if (!error)
    error = reader.readInteger(minorVersion);
  if (error)
    return toModularError(std::move(error));
  if (majorVersion != 1)
    return Error("NPY format version " + Twine(majorVersion) +
                 " not supported");
  uint16_t headerLen;
  if ((error = reader.readInteger(headerLen)))
    return toModularError(std::move(error));
  StringRef headerText;
  if ((error = reader.readFixedString(headerText, headerLen)))
    return toModularError(std::move(error));
  auto headerJSONOr = parseNPYMetadata(headerText);
  if (headerJSONOr.isError())
    return headerJSONOr.takeError();
  Header header;
  {
    llvm::json::Path::Root rootPath("<NPY header>");
    if (!fromJSON(*headerJSONOr, header, rootPath))
      return toModularError(rootPath.getError());
  }
  if (header.fortranOrder)
    return Error("Fortran order not supported");
  DType dtype = descrToDType(header.descr);
  if (dtype == DType::invalid)
    return Error("Unable to map descr '" + Twine(header.descr) +
                 "' to Modular-supported data type");
  TensorSpec spec(header.shape, dtype);
  ArrayRef<uint8_t> tensorData;
  error = reader.readBytes(tensorData, spec.getSizeInBytes());
  if (error)
    return toModularError(std::move(error));
  return Tensor(std::move(spec), static_cast<const void *>(tensorData.data()),
                Tensor::own(std::move(npyFileBuffer)));
}
