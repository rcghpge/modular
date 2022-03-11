//===- BEFFile.cpp --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main entrypoint for the BEFToMLIR library.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/BEFFile.h"
#include "BEFFileImpl.h"
#include "GraphRT/BEF/BEFReader.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Support/RCRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
using namespace M;

//===----------------------------------------------------------------------===//
// BEFFile
//===----------------------------------------------------------------------===//

/// Home the vtable in a .cpp file.
BEFFile::~BEFFile() {}

//===----------------------------------------------------------------------===//
// BEFFileReader
//===----------------------------------------------------------------------===//

namespace {
/// This class decodes the key sections of the specified BEF file, capturing
/// some of their ranges into BEFFileImpl and completely decoding others.
///
/// The convention here is that these functions return false on success and
/// returns true and fill in `errorMessage` on failure.
class BEFFileReader : public BEFReader {
public:
  BEFFileReader(ArrayRef<uint8_t> file, const PrimitiveRegistry &primitives,
                BEFFileImpl &befFile)
      : BEFReader(file,
                  [=](Error message) { this->error = std::move(message); }),
        primitives(primitives), befFile(befFile) {}

  Optional<Error> &getError() { return error; }

  /// Decode the key sections of the specified BEF file, capturing some of their
  /// ranges into BEFFileImpl and completely decoding others.
  bool readFileContents();

private:
  /// Decode the next section and remember it if we care about it.
  bool readNextSection();

  /// Decode the primitive names section.
  bool readPrimitiveNamesSection(BEFReader &stringReader);
  /// Decode the function index section.
  bool readFunctionIndexSection(BEFReader &stringReader);

  /// This is filled in if an error occurs.
  Optional<Error> error;

  ArrayRef<uint8_t> primitiveNames, functionIndex;

  /// This is a reference to the (primitive name -> impl) mapping table.
  const PrimitiveRegistry &primitives;

  // This is the file structure we're reading.
  BEFFileImpl &befFile;
};
} // end anonymous namespace

/// Decode the key sections of the specified BEF file, capturing some of their
/// ranges into BEFFileImpl and completely decoding others.  Return true and
/// fill in errorMessage on failure.
bool BEFFileReader::readFileContents() {
  // Start by reading all the sections, capturing the ones we care about in
  // BEFFileInfo.
  while (!empty()) {
    if (readNextSection())
      return true;
  }

  BEFReader stringReader(befFile.strings, [=](Error message) {
    this->error = std::move(message);
  });

  // Now that we've figured out the contents of the sections, resolve the
  // things we parse ahead of time.
  return readPrimitiveNamesSection(stringReader) ||
         readFunctionIndexSection(stringReader);
}

/// Decode the primitive names section.
bool BEFFileReader::readPrimitiveNamesSection(BEFReader &stringReader) {
  setContents(primitiveNames);

  size_t numPrimitives;
  if (parseVBR(numPrimitives))
    return true;
  befFile.primitives.reserve(numPrimitives);

  // Read until we run out of names.
  while (!empty()) {
    // Each primitive name is encoded as an offset into the string table.
    size_t stringOffset;
    StringRef name;
    if (parseVBR(stringOffset) ||
        stringReader.parseNULTerminatedStringAt(stringOffset, name))
      return true;

    auto *primitiveImpl = primitives.get(name);
    if (!primitiveImpl)
      return emitError("unknown primitive '" + name + "'");

    // Otherwise remember it.
    befFile.primitives.push_back(primitiveImpl);
  }

  return false;
}

/// Decode the function index section.
bool BEFFileReader::readFunctionIndexSection(BEFReader &stringReader) {
  setContents(functionIndex);

  size_t numFunctions;
  if (parseVBR(numFunctions))
    return true;

  auto *fnMemory = (BEFFunction *)malloc(numFunctions * sizeof(BEFFunction));

  // Read until we run out of function descriptors.
  size_t fnNumber = 0;
  auto reportIncorrectFunctionCount = [&]() -> bool {
    // Destroy the partially constructed functions array.
    while (fnNumber)
      fnMemory[--fnNumber].~BEFFunction();
    free(fnMemory);
    return emitError("function count is incorrect");
  };

  while (!empty()) {
    if (fnNumber > numFunctions)
      return reportIncorrectFunctionCount();

    size_t functionOffset, nameOffset, locOffset, numArgs, numResults;
    StringRef name;
    if (parseVBR(functionOffset) || parseVBR(nameOffset) ||
        parseVBR(locOffset) || parseVBR(numArgs) || parseVBR(numResults) ||
        stringReader.parseNULTerminatedStringAt(nameOffset, name))
      return true;

    /// Remember this function in the symbol table if it isn't anonymous.
    if (!name.empty())
      befFile.functionSymbolTable[name] = fnNumber;

    /// Create the actual BEFFunction.
    new (&fnMemory[fnNumber++]) BEFFunction(
        name, numArgs, numResults, functionOffset, locOffset, &befFile);
  }

  // Check that the correct number of functions was actually present.
  if (fnNumber != numFunctions)
    return reportIncorrectFunctionCount();
  // Otherwise, we succeeded!
  befFile.functions = fnMemory;
  befFile.numFunctions = numFunctions;
  return false;
}

/// Decode the next section and remember it if we care about it.
bool BEFFileReader::readNextSection() {
  // Decode the section ID and find the section contents.
  uint8_t sectionID;
  ArrayRef<uint8_t> sectionContents;
  if (readSection(sectionID, sectionContents))
    return emitError("BEF file section header corrupted");

  // Remember the sections we care about either in BEFFileInfo (if they get
  // decoded lazily by the function executor) or in BEFFileReader (if they're
  // decoded eagerly at file load time).  Ignore sections we don't care about.
  switch (static_cast<BEF::SectionID>(sectionID)) {
  default: // Ignore unknown section.
    return false;

  case BEF::SectionID::LOCATION_FILENAMES:
    befFile.locationFilenames = sectionContents;
    return false;

  // We just remember the locations of the LocationPositions section - it is
  // directly indexed with a byte offset.
  case BEF::SectionID::LOCATION_POSITIONS:
    befFile.locationPositions = sectionContents;
    return false;

  // We just remember the locations of the Strings section - it is directly
  // indexed with a byte offset.
  case BEF::SectionID::STRINGS:
    befFile.strings = sectionContents;
    return false;

  // We just remember the locations of the Constants section - it is directly
  // indexed with a byte offset.
  case BEF::SectionID::ATTRIBUTES:
    befFile.attributes = sectionContents;
    return false;

  case BEF::SectionID::PRIMITIVE_NAMES:
    primitiveNames = sectionContents;
    return false;

  // We just remember the locations of the Functions section - it is directly
  // indexed with a byte offset from the FunctionIndex section.
  case BEF::SectionID::BEF_FUNCTIONS:
    befFile.befFunctions = sectionContents;
    return false;

  case BEF::SectionID::BEF_FUNCTION_INDEX:
    functionIndex = sectionContents;
    return false;
  }
}

//===----------------------------------------------------------------------===//
// BEFFileImpl
//===----------------------------------------------------------------------===//

BEFFileImpl::~BEFFileImpl() { free((void *)functions); }

/// Get a list of functions out of the BEF file.
std::vector<const Function *> BEFFile::getAllFunctions() const {
  auto befFile = static_cast<const BEFFileImpl *>(this);

  // All of the result functions are stored in a dense array, but we want to
  // project them as individual objects.
  std::vector<const Function *> result;
  result.reserve(befFile->numFunctions);
  for (size_t i = 0, e = befFile->numFunctions; i != e; ++i)
    result.push_back(befFile->functions + i);
  return result;
}

/// Find a Function with the specified name, returning null if it isn't found.
const Function *BEFFile::getFunction(StringRef name) const {
  auto befFile = static_cast<const BEFFileImpl *>(this);
  auto it = befFile->functionSymbolTable.find(name);
  return it != befFile->functionSymbolTable.end()
             ? &befFile->functions[it->second]
             : nullptr;
}

/// Open and read a BEF file, setting up our internal state and returning a
/// pointer to our initialized object on success.  On failure, an error
/// message is emitted to the errorHandler and a null RCRef is returned.
///
ErrorOr<RCRef<BEFFile>> BEFFile::open(StringRef file,
                                      const PrimitiveRegistry &primitives) {
  if (!BEF::hasBEFMagicNumber(file))
    return Error("not a BEF file");
  file = file.drop_front(2); // Drop the magic number.

  auto befFilePtr = RCRef<BEFFileImpl>::create();

  ArrayRef<uint8_t> fileData((const uint8_t *)file.data(), file.size());
  BEFFileReader reader(fileData, primitives, *befFilePtr);
  if (reader.readFileContents()) {
    assert(reader.getError() && "Failed but didn't produce an error!");
    return std::move(*reader.getError());
  }

  return befFilePtr;
}
