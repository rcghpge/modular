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
#include "GraphRT/BEF/BEFEncoding.h"
#include "GraphRT/BEF/BEFReader.h"
#include "LLCL/Support/RCRef.h"
#include "llvm/ADT/StringRef.h"
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
  BEFFileReader(ArrayRef<uint8_t> file, BEFFileImpl &befFile)
      : BEFReader(file,
                  [=](const char *message) { this->errorMessage = message; }),
        befFile(befFile) {}

  const char *getErrorMessage() const { return errorMessage; }

  /// Decode the key sections of the specified BEF file, capturing some of their
  /// ranges into BEFFileImpl and completely decoding others.
  bool readFileContents();

private:
  /// Decode the next section and remember it if we care about it.
  bool readNextSection();

  /// Decode the primitive names section.
  bool readPrimitivesSection(BEFReader &stringReader);
  /// Decode the function index section.
  bool readFunctionIndexSection(BEFReader &stringReader);

  /// This is filled in if an error occurs.  It always contains a static string
  /// giving a sense of what went wrong.
  const char *errorMessage = nullptr;

  ArrayRef<uint8_t> primitiveNames, functionIndex;

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

  BEFReader stringReader(befFile.strings, [=](const char *message) {
    this->errorMessage = message;
  });

  // Now that we've figured out the contents of the sections, resolve the
  // things we parse ahead of time.
  return readPrimitivesSection(stringReader) ||
         readFunctionIndexSection(stringReader);
}

/// Decode the primitive names section.
bool BEFFileReader::readPrimitivesSection(BEFReader &stringReader) {
  setContents(primitiveNames);

  // Read until we run out of names.
  while (!empty()) {
    // Each primitive name is encoded as an offset into the string table.
    size_t stringOffset;
    StringRef name;
    if (parseVBR(stringOffset) ||
        stringReader.parseNULTerminatedStringAt(stringOffset, name))
      return true;

      // TODO: Implement kernel registry + lookup.
#if 0
    auto *primitive = registry->getPrimitive(name);
    if (!primitive)
      return diagnoseUnknownPrimitive(name);
    // Otherwise remember it.
    befFile.primitives.push_back(primitive);
#endif
  }

  return false;
}

/// Decode the function index section.
bool BEFFileReader::readFunctionIndexSection(BEFReader &stringReader) {
  setContents(functionIndex);

  // Read until we run out of function descriptors.
  while (!empty()) {
    size_t functionOffset, nameOffset, locOffset, numArgs, numResults;
    StringRef name;
    if (parseVBR(functionOffset) || parseVBR(nameOffset) ||
        parseVBR(locOffset) || parseVBR(numArgs) || parseVBR(numResults) ||
        stringReader.parseNULTerminatedStringAt(nameOffset, name))
      return true;

    // TODO: Remember this function in a place we can look it up.
  }
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

BEFFileImpl::~BEFFileImpl() {}

/// Open and read a BEF file, setting up our internal state and returning a
/// pointer to our initialized object on success.  On failure, an error
/// message is emitted to the errorHandler and a null RCRef is returned.
///
ErrorOr<RCRef<BEFFile>> BEFFile::open(StringRef file, Runtime &runtime) {
  if (!BEF::hasBEFMagicNumber(file))
    return Error("not a BEF file");
  file = file.drop_front(2); // Drop the magic number.

  auto befFilePtr = RCRef<BEFFileImpl>::create(runtime);

  ArrayRef<uint8_t> fileData((const uint8_t *)file.data(), file.size());
  BEFFileReader reader(fileData, *befFilePtr);
  if (reader.readFileContents()) {
    assert(reader.getErrorMessage() && "Failed but didn't produce an error!");
    return Error::getStaticString(reader.getErrorMessage());
  }

  return befFilePtr;
}