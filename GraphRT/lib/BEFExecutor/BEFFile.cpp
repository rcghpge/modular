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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <utility>
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
/// The convention here is that these methods return false on success and
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
  /// Decode the invokable index section.
  bool readInvokableIndexSection(BEFReader &stringReader);

  /// This is filled in if an error occurs.
  Optional<Error> error;

  ArrayRef<uint8_t> invokableIndex;

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
         readInvokableIndexSection(stringReader);
}

/// Decode the primitive names section.
bool BEFFileReader::readPrimitiveNamesSection(BEFReader &stringReader) {
  setContents(befFile.primitiveNames);

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

/// Decode the invokable index section.
bool BEFFileReader::readInvokableIndexSection(BEFReader &stringReader) {
  setContents(invokableIndex);

  size_t numInvokables;
  if (parseVBR(numInvokables))
    return true;

  auto *fnMemory =
      (BEFInvokableImpl *)malloc(numInvokables * sizeof(BEFInvokableImpl));

  // Read until we run out of invokable descriptors.
  size_t regionNumber = 0;
  auto reportIncorrectInvokableCount = [&]() -> bool {
    // Destroy the partially constructed invokables array.
    while (regionNumber)
      fnMemory[--regionNumber].~BEFInvokableImpl();
    free(fnMemory);
    return emitError("invokable count is incorrect");
  };

  // Use a MapVector here to preserve input ordering.
  llvm::MapVector<std::pair<StringRef, StringRef>,
                  SmallVector<BEFInvokable *, 1>>
      invokableMap;
  while (!empty()) {
    if (regionNumber > numInvokables)
      return reportIncorrectInvokableCount();

    size_t bodyOffset, primitiveOffset, symbolOffset, numArgs, numResults;
    StringRef primitive, symbol;
    if (parseVBR(primitiveOffset) || parseVBR(symbolOffset) ||
        parseVBR(bodyOffset) || parseVBR(numArgs) || parseVBR(numResults) ||
        stringReader.parseNULTerminatedStringAt(primitiveOffset, primitive) ||
        stringReader.parseNULTerminatedStringAt(symbolOffset, symbol))
      return true;

    /// Create the actual BEFInvokable.
    BEFInvokable *inv = new (&fnMemory[regionNumber++]) BEFInvokableImpl(
        primitive, symbol, numArgs, numResults, bodyOffset, &befFile);

    /// Remember this invokable in the symbol table if it isn't anonymous by
    /// attaching it to the appropriate top level op.
    if (!symbol.empty()) {
      auto didEmplace = invokableMap.insert(
          {std::pair<StringRef, StringRef>{primitive, symbol},
           SmallVector<BEFInvokable *, 1>{inv}});
      if (!didEmplace.second)
        didEmplace.first->second.push_back(inv);
    }
  }

  // Populate the list of top level ops.
  for (auto &entry : invokableMap) {
    befFile.topLevelOps.emplace_back(entry.first.first, entry.first.second,
                                     entry.second);
  }

  // Check that the correct number of invokables was actually present.
  if (regionNumber != numInvokables)
    return reportIncorrectInvokableCount();
  // Otherwise, we succeeded!
  befFile.invokables = fnMemory;
  befFile.numInvokables = numInvokables;
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
  // decoded lazily by the invokable executor) or in BEFFileReader (if they're
  // decoded eagerly at file load time).  Ignore sections we don't care about.
  switch (static_cast<BEF::SectionID>(sectionID)) {
  default: // Ignore unknown section.
    return false;

  case BEF::SectionID::kLocationFilenames:
    befFile.locationFilenames = sectionContents;
    return false;

  // We just remember the locations of the kLocationPositions section - it is
  // directly indexed with a byte offset.
  case BEF::SectionID::kLocationPositions:
    befFile.locationPositions = sectionContents;
    return false;

  // We just remember the locations of the kStrings section - it is directly
  // indexed with a byte offset.
  case BEF::SectionID::kStrings:
    befFile.strings = sectionContents;
    return false;

  // We just remember the locations of the kAttributes section - it is directly
  // indexed with a byte offset.
  case BEF::SectionID::kAttributes:
    befFile.attributes = sectionContents;
    return false;

  case BEF::SectionID::kPrimitiveNames:
    befFile.primitiveNames = sectionContents;
    return false;

  // We just remember the locations of the kInvokablesBodies section - it is
  // directly indexed with a byte offset from the kinvokablesIndex section.
  case BEF::SectionID::kTopLevelOpBodies:
    befFile.invokableBodies = sectionContents;
    return false;

  case BEF::SectionID::kTopLevelOpIndex:
    invokableIndex = sectionContents;
    return false;
  }
}

//===----------------------------------------------------------------------===//
// BEFFileImpl
//===----------------------------------------------------------------------===//

BEFFileImpl::~BEFFileImpl() {
  free(const_cast<void *>(static_cast<const void *>(invokables)));
}

/// Get a list of invokables out of the BEF file.
ArrayRef<TopLevelEntity> BEFFile::getAllTopLevelEntities() const {
  auto befFile = static_cast<const BEFFileImpl *>(this);
  return befFile->topLevelOps;
}

/// Find an invokable with the specified name, returning null if it isn't found.
Optional<const TopLevelEntity>
BEFFile::getTopLevelEntity(StringRef primitive, StringRef symbol) const {
  auto befFile = static_cast<const BEFFileImpl *>(this);
  auto found = llvm::find_if(
      befFile->topLevelOps, [&](const TopLevelEntity &topLevelOp) {
        return topLevelOp.getPrimitiveName() == primitive &&
               topLevelOp.getSymbolName() == symbol;
      });
  if (found == befFile->topLevelOps.end())
    return {};

  return std::move(*found);
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
