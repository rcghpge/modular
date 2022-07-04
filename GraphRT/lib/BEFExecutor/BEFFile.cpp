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
  BEFFileReader(ArrayRef<uint8_t> file, BEFFileImpl &befFile)
      : BEFReader(file,
                  [=](Error message) { this->error = std::move(message); }),
        befFile(befFile) {}

  Optional<Error> &getError() { return error; }

  /// Decode the key sections of the specified BEF file, capturing some of their
  /// ranges into BEFFileImpl and completely decoding others.
  bool readFileContents();

private:
  /// Decode the next section and remember it if we care about it.
  bool readNextSection();

  /// Decode the invokable index section.
  bool readInvokableIndexSection(BEFReader &stringReader);

  /// This is filled in if an error occurs.
  Optional<Error> error;

  ArrayRef<uint8_t> invokableIndex;

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
  return readInvokableIndexSection(stringReader);
}

/// Decode the invokable index section.
bool BEFFileReader::readInvokableIndexSection(BEFReader &stringReader) {
  setContents(invokableIndex);

  size_t numInvokables;
  if (parseVBR(numInvokables))
    return true;

  if (numInvokables < 1)
    return emitError("BEF file must have at least one invokable");

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

  while (!empty()) {
    if (regionNumber > numInvokables)
      return reportIncorrectInvokableCount();

    size_t bodyOffset, numArgs, numResults;
    if (parseVBR(bodyOffset) || parseVBR(numArgs) || parseVBR(numResults))
      return true;

    /// Create the actual BEFInvokable.
    (void)new (&fnMemory[regionNumber++])
        BEFInvokableImpl(numArgs, numResults, bodyOffset, &befFile);
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

/// This method enumerates the primitive name table, calling the specified
/// callable on each entry.  The callable function should return a bool, and
/// a `true` result indicates that iteration should stop or `false` to
/// continue.
ErrorOrSuccess BEFFileImpl::enumeratePrimitiveNames(
    std::function<ErrorOrSuccess(StringRef)> callbackFn) {
  ErrorOrSuccess errorResult;
  BEFReader primNameReader(primitiveNames,
                           [&](Error err) { errorResult = std::move(err); });
  BEFReader stringReader(strings,
                         [&](Error err) { errorResult = std::move(err); });

  size_t numPrimitives;
  if (primNameReader.parseVBR(numPrimitives))
    return errorResult;
  primitives.reserve(numPrimitives);

  // Read until we run out of names.
  while (!primNameReader.empty()) {
    // Each primitive name is encoded as an offset into the string table.
    size_t stringOffset;
    StringRef name;
    if (primNameReader.parseVBR(stringOffset) ||
        stringReader.parseNULTerminatedStringAt(stringOffset, name))
      return errorResult;

    // Invoke the callback, propagating an error if needed.
    if (auto err = callbackFn(name))
      return err;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BEFFile implementation
//===----------------------------------------------------------------------===//

/// Return true if the specified file contents look like it is a BEF file, by
/// checking the magic number.
bool M::isBEFFile(llvm::StringRef contents) {
  return BEF::hasBEFMagicNumber(contents);
}

/// Home the vtable in a .cpp file.
BEFFile::~BEFFile() = default;

/// This method inspects the contents of the BEF file to make an educated
/// guess of what it contains, allowing for better QoI and usability of tools.
BEFContents BEFFile::sniffFileContents() const {
  BEFContents result{BEFContents::kUnknown};
  (void)((BEFFileImpl *)const_cast<BEFFile *>(this))
      ->enumeratePrimitiveNames([&](StringRef primitiveName) -> ErrorOrSuccess {
        if (primitiveName == "tflp.model") {
          result = BEFContents{BEFContents::kTFLiteModel};
          return Error("done"); // stop iteration early.
        }

        if (primitiveName == "faux.testcase") {
          result = BEFContents{BEFContents::kFauxModel};
          return Error("done"); // stop iteration early.
        }
        return success();
      });

  return result;
}

/// Open and read a BEF file, setting up our internal state and returning a
/// pointer to our initialized object on success.  On failure, an error
/// message is emitted to the errorHandler and a null RCRef is returned.
///
ErrorOr<RCRef<BEFFile>> BEFFile::open(StringRef file) {
  if (!BEF::hasBEFMagicNumber(file))
    return Error("not a BEF file");
  file = file.drop_front(2); // Drop the magic number.

  auto befFilePtr = RCRef<BEFFileImpl>::create();

  ArrayRef<uint8_t> fileData((const uint8_t *)file.data(), file.size());
  BEFFileReader reader(fileData, *befFilePtr);
  if (reader.readFileContents()) {
    assert(reader.getError() && "Failed but didn't produce an error!");
    return std::move(*reader.getError());
  }

  return befFilePtr;
}

/// Check that the primitives name table is valid and can be resolved to
/// entries in the specified registry.
ErrorOrSuccess BEFFile::installPrimitives(const PrimitiveRegistry &registry) {
  auto *befFile = static_cast<BEFFileImpl *>(this);
  return befFile->enumeratePrimitiveNames(
      [&](StringRef primitiveName) -> ErrorOrSuccess {
        auto *primitiveImpl = registry.get(primitiveName);
        if (!primitiveImpl)
          return Error("unknown primitive '" + primitiveName + "'");

        // Otherwise remember it.
        befFile->primitives.push_back(primitiveImpl);
        return success();
      });
}

/// Open and read a BEF file, setting up our internal state and returning a
/// pointer to our initialized object on success.  On an initial parse
/// failure, an error message is returned in the `ErrorOr`.  BEF file format
/// errors discovered lazily when running invokables, so they are returned
/// through the invokable's result AsyncValue's.
///
/// Note that this doesn't set up the primitive table, it requires an explicit
/// call to `installPrimitives` to do so.
///
/// TODO: This should (optionally) manage ownership of the underlying data
/// passed in.  For example, we could take a closure to run when the lifetime
/// of the BEFFile is done, or have a reference counted abstraction for
/// holding memory buffers.
ErrorOr<RCRef<BEFFile>> BEFFile::open(StringRef file,
                                      const PrimitiveRegistry &registry) {
  auto resultOr = open(file);
  if (failed(resultOr))
    return resultOr;
  auto result = resultOr.takeValue();
  if (auto err = result->installPrimitives(registry))
    return err.takeError();
  return result;
}

const BEFInvokable *BEFFile::getTopLevelInvokable() const {
  auto befFile = static_cast<const BEFFileImpl *>(this);
  assert(befFile->numInvokables > 0 && "Expected at least 1 invokable");
  // The last invokable is the module. We are guaranteed to have at least one
  // invokable, because parsing the file will fail if we don't.
  return &befFile->invokables[befFile->numInvokables - 1];
}
