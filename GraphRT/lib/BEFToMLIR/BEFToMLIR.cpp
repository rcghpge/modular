//===- BEFToMLIR.cpp - BEF to MLIR file format converter ------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main entrypoint for the BEFToMLIR library.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFToMLIR.h"
#include "GraphRT/BEF/BEFEncoding.h"
#include "GraphRT/BEFDialect/BEFOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
using namespace M;

//===----------------------------------------------------------------------===//
// BEFReader
//===----------------------------------------------------------------------===//

namespace {
/// This helper class implements low level binary parsing logic for a byte
/// range.  The general protocol for failable parsing routines is that they
/// return false and fill in their operand on success.  On failure, they emit an
/// error to the MLIR location specified at constructure and return true.
class BEFReader {
public:
  explicit BEFReader(ArrayRef<uint8_t> contents, Location loc)
      : contents(contents), loc(loc) {}

  // Return true if we've read the entire file.
  bool empty() const { return contents.empty(); }

  bool emitError(const char *message) {
    mlir::emitError(loc) << message;
    return true;
  }

  // Read a single byte from the stream.
  bool readByte(uint8_t &value) {
    if (contents.empty())
      return emitError("ran out of bytes");
    value = contents.front();
    contents = contents.drop_front();
    return false;
  }

  // Read a VBR encoded integer from the byte stream.
  bool readVBR(size_t &value) {
    uint8_t nextByte;
    if (readByte(nextByte))
      return true;

    value = nextByte & 127;
    while ((nextByte & 128) != 0) {
      if (readByte(nextByte))
        return true;

      value = (value << 7) | size_t(nextByte & 127);
    }
    return false;
  }

  // Skip over bytes until reaching the specified alignment.
  bool alignTo(unsigned alignment) {
    if (!llvm::isPowerOf2_32(alignment))
      return emitError("tried to align to non-power-of-two");

    uint8_t padding;
    while (uintptr_t(contents.data()) & (uintptr_t(alignment) - 1))
      if (readByte(padding))
        return true;

    return false;
  }

  /// Take some number of bytes from the input file and put them into result.
  bool parseBytes(size_t length, ArrayRef<uint8_t> &result) {
    if (length > contents.size())
      return emitError("missing block of data");
    result = contents.take_front(length);
    contents = contents.drop_front(length);
    return false;
  }

  /// Parse a NUL terminated string into the specified StringRef, not including
  /// the NUL terminator.
  bool parseNULTerminatedStringAt(StringRef &result, size_t position = 0) {
    if (position >= contents.size())
      return emitError("invalid location for string");

    auto *start = (const char *)contents.data() + position;
    auto nul = (const char *)memchr(start, 0, contents.size());
    if (!nul)
      return emitError("malformed NUL terminated string");
    result = StringRef(start, nul - start);
    return false;
  }

private:
  ArrayRef<uint8_t> contents;
  Location loc;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BEFToMLIRConverter
//===----------------------------------------------------------------------===//

namespace {
// This class reads a BEF file and converts it to a mlir module.
class BEFToMLIRConverter {
public:
  BEFToMLIRConverter(Location loc) : module(ModuleOp::create(loc)) {}

  ModuleOp takeModule() { return module.release(); }

  bool readHeader(BEFReader &reader);
  bool findSections(BEFReader &reader);
  bool readLocationPositions();

private:
  Location getLoc() const { return module.get()->getLoc(); }
  BEFReader getSectionReader(BEF::SectionID ID) const {
    ArrayRef<uint8_t> data;
    if (unsigned(ID) < sections.size())
      data = sections[unsigned(ID)];
    return BEFReader(data, getLoc());
  }

  mlir::OwningOpRef<ModuleOp> module;

  /// This is a list of sections indexed by the BEF section ID.
  SmallVector<ArrayRef<uint8_t>, 12> sections;

  /// This is list of filenames indexed by occurrence in the LOCATION_FILENAMES
  /// section.
  SmallVector<StringRef> locationFilenames;
  /// This is a list of locations indexed by their number in LOCATION_POSITIONS.
  SmallVector<Location> locations;
};
} // end anonymous namespace.

/// Read the magic number from the start of the BEF file and check it.
bool BEFToMLIRConverter::readHeader(BEFReader &reader) {
  // Read magic number.
  uint8_t byte1, byte2;
  if (reader.readByte(byte1) || reader.readByte(byte2))
    return true;

  if (byte1 != BEF::MAGIC1 || byte2 != BEF::MAGIC2 + BEF::VERSION0)
    return reader.emitError("missing BEF magic number");

  return false;
}

/// Parse the top level section table into the `sections` array, but do not
/// interpret the sections.
bool BEFToMLIRConverter::findSections(BEFReader &reader) {
  while (!reader.empty()) {
    uint8_t sectionID;
    size_t length;
    if (reader.readByte(sectionID) || reader.readVBR(length))
      return true;
    if (sectionID == (unsigned)BEF::SectionID::INVALID_SECTION)
      return reader.emitError("invalid section ID");

    // The low bit of the size is a boolean indicating whether there is an
    // alignment byte + padding present.
    if ((length & 1) != 0) {
      // If there is an alignment byte, read the alignment byte, and skip over
      // padding bytes until we reach the desired alignment.
      uint8_t alignment;
      if (reader.readByte(alignment) || reader.alignTo(alignment))
        return true;
    }

    // Make sure we have space in the section table for this entry and haven't
    // already seen this section.
    if (sectionID >= sections.size())
      sections.resize(sectionID + 1);
    if (sections[sectionID].data())
      return reader.emitError("duplicate section found");

    // Okay, the returned data is whatever is left now.  We shift by one to
    // eliminate the alignment bit.
    if (reader.parseBytes(length >> 1, sections[sectionID]))
      return true;
  }

  return false;
}

bool BEFToMLIRConverter::readLocationPositions() {
  BEFReader filenameReader =
      getSectionReader(BEF::SectionID::LOCATION_FILENAMES);
  BEFReader reader = getSectionReader(BEF::SectionID::LOCATION_POSITIONS);
  MLIRContext *ctx = module->getContext();

  // It is common to have many references to the same filename.  Parse and
  // unique each one a single time.
  DenseMap<size_t, StringAttr> filenames;
  while (!reader.empty()) {
    size_t filenameOffset, lineNo, colNo;
    if (reader.readVBR(filenameOffset) || reader.readVBR(lineNo) ||
        reader.readVBR(colNo))
      return true;

    auto &filenameEntry = filenames[filenameOffset];
    if (!filenameEntry) {
      StringRef filename;
      if (filenameReader.parseNULTerminatedStringAt(filename, filenameOffset))
        return true;
      filenameEntry = StringAttr::get(ctx, filename);
    }
    locations.push_back(
        mlir::FileLineColLoc::get(filenameEntry, lineNo, colNo));
  }
  return false;
}

//===----------------------------------------------------------------------===//
// translateBEFToMLIR
//===----------------------------------------------------------------------===//

/// Read the specified a BEF file from the specified binary buffer, and
/// returning the created module on success.  On failure, report an error at the
/// specified location and return null.
ModuleOp M::translateBEFToMLIR(ArrayRef<uint8_t> buffer, Location loc) {
  // We currently always get our buffer out of a SourceMgr so we know it is
  // aligned.  This allows alignTo to be efficient and simple.  If we care about
  // supporting non-aligned inputs in the future, we can either copy the input
  // buffer to realign it when necessary or make alignTo more complicated by
  // tracking the amount we're offset from natural alignment.
  assert((uintptr_t(buffer.data()) & 7) == 0 &&
         "Input buffer pointer must be 8-byte aligned for this logic to work");

  BEFToMLIRConverter converter(loc);

  BEFReader reader(buffer, loc);
  // Read/validate the header and section table.
  if (converter.readHeader(reader) || converter.findSections(reader) ||
      // Parse sections in a predictable order.
      converter.readLocationPositions())
    return {};

  return converter.takeModule();
}
