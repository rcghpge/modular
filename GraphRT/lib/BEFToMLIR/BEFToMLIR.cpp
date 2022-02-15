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
#include "mlir/IR/Builders.h"
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

  const uint8_t *data() const { return contents.data(); }

  bool emitError(const char *message) const {
    mlir::emitError(loc) << message;
    return true;
  }

  // Read a single byte from the stream.
  bool parseByte(uint8_t &value) {
    if (contents.empty())
      return emitError("ran out of bytes");
    value = contents.front();
    contents = contents.drop_front();
    return false;
  }

  // Read a VBR encoded integer from the byte stream.
  bool parseVBR(size_t &value) {
    uint8_t nextByte;
    if (parseByte(nextByte))
      return true;

    value = nextByte & 127;
    while ((nextByte & 128) != 0) {
      if (parseByte(nextByte))
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
      if (parseByte(padding))
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

  /// Ignore the first `length` bytes in this reader.
  bool dropBytes(size_t length) {
    ArrayRef<uint8_t> unused;
    return parseBytes(length, unused);
  }

  /// Parse a NUL terminated string into the specified StringRef, not including
  /// the NUL terminator.
  bool parseNULTerminatedStringAt(size_t position, StringRef &result) {
    if (position >= contents.size())
      return emitError("invalid location for string");

    auto *start = (const char *)contents.data() + position;
    auto nul = (const char *)memchr(start, 0, contents.size());
    if (!nul)
      return emitError("malformed NUL terminated string");
    result = StringRef(start, nul - start);
    return false;
  }

  /// Parse a NUL terminated string into the specified StringAttr, not including
  /// the NUL terminator.
  bool parseNULTerminatedStringAt(size_t position, StringAttr &result) {
    StringRef resultStr;
    if (parseNULTerminatedStringAt(position, resultStr))
      return true;
    result = StringAttr::get(loc.getContext(), resultStr);
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

  bool parseHeader(BEFReader &reader);
  bool findSections(BEFReader &reader);
  bool parseLocationPositions();
  bool parseAttributes();
  bool parsePrimitiveNames();
  bool parseFunctions();

private:
  MLIRContext *getContext() const { return module.get()->getContext(); }
  Location getLoc() const { return module.get()->getLoc(); }

  BEFReader getSectionReader(BEF::SectionID ID) const {
    ArrayRef<uint8_t> data;
    if (unsigned(ID) < sections.size())
      data = sections[unsigned(ID)];
    return BEFReader(data, getLoc());
  }

  bool parseAttributeAt(size_t offset, uint8_t attrKind,
                        const BEFReader &attrSection, Attribute &resultAttr);

  bool parseLocation(BEFReader &reader, Location &loc) {
    size_t locOffset;
    if (reader.parseVBR(locOffset))
      return true;
    auto it = locations.find(locOffset);
    if (it == locations.end())
      return reader.emitError("invalid location");
    loc = it->second;
    return false;
  }

  mlir::OwningOpRef<ModuleOp> module;

  /// This is a list of sections indexed by the BEF section ID.
  SmallVector<ArrayRef<uint8_t>, 12> sections;

  /// This is list of filenames indexed by occurrence in the LOCATION_FILENAMES
  /// section.
  SmallVector<StringRef> locationFilenames;

  /// This is a list of locations identified by their offset in
  /// LOCATION_POSITIONS.
  DenseMap<size_t, Location> locations;

  /// This captures the parsed attribute at an offset in the ATTRIBUTES section.
  DenseMap<size_t, Attribute> attributeAtOffset;

  /// This is the name for each primitive, indexed by primitive ID.
  SmallVector<StringAttr> primitiveNames;

  /// This is the set of parsed functions indexed by function ID.
  SmallVector<BEF::FuncOp> functions;
};
} // end anonymous namespace.

/// Read the magic number from the start of the BEF file and check it.
bool BEFToMLIRConverter::parseHeader(BEFReader &reader) {
  // Read magic number.
  uint8_t byte1, byte2;
  if (reader.parseByte(byte1) || reader.parseByte(byte2))
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
    if (reader.parseByte(sectionID) || reader.parseVBR(length))
      return true;
    if (sectionID == (unsigned)BEF::SectionID::INVALID_SECTION)
      return reader.emitError("invalid section ID");

    // The low bit of the size is a boolean indicating whether there is an
    // alignment byte + padding present.
    if ((length & 1) != 0) {
      // If there is an alignment byte, read the alignment byte, and skip over
      // padding bytes until we reach the desired alignment.
      uint8_t alignment;
      if (reader.parseByte(alignment) || reader.alignTo(alignment))
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

bool BEFToMLIRConverter::parseLocationPositions() {
  BEFReader filenameReader =
      getSectionReader(BEF::SectionID::LOCATION_FILENAMES);
  BEFReader reader = getSectionReader(BEF::SectionID::LOCATION_POSITIONS);

  const uint8_t *sectionStart = reader.data();

  // It is common to have many references to the same filename.  Parse and
  // unique each one a single time.
  DenseMap<size_t, StringAttr> filenames;
  while (!reader.empty()) {
    size_t filenameOffset, lineNo, colNo;
    size_t offset = reader.data() - sectionStart;
    if (reader.parseVBR(filenameOffset) || reader.parseVBR(lineNo) ||
        reader.parseVBR(colNo))
      return true;

    auto &filenameEntry = filenames[filenameOffset];
    if (!filenameEntry) {
      if (filenameReader.parseNULTerminatedStringAt(filenameOffset,
                                                    filenameEntry))
        return true;
    }
    locations.insert(
        {offset, mlir::FileLineColLoc::get(filenameEntry, lineNo, colNo)});
  }
  return false;
}

/// Parse an attribute at the specified
bool BEFToMLIRConverter::parseAttributeAt(size_t offset, uint8_t attrKind,
                                          const BEFReader &attrSection,
                                          Attribute &resultAttr) {
  // Copy the BEFReader (a cheap operation) so we can slice out of its body.
  BEFReader attrReader = attrSection;
  if (attrReader.dropBytes(offset))
    return true;

  unsigned size = 1;
  uint8_t byte;
  switch ((BEF::AttributeKind)attrKind) {
  case BEF::AttributeKind::I1:
    if (attrReader.parseByte(byte))
      return true;
    resultAttr = BoolAttr::get(getContext(), byte);
    return false;
  default:
    return attrSection.emitError("unknown attribute");
  case BEF::AttributeKind::I64:
    size <<= 1;
    LLVM_FALLTHROUGH;
  case BEF::AttributeKind::I32:
    size <<= 1;
    LLVM_FALLTHROUGH;
  case BEF::AttributeKind::I16:
    size <<= 1;
    LLVM_FALLTHROUGH;
  case BEF::AttributeKind::I8: {
    // Parse integers of various sizes.
    uint64_t result = 0;
    for (unsigned i = 0; i != size; ++i) {
      if (attrReader.parseByte(byte))
        return true;
      result |= uint64_t(byte) << (i * 8);
    }
    auto type = IntegerType::get(getContext(), size * 8);
    resultAttr = IntegerAttr::get(type, byte);
    return false;
    // TODO: Handle strings, arrays, and floating point.
  }
  }
}

bool BEFToMLIRConverter::parseAttributes() {
  BEFReader attrReader = getSectionReader(BEF::SectionID::ATTRIBUTES);
  BEFReader attrKindReader =
      getSectionReader(BEF::SectionID::ATTRIBUTE_KIND_MD);

  // Read until we run out of attribute kind descriptors.
  while (!attrKindReader.empty()) {
    size_t offset;
    uint8_t attrKind;
    Attribute resultAttr;
    if (attrKindReader.parseVBR(offset) || attrKindReader.parseByte(attrKind) ||
        parseAttributeAt(offset, attrKind, attrReader, resultAttr))
      return true;
    attributeAtOffset[offset] = resultAttr;
  }

  return false;
}

/// Build the "primitiveNames" array by parsing the PRIMITIVE_NAMES section,
/// resolving them all to StringAttr's.
bool BEFToMLIRConverter::parsePrimitiveNames() {
  BEFReader reader = getSectionReader(BEF::SectionID::PRIMITIVE_NAMES);
  BEFReader stringReader = getSectionReader(BEF::SectionID::STRINGS);

  // Read until we run out of names.
  while (!reader.empty()) {
    size_t stringOffset;
    StringAttr name;
    if (reader.parseVBR(stringOffset) ||
        stringReader.parseNULTerminatedStringAt(stringOffset, name))
      return true;

    primitiveNames.push_back(name);
  }

  return false;
}

bool BEFToMLIRConverter::parseFunctions() {
  BEFReader reader = getSectionReader(BEF::SectionID::BEF_FUNCTION_INDEX);
  BEFReader stringReader = getSectionReader(BEF::SectionID::STRINGS);

  SmallVector<size_t> functionOffsets;

  auto funcBuilder = OpBuilder::atBlockBegin(module->getBody());

  SmallVector<Type> inputTypes, resultTypes;
  Type unknownType = BEF::UnknownType::get(funcBuilder.getContext());

  // Read until we run out of function descriptors.
  Location loc = UnknownLoc::get(funcBuilder.getContext());
  while (!reader.empty()) {
    size_t functionOffset, nameOffset, numArgs, numResults;
    StringAttr name;
    if (reader.parseVBR(functionOffset) || reader.parseVBR(nameOffset) ||
        parseLocation(reader, loc) || reader.parseVBR(numArgs) ||
        reader.parseVBR(numResults) ||
        stringReader.parseNULTerminatedStringAt(nameOffset, name))
      return true;

    functionOffsets.push_back(functionOffset);

    // TODO: If we have type metadata, parse the types of the inputs and
    // results.
    inputTypes.assign(numArgs, unknownType);
    resultTypes.assign(numResults, unknownType);

    auto func =
        funcBuilder.create<BEF::FuncOp>(loc, name, inputTypes, resultTypes);
    functions.push_back(func);
    inputTypes.clear();
    resultTypes.clear();

    // FIXME: Handle properly.
    auto bodyBuilder = OpBuilder::atBlockBegin(func.getBodyBlock());
    bodyBuilder.create<BEF::OutputOp>(loc, ArrayRef<Value>());
  }

  // TODO: Go parse the function bodies from functionOffsets.
  return false;
}

//===----------------------------------------------------------------------===//
// translateBEFToMLIR
//===----------------------------------------------------------------------===//

/// Read the specified a BEF file from the specified binary buffer, and
/// returning the created module on success.  On failure, report an error at the
/// specified location and return null.
ModuleOp M::translateBEFToMLIR(ArrayRef<uint8_t> buffer, Location loc) {
  loc.getContext()->loadDialect<BEF::BEFDialect>();

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
  if (converter.parseHeader(reader) || converter.findSections(reader) ||
      // Parse sections in a specific order, unbound by their order in the file.
      converter.parseLocationPositions() || converter.parseAttributes() ||
      converter.parsePrimitiveNames() || converter.parseFunctions())
    return {};

  return converter.takeModule();
}
