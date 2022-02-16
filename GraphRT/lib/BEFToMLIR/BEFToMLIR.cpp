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
#include <numeric>
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

  /// Parse a single byte from the stream.
  bool parseByte(uint8_t &value) {
    if (contents.empty())
      return emitError("ran out of bytes");
    value = contents.front();
    contents = contents.drop_front();
    return false;
  }

  /// Parse two bytes in little endian format from the stream.
  bool parseInt2(uint16_t &value) {
    uint8_t a, b;
    if (parseByte(a) || parseByte(b))
      return true;
    value = uint16_t(a) | (uint16_t(b) << 8);
    return false;
  }

  /// Parse four bytes in little endian format from the stream.
  bool parseInt4(uint32_t &value) {
    uint16_t a, b;
    if (parseInt2(a) || parseInt2(b))
      return true;
    value = uint32_t(a) | (uint32_t(b) << 16);
    return false;
  }

  /// Parse N int4 values as an array.
  bool parseInt4Array(size_t numElements, SmallVectorImpl<uint32_t> &result) {
    while (numElements--) {
      result.push_back(0);
      if (parseInt4(result.back()))
        return true;
    }
    return false;
  }

  /// Parse a VBR encoded integer from the byte stream.
  bool parseVBR(size_t &result) {
    uint8_t nextByte;
    if (parseByte(nextByte))
      return true;

    result = nextByte & 127;
    while ((nextByte & 128) != 0) {
      if (parseByte(nextByte))
        return true;

      result = (result << 7) | size_t(nextByte & 127);
    }
    return false;
  }

  /// Parse an inverted VBR that ends at the specified offset.
  bool parseInvertedVBRAt(size_t offset, size_t &result) const {
    result = 0;
    uint8_t lastByte;
    do {
      if (offset == 0)
        return emitError("invalid inverted VBR");
      lastByte = contents[offset - 1];
      result = (result << 7) | (lastByte & 0x7F);
      --offset;
    } while (lastByte & 0x80);
    return false;
  }

  /// Skip over bytes until reaching the specified alignment.
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
  bool parseNULTerminatedStringAt(size_t position, StringRef &result) const {
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
  bool parseNULTerminatedStringAt(size_t position, StringAttr &result) const {
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
  bool parseFunctionBody(BEF::FuncOp func, size_t offset,
                         BEFReader &attrNameReader);

  bool resolveLocation(size_t locOffset, Location &loc) {
    auto it = locations.find(locOffset);
    if (it == locations.end())
      return mlir::emitError(module->getLoc()) << "invalid location", true;
    loc = it->second;
    return false;
  }

  bool resolvePrimitiveName(size_t primIndex, Location loc,
                            StringAttr &result) {
    if (primIndex >= primitiveNames.size())
      return mlir::emitError(loc) << "invalid primitive name", true;
    result = primitiveNames[primIndex];
    return false;
  }

  bool resolveAttribute(size_t attrOffset, Location loc, Attribute &result) {
    auto it = attributeAtOffset.find(attrOffset);
    if (it == attributeAtOffset.end())
      return mlir::emitError(loc) << "invalid attribute offset", true;
    result = it->second;
    return false;
  }

  bool resolveMetadataString(size_t offset, Location loc, StringAttr &result) {
    BEFReader stringReader = getSectionReader(BEF::SectionID::STRINGS_MD);
    return stringReader.parseNULTerminatedStringAt(offset, result);
  }

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

  // Parse a single attribute value from attrReader and return it, or null on
  // an error.
  auto parseAttribute = [&]() -> Attribute {
    unsigned size = 1;
    uint8_t byte;
    switch ((BEF::AttributeKind)attrKind) {
    default:
      return attrSection.emitError("unknown attribute"), Attribute();
    case BEF::AttributeKind::I1:
      if (attrReader.parseByte(byte))
        return {};
      return BoolAttr::get(getContext(), byte);
    case BEF::AttributeKind::I64:
    case BEF::AttributeKind::F64:
      size <<= 1;
      LLVM_FALLTHROUGH;
    case BEF::AttributeKind::I32:
    case BEF::AttributeKind::F32:
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
          return {};
        result |= uint64_t(byte) << (i * 8);
      }
      auto resultInt = APInt(size * 8, result);

      if (attrKind == (unsigned)BEF::AttributeKind::F32)
        return FloatAttr::get(mlir::Float32Type::get(getContext()),
                              APFloat(resultInt.bitsToFloat()));

      if (attrKind == (unsigned)BEF::AttributeKind::F64)
        return FloatAttr::get(mlir::Float64Type::get(getContext()),
                              APFloat(resultInt.bitsToDouble()));

      return IntegerAttr::get(IntegerType::get(getContext(), size * 8),
                              resultInt);
    }
    case BEF::AttributeKind::STRING: {
      size_t length;
      ArrayRef<uint8_t> result;
      if (attrSection.parseInvertedVBRAt(offset, length) ||
          attrReader.parseBytes(length, result))
        return {};
      return StringAttr::get(
          getContext(), StringRef((const char *)result.data(), result.size()));
    }
    }
  };

  // If this is a scalar attribute value, just parse the single attribute.
  if ((attrKind & (uint8_t)BEF::AttributeKind::ARRAY_TYPE) == 0) {
    resultAttr = parseAttribute();
    return resultAttr == Attribute();
  }

  // Otherwise, if this is an array of attributes, decode each of them and then
  // assemble the array.
  size_t length;
  if (attrSection.parseInvertedVBRAt(offset, length))
    return true;
  attrKind &= ~(uint8_t)BEF::AttributeKind::ARRAY_TYPE;

  SmallVector<Attribute> elements;
  while (length--) {
    elements.push_back(parseAttribute());
    if (!elements.back())
      return true;
  }

  resultAttr = ArrayAttr::get(getContext(), elements);
  return false;
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
  BEFReader attrNameReader =
      getSectionReader(BEF::SectionID::ATTRIBUTE_NAME_MD);

  SmallVector<size_t> functionOffsets;

  auto funcBuilder = OpBuilder::atBlockBegin(module->getBody());

  SmallVector<Type> inputTypes, resultTypes;
  Type unknownType = BEF::UnknownType::get(funcBuilder.getContext());

  // Read until we run out of function descriptors.
  Location loc = UnknownLoc::get(funcBuilder.getContext());
  while (!reader.empty()) {
    size_t functionOffset, nameOffset, locOffset, numArgs, numResults;
    StringAttr name;
    if (reader.parseVBR(functionOffset) || reader.parseVBR(nameOffset) ||
        reader.parseVBR(locOffset) || resolveLocation(locOffset, loc) ||
        reader.parseVBR(numArgs) || reader.parseVBR(numResults) ||
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
  }

  // Decode each of the function bodies.
  for (auto fnAndOffset : llvm::zip(functions, functionOffsets)) {
    if (parseFunctionBody(std::get<0>(fnAndOffset), std::get<1>(fnAndOffset),
                          attrNameReader))
      return true;
  }

  return false;
}

namespace {
class FunctionParser {
public:
  FunctionParser(BEF::FuncOp func, BEFReader &funcReader,
                 BEFReader &attrNameReader, BEFToMLIRConverter &converter)
      : func(func), funcReader(funcReader), attrNameReader(attrNameReader),
        converter(converter) {}

  /// Entrypoint to parse the function body.
  bool parse();

private:
  bool parseRegisterTable();
  bool parsePrimitiveTable();
  bool parseResultRegs();
  bool parsePrimitiveList();

  BEF::FuncOp func;
  BEFReader &funcReader;

  /// This is non-empty if the file has an attribute names section.
  BEFReader &attrNameReader;
  BEFToMLIRConverter &converter;

  /// This contains the number of operands that must be resolved before the
  /// primitive is executed by the BEF executor.  This is typically equal to the
  /// number of operands in the primitive, but is one for nonstrict ops.
  SmallVector<size_t, 4> primitiveTable;
  /// This contains the register numbers that are output by the BEF function.
  SmallVector<size_t> resultRegs;
};
} // end anonymous namespace

/// Entrypoint to parse the function body.
bool FunctionParser::parse() {
  return parseRegisterTable() || parsePrimitiveTable() || parseResultRegs() ||
         parsePrimitiveList();
}

/// Parse and ignore the "NumUses" register table.
bool FunctionParser::parseRegisterTable() {
  size_t numRegisters;
  if (funcReader.parseVBR(numRegisters))
    return true;
  while (numRegisters--) {
    size_t numUses;
    if (funcReader.parseVBR(numUses))
      return true;
  }
  return false;
}

bool FunctionParser::parsePrimitiveTable() {
  size_t numPrimitives;
  if (funcReader.parseVBR(numPrimitives))
    return true;

  while (numPrimitives--) {
    size_t offset, numOperandsBeforeExecuting;
    if (funcReader.parseVBR(offset) ||
        funcReader.parseVBR(numOperandsBeforeExecuting))
      return true;
    primitiveTable.push_back(numOperandsBeforeExecuting);
  }

  return false;
}

bool FunctionParser::parseResultRegs() {
  for (size_t i = 0, e = func.getResultTypes().size(); i != e; ++i) {
    size_t registerIndex;
    if (funcReader.parseVBR(registerIndex))
      return true;
    resultRegs.push_back(registerIndex);
  }
  return false;
}

bool FunctionParser::parsePrimitiveList() {
  if (funcReader.alignTo(BEF::PRIMITIVE_ENTRY_ALIGNMENT))
    return true;

  Block &block = *func.getBodyBlock();
  auto bodyBuilder = OpBuilder::atBlockBegin(&block);
  Location loc = func.getLoc();
  Type unknownType = BEF::UnknownType::get(bodyBuilder.getContext());

  // This is a table of the SSA values that make up the register table.  BEF
  // doesn't support forward references, so we can just build this top-down.
  SmallVector<Value, 16> registerValues;

  // Read the primitive records.
  for (size_t i = 0, e = primitiveTable.size(); i != e; ++i) {
    uint32_t primitiveID, locOffset;
    uint16_t numOperands, numAttrs, numFunctions, numResults;
    StringAttr primitiveName;
    if (funcReader.parseInt4(primitiveID) || funcReader.parseInt4(locOffset) ||
        converter.resolveLocation(locOffset, loc) ||
        converter.resolvePrimitiveName(primitiveID, loc, primitiveName) ||
        funcReader.parseInt2(numOperands) || funcReader.parseInt2(numAttrs) ||
        funcReader.parseInt2(numFunctions) || funcReader.parseInt2(numResults))
      return true;

    // We read and ignore the # users for each result.
    SmallVector<uint32_t> numResultUsers, operands, attrsOffsets, functions,
        resultNos;
    if (funcReader.parseInt4Array(numResults, numResultUsers) ||
        funcReader.parseInt4Array(numOperands, operands) ||
        funcReader.parseInt4Array(numAttrs, attrsOffsets) ||
        funcReader.parseInt4Array(numResults, resultNos))
      return true;

    // Resolve all the SSA values for the operands.
    SmallVector<Value> operandValues;
    for (auto operand : operands) {
      if (operand >= registerValues.size())
        return funcReader.emitError("invalid operand register #");
      operandValues.push_back(registerValues[operand]);
    }

    // The "usedBy" records are only used by the BEF executor, we skip over them
    // without interpreting them.
    auto numResultUserEntries =
        std::accumulate(numResultUsers.begin(), numResultUsers.end(), 0);
    if (funcReader.dropBytes(numResultUserEntries * sizeof(uint32_t)))
      return true;

    // The first primitive is a pseudo op which only defines the argument
    // registers.
    if (i == 0 && !func.getArgumentTypes().empty()) {
      if (numOperands != 0 || numAttrs != 0 || numFunctions != 0 ||
          numResults != func.getArgumentTypes().size())
        return funcReader.emitError("malformed argument pseudo op");

      size_t elt = 0;
      assert(block.getNumArguments() == numResults);
      for (auto arg : block.getArguments()) {
        assert(resultNos[elt++] == registerValues.size() &&
               "result register mismatch");
        registerValues.push_back(arg);
      }
      continue;
    }

    // Resolve attributes.
    SmallVector<Attribute> attrs, attrNames;
    for (unsigned attrOffset : attrsOffsets) {
      Attribute attr;
      if (converter.resolveAttribute(attrOffset, loc, attr))
        return true;
      attrs.push_back(attr);
    }

    if (!attrNameReader.empty()) {
      for (unsigned attrOffset : attrsOffsets) {
        (void)attrOffset;
        size_t nameOffset;
        StringAttr name;
        if (attrNameReader.parseVBR(nameOffset) ||
            converter.resolveMetadataString(nameOffset, loc, name))
          return true;
        attrNames.push_back(name);
      }
    }

    // Otherwise we need to create a primitive for this.
    if (numFunctions != 0) {
      // TODO: read all the names from attrNameReader!
      return mlir::emitError(loc, "FIXME: Don't support decoding regions yet!"),
             true;
    }

    // If this operation has more than one operand but only requires one to be
    // available before executing the primitive, then it is a nonstrict
    // primitive.
    mlir::UnitAttr nonStrict;
    if (operands.size() > 1 && primitiveTable[i] == 1)
      nonStrict = mlir::UnitAttr::get(loc.getContext());

    // Since we don't have type information, we create a bef.opaque operation.
    SmallVector<Type> resultTypes(numResults, unknownType);
    auto resultOp = bodyBuilder.create<BEF::OpaqueOp>(
        loc, resultTypes, primitiveName, bodyBuilder.getArrayAttr(attrs),
        nonStrict, operandValues, numFunctions);

    // TODO: Switch to a first class representation instead of OpaqueOp.
    if (!attrNames.empty())
      resultOp->setAttr("bef.attr.names", bodyBuilder.getArrayAttr(attrNames));

    // Add all the results to the register table so subsequent ops may use it.
    size_t elt = 0;
    for (auto result : resultOp.getResults()) {
      assert(resultNos[elt++] == registerValues.size() &&
             "result register mismatch");
      registerValues.push_back(result);
    }
  }

  // Make bef.output that produces the result regs for the function.
  SmallVector<Value> operandValues;
  for (auto resultReg : resultRegs) {
    if (resultReg >= registerValues.size())
      return funcReader.emitError("invalid result register #");
    operandValues.push_back(registerValues[resultReg]);
  }
  bodyBuilder.create<BEF::OutputOp>(loc, operandValues);
  return false;
}

bool BEFToMLIRConverter::parseFunctionBody(BEF::FuncOp func, size_t offset,
                                           BEFReader &attrNameReader) {
  BEFReader funcReader = getSectionReader(BEF::SectionID::BEF_FUNCTIONS);

  return funcReader.dropBytes(offset) ||
         FunctionParser(func, funcReader, attrNameReader, *this).parse();
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
