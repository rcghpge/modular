//===- BEFToMLIR.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main entrypoint for the BEFToMLIR library.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFToMLIR.h"
#include "GraphRT/BEF/BEFReader.h"
#include "GraphRT/BEFDialect/BEFOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include <numeric>
using namespace M;

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
  bool parseTypes();
  bool parsePrimitiveNames();
  bool parseInvokables();
  bool parseInvokableBody(BEF::InvokableOp invokable, BEFReader &bodyReader,
                          BEFReader &attrNameReader,
                          BEFReader &mlirTypesReader);

  bool emitError(Error message) {
    module->emitError() << message;
    return true;
  }

  bool resolveLocation(size_t locOffset, Location &loc) {
    auto it = locations.find(locOffset);
    if (it == locations.end())
      return emitError("invalid location");
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
    return parseNULTerminatedStringAt(offset, stringReader, result);
  }

  bool resolveInvokable(size_t invokableIndex, Location loc,
                        BEF::InvokableOp &invokableOp) {
    if (invokableIndex >= invokables.size())
      return mlir::emitError(loc, "invalid invokable index"), true;

    invokableOp = invokables[invokableIndex];
    return false;
  }

  bool resolveType(size_t typeID, Location loc, Type &result) {
    if (typeID >= types.size())
      return mlir::emitError(loc) << "invalid type ID #" << typeID, true;
    result = types[typeID];
    return false;
  }

  BEFReader getReaderFor(ArrayRef<uint8_t> data) {
    return BEFReader(data,
                     [=](Error message) { emitError(std::move(message)); });
  }

private:
  MLIRContext *getContext() const { return module.get()->getContext(); }
  Location getLoc() const { return module.get()->getLoc(); }

  /// Parse a NUL terminated string into the specified StringAttr, not including
  /// the NUL terminator.
  bool parseNULTerminatedStringAt(size_t position, BEFReader &reader,
                                  StringAttr &result) {
    StringRef resultStr;
    if (reader.parseNULTerminatedStringAt(position, resultStr))
      return true;
    result = StringAttr::get(getContext(), resultStr);
    return false;
  }

  BEFReader getSectionReader(BEF::SectionID ID) {
    ArrayRef<uint8_t> data;
    if (unsigned(ID) < sections.size())
      data = sections[unsigned(ID)];
    return getReaderFor(data);
  }

  bool parseAttribute(size_t offset, const BEFReader &attrSection,
                      Attribute &resultAttr);

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

  /// This is the set of parsed invokables indexed by invokable ID.
  SmallVector<BEF::InvokableOp> invokables;

  /// The is the parsed version of the type list when metadata is enabled.
  std::vector<Type> types;
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
    ArrayRef<uint8_t> sectionData;
    if (reader.readSection(sectionID, sectionData))
      return true;

    // Make sure we have space in the section table for this entry and haven't
    // already seen this section.
    if (sectionID >= sections.size())
      sections.resize(sectionID + 1);
    if (sections[sectionID].data())
      return reader.emitError("duplicate section found");

    sections[sectionID] = sectionData;
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
      if (parseNULTerminatedStringAt(filenameOffset, filenameReader,
                                     filenameEntry))
        return true;
    }
    locations.insert(
        {offset, mlir::FileLineColLoc::get(filenameEntry, lineNo, colNo)});
  }
  return false;
}

/// Parse an attribute at the specified offset.
bool BEFToMLIRConverter::parseAttribute(size_t offset,
                                        const BEFReader &attrSection,
                                        Attribute &resultAttr) {
  // Copy the BEFReader (a cheap operation) so we can slice out of its body.
  BEFReader attrReader = getReaderFor(attrSection.getContents());
  if (attrReader.dropBytes(offset))
    return true;

  auto parseDictionaryAttr = [&]() -> Attribute {
    size_t length;
    if (attrSection.parseInvertedVBRAt(offset - 1, length))
      return {};

    NamedAttrList outAttrList;
    for (size_t i = 0; i < length; ++i) {
      size_t nameOffset;
      size_t attrOffset;
      Attribute name;
      Attribute value;
      if (attrReader.parseVBR(nameOffset) || attrReader.parseVBR(attrOffset) ||
          parseAttribute(offset - nameOffset, attrSection, name) ||
          parseAttribute(offset - attrOffset, attrSection, value))
        return {};

      // Construct the NamedAttribute
      outAttrList.push_back(NamedAttribute(name.cast<StringAttr>(), value));
    }
    // Produce a DictionaryAttr for the converter.
    return outAttrList.getDictionary(getContext());
  };

  // Parse a single attribute value from attrReader and return it, or null on
  // an error.
  auto parseAttribute = [&](BEF::AttributeKind attrKind) -> Attribute {
    // Handle strings here. They aren't bitwidth-encoded, so we can just deal
    // with them up front.
    if (attrKind == BEF::AttributeKind::kString) {
      size_t length;
      ArrayRef<uint8_t> result;
      if (attrSection.parseInvertedVBRAt(offset - 1, length) ||
          attrReader.parseBytes(length, result))
        return {};
      return StringAttr::get(
          getContext(), StringRef((const char *)result.data(), result.size()));
    }

    if (attrKind == BEF::AttributeKind::kDictionary)
      return parseDictionaryAttr();

    uint8_t byte;
    // I1 attrs are a single byte in the BEF format, but are also a separate
    // attribute type, and so are special-cased here.
    if (attrKind == BEF::AttributeKind::kI1) {
      if (attrReader.parseByte(byte))
        return {};
      return BoolAttr::get(getContext(), byte);
    }

    // Compute the size of the integer by taking the attribute kind. The
    // encoding of the attrKind means that we can get the bitwidth by simply
    // masking off the last bit.
    unsigned sizeInBits = ((uint8_t)attrKind & (~(uint8_t)0b1));

    // If the bitwidth is not a power of 2 and it's not less than 8, then it's
    // not a valid size. This is very unlikely to happen unless there's a logic
    // flaw elsewhere, i.e. a string that wasn't caught above.
    if (!llvm::isPowerOf2_64(sizeInBits) || sizeInBits > 64) {
      emitError(Twine("invalid bitwidth ") + Twine(sizeInBits));
      return {};
    }

    unsigned sizeInBytes = sizeInBits / 8;

    // Parse integers of various sizes.
    uint64_t result = 0;
    for (unsigned i = 0; i != sizeInBytes; ++i) {
      if (attrReader.parseByte(byte))
        return {};
      result |= uint64_t(byte) << (i * 8);
    }
    auto resultInt = APInt(sizeInBytes * 8, result);

    if (attrKind == BEF::AttributeKind::kF32)
      return FloatAttr::get(mlir::Float32Type::get(getContext()),
                            APFloat(resultInt.bitsToFloat()));

    if (attrKind == BEF::AttributeKind::kF64)
      return FloatAttr::get(mlir::Float64Type::get(getContext()),
                            APFloat(resultInt.bitsToDouble()));

    return IntegerAttr::get(IntegerType::get(getContext(), sizeInBytes * 8),
                            resultInt);
  };

  // The attribute kind always precedes the attribute value in a BEF file. In
  // the case of length-delimited types (like string and array) the layout is
  // [length, kind byte, values].
  uint8_t attrKind;
  if (attrSection.parseByteAt(offset - 1, attrKind))
    return true;

  // If this is a scalar attribute value, just parse the single attribute.
  if ((attrKind & (uint8_t)BEF::AttributeKind::kArrayType) == 0) {
    resultAttr = parseAttribute(BEF::AttributeKind(attrKind));
    return resultAttr == Attribute();
  }

  // Otherwise, if this is an array of attributes, decode each of them and then
  // assemble the array.
  size_t length;
  if (attrSection.parseInvertedVBRAt(offset - 1, length))
    return true;
  attrKind &= ~(uint8_t)BEF::AttributeKind::kArrayType;

  SmallVector<Attribute> elements;
  while (length--) {
    elements.push_back(parseAttribute(BEF::AttributeKind(attrKind)));
    if (!elements.back())
      return true;
  }

  resultAttr = ArrayAttr::get(getContext(), elements);
  return false;
}

bool BEFToMLIRConverter::parseAttributes() {
  BEFReader attrReader = getSectionReader(BEF::SectionID::ATTRIBUTES);
  BEFReader attrIndexMDReader =
      getSectionReader(BEF::SectionID::ATTRIBUTE_INDEX_MD);

  // Read until we run out of attribute offsets.
  while (!attrIndexMDReader.empty()) {
    size_t offset;
    Attribute resultAttr;
    if (attrIndexMDReader.parseVBR(offset) ||
        parseAttribute(offset, attrReader, resultAttr))
      return true;
    attributeAtOffset[offset] = resultAttr;
  }

  return false;
}

bool BEFToMLIRConverter::parseTypes() {
  BEFReader section = getSectionReader(BEF::SectionID::MLIRTYPE_STRINGS_MD);

  // Read until we run out of type strings.
  while (!section.empty()) {
    StringRef typeStr;
    if (section.parseNULTerminatedString(typeStr))
      return true;

    types.push_back(mlir::parseType(typeStr, getContext()));
    if (!types.back()) // parseType already emits an error.
      return true;
  }
  return false;
}

/// Build the "primitiveNames" array by parsing the PRIMITIVE_NAMES section,
/// resolving them all to StringAttr's.
bool BEFToMLIRConverter::parsePrimitiveNames() {
  BEFReader reader = getSectionReader(BEF::SectionID::PRIMITIVE_NAMES);
  BEFReader stringReader = getSectionReader(BEF::SectionID::STRINGS);

  // Read the count of primitive names, and verify it later.
  size_t numPrimitives;
  if (reader.parseVBR(numPrimitives))
    return true;
  primitiveNames.reserve(numPrimitives);

  // Read until we run out of names.
  while (!reader.empty()) {
    // Each primitive name is encoded as an offset into the string table.
    size_t stringOffset;
    StringAttr name;
    if (reader.parseVBR(stringOffset) ||
        parseNULTerminatedStringAt(stringOffset, stringReader, name))
      return true;
    primitiveNames.push_back(name);
  }

  // Check that numPrimitives is right.
  if (numPrimitives != primitiveNames.size())
    return emitError("expected " + Twine(numPrimitives) +
                     " primitives, but found " + Twine(primitiveNames.size()));

  return false;
}

bool BEFToMLIRConverter::parseInvokables() {
  BEFReader reader = getSectionReader(BEF::SectionID::INVOKABLES_INDEX);
  BEFReader stringReader = getSectionReader(BEF::SectionID::STRINGS);
  BEFReader attrNameReader =
      getSectionReader(BEF::SectionID::ATTRIBUTE_NAME_MD);
  BEFReader mlirTypesReader = getSectionReader(BEF::SectionID::MLIRTYPE_MD);

  size_t numInvokables;
  if (reader.parseVBR(numInvokables))
    return true;

  SmallVector<size_t> invokableOffsets;
  invokableOffsets.reserve(numInvokables);
  invokables.reserve(numInvokables);

  auto builder = OpBuilder::atBlockBegin(module->getBody());

  SmallVector<Type> inputTypes, resultTypes;
  Type unknownType = BEF::UnknownType::get(builder.getContext());

  // Read until we run out of invokable descriptors.
  Location loc = UnknownLoc::get(builder.getContext());
  while (!reader.empty()) {
    size_t bodyOffset, nameOffset, numArgs, numResults;
    StringAttr name;
    if (reader.parseVBR(bodyOffset) || reader.parseVBR(nameOffset) ||
        reader.parseVBR(numArgs) || reader.parseVBR(numResults) ||
        parseNULTerminatedStringAt(nameOffset, stringReader, name))
      return true;

    invokableOffsets.push_back(bodyOffset);
    inputTypes.assign(numArgs, unknownType);
    resultTypes.assign(numResults, unknownType);

    auto invokable =
        builder.create<BEF::InvokableOp>(loc, name, inputTypes, resultTypes);
    invokables.push_back(invokable);
    inputTypes.clear();
    resultTypes.clear();
  }

  if (invokables.size() != numInvokables)
    return emitError("expected " + Twine(numInvokables) +
                     " invokables, but found " + Twine(invokables.size()));

  // Decode each of the invokable bodies.
  BEFReader bodyReader = getSectionReader(BEF::SectionID::INVOKABLES_BODIES);
  for (auto inv : invokables) {
    if (parseInvokableBody(inv, bodyReader, attrNameReader, mlirTypesReader))
      return true;
  }

  return false;
}

namespace {
class InvokableBodyParser {
public:
  InvokableBodyParser(BEF::InvokableOp invokable, BEFReader &bodyReader,
                      BEFReader &attrNameReader, BEFReader &mlirTypesReader,
                      BEFToMLIRConverter &converter)
      : invokable(invokable), bodyReader(bodyReader),
        attrNameReader(attrNameReader), mlirTypesReader(mlirTypesReader),
        converter(converter) {}

  /// Entrypoint to parse the invokable body.
  bool parse();

private:
  bool parseRegisterTable();
  bool parsePrimitiveTable(size_t numPrimitives);
  bool parseResultRegs();
  bool parsePrimitiveList();

  BEF::InvokableOp invokable;
  BEFReader &bodyReader;

  BEFReader &attrNameReader;  // A (possibly empty) ATTRIBUTE_NAMES_MD section.
  BEFReader &mlirTypesReader; // A (possibly empty) MLIRTYPE_MD section.
  BEFToMLIRConverter &converter;

  /// This contains the number of operands that must be resolved before the
  /// primitive is executed by the BEF executor.  This is typically equal to the
  /// number of operands in the primitive, but is one for nonstrict ops.
  SmallVector<size_t, 4> primitiveTable;
  /// This contains the register numbers that are output by the invokable.
  SmallVector<size_t> resultRegs;
};
} // end anonymous namespace

/// Entrypoint to parse the invokable body.
bool InvokableBodyParser::parse() {
  size_t numPrimitives;
  return bodyReader.parseVBR(numPrimitives) || parseRegisterTable() ||
         parsePrimitiveTable(numPrimitives) || parseResultRegs() ||
         parsePrimitiveList();
}

/// Parse and ignore the "NumUses" register table.
bool InvokableBodyParser::parseRegisterTable() {
  size_t numRegisters;
  if (bodyReader.parseVBR(numRegisters))
    return true;
  while (numRegisters--) {
    size_t numUses;
    if (bodyReader.parseVBR(numUses))
      return true;
  }
  return false;
}

bool InvokableBodyParser::parsePrimitiveTable(size_t numPrimitives) {
  while (numPrimitives--) {
    size_t offset, numOperandsBeforeExecuting;
    if (bodyReader.parseVBR(offset) ||
        bodyReader.parseVBR(numOperandsBeforeExecuting))
      return true;
    primitiveTable.push_back(numOperandsBeforeExecuting);
  }

  return false;
}

bool InvokableBodyParser::parseResultRegs() {
  for (size_t i = 0, e = invokable.getResultTypes().size(); i != e; ++i) {
    size_t registerIndex;
    if (bodyReader.parseVBR(registerIndex))
      return true;
    resultRegs.push_back(registerIndex);
  }
  return false;
}

bool InvokableBodyParser::parsePrimitiveList() {
  if (bodyReader.alignTo(BEF::PRIMITIVE_ENTRY_ALIGNMENT))
    return true;

  Block &block = *invokable.getBodyBlock();
  auto bodyBuilder = OpBuilder::atBlockBegin(&block);
  Location loc = invokable.getLoc();
  Type unknownType = BEF::UnknownType::get(bodyBuilder.getContext());
  bool hasMLIRTypes = !mlirTypesReader.empty();

  // This is a table of the SSA values that make up the register table.  BEF
  // doesn't support forward references, so we can just build this top-down.
  SmallVector<Value, 16> registerValues;

  // Read the primitive records.
  for (size_t i = 0, e = primitiveTable.size(); i != e; ++i) {
    bool isOpaqueOp = false;
    uint32_t primitiveID, locOffset;
    uint16_t numOperands, numAttrs, numInvokables, numResults;
    StringAttr primitiveName;
    uint32_t resultRegisterNo = ~0U;
    if (bodyReader.parseInt4(primitiveID) || bodyReader.parseInt4(locOffset) ||
        converter.resolveLocation(locOffset, loc) ||
        converter.resolvePrimitiveName(primitiveID, loc, primitiveName) ||
        bodyReader.parseInt2(numOperands) || bodyReader.parseInt2(numAttrs) ||
        bodyReader.parseInt2(numInvokables) || bodyReader.parseInt2(numResults))
      return true;

    // We read and ignore the # users for each result.
    SmallVector<uint32_t> numResultUsers, operands, attrsOffsets,
        invokablesIndices;
    if (bodyReader.parseInt4Array(numResults, numResultUsers) ||
        bodyReader.parseInt4Array(numOperands, operands) ||
        bodyReader.parseInt4Array(numAttrs, attrsOffsets) ||
        bodyReader.parseInt4Array(numInvokables, invokablesIndices))
      return true;

    // If the primitive had results, the register number for the first one is
    // emitted.
    if (numResults) {
      if (bodyReader.parseInt4(resultRegisterNo))
        return true;
    }

    // Resolve all the SSA values for the operands.
    SmallVector<Value> operandValues;
    for (auto operand : operands) {
      if (operand >= registerValues.size())
        return bodyReader.emitError("invalid operand register #");
      operandValues.push_back(registerValues[operand]);
    }

    // The "usedBy" records are only used by the BEF executor, we skip over them
    // without interpreting them.
    auto numResultUserEntries =
        std::accumulate(numResultUsers.begin(), numResultUsers.end(), 0);
    if (bodyReader.dropBytes(numResultUserEntries * sizeof(uint32_t)))
      return true;

    // Resolve all the result types.
    SmallVector<Type> resultTypes;
    if (!hasMLIRTypes) {
      // If we don't have a types section, then just use !bef.unknown type.
      resultTypes.assign(numResults, unknownType);
    } else {
      for (size_t i = 0, e = numResults; i != e; ++i) {
        size_t typeID;
        Type type;
        if (mlirTypesReader.parseVBR(typeID) ||
            converter.resolveType(typeID, loc, type))
          return true;
        resultTypes.push_back(type);
        // If we see a !bef.unknown type, then we assume this operation lacks
        // type information, and therefore needs to be constructed as a
        // bef.opaque operation.
        isOpaqueOp |= type.isa<BEF::UnknownType>();
      }
    }

    // The first primitive is a pseudo op which only defines the argument
    // registers.
    if (i == 0 && !invokable.getArgumentTypes().empty()) {
      if (numOperands != 0 || numAttrs != 0 || numInvokables != 0 ||
          numResults != invokable.getArgumentTypes().size())
        return bodyReader.emitError("malformed argument pseudo op");

      size_t argNo = 0;
      assert(block.getNumArguments() == numResults);
      for (auto arg : block.getArguments()) {
        assert(resultRegisterNo++ == registerValues.size() &&
               "result register mismatch");
        registerValues.push_back(arg);
        arg.setType(resultTypes[argNo++]);
      }
      continue;
    }

    // Resolve attributes.
    SmallVector<Attribute> attrs;
    SmallVector<StringAttr> attrNames;
    for (unsigned attrOffset : attrsOffsets) {
      Attribute attr;
      if (converter.resolveAttribute(attrOffset, loc, attr))
        return true;
      attrs.push_back(attr);
    }

    // If we have attribute names, decode them.
    if (!attrNameReader.empty()) {
      for (size_t attr = 0, e = attrsOffsets.size(); attr != e; ++attr) {
        size_t nameOffset;
        StringAttr name;
        if (attrNameReader.parseVBR(nameOffset) ||
            converter.resolveMetadataString(nameOffset, loc, name))
          return true;
        attrNames.push_back(name);
        // Encoded bef.opaque nodes won't have attribute names.  Reconstruct
        // them as opaque ops.
        isOpaqueOp |= name.getValue().empty();
      }
    }

    // If the invokable has references to symbols or regions, then decode them
    // now.
    SmallVector<Block *> regionsBlocksToTake;
    for (uint32_t invokableIdx : invokablesIndices) {
      BEF::InvokableOp InvokableOp;
      if (converter.resolveInvokable(invokableIdx, loc, InvokableOp))
        return true;

      // If this is an anonymous invokable, then it is a region on the
      // operation, otherwise there should be an attribute name.
      bool isRegion = invokable->getName().getStringRef().empty();
      if (isRegion) {
        // Check to see if someone already used this invokable as an
        // anonymous operation region.
        if (InvokableOp.getBodyRegion().empty())
          return mlir::emitError(loc,
                                 "anonymous invokable used multiple times"),
                 true;

        Block *block = &InvokableOp.getBodyRegion().back();
        InvokableOp.getBodyRegion().getBlocks().remove(*block);
        regionsBlocksToTake.push_back(block);
      } else {
        // If this is a symbol ref, check for a name.
        if (!attrNameReader.empty()) {
          size_t nameOffset;
          StringAttr name;
          if (attrNameReader.parseVBR(nameOffset) ||
              converter.resolveMetadataString(nameOffset, loc, name))
            return true;
          attrNames.push_back(name);
        }

        // Regardless of wehther we have a name, rememer this as a
        // FlatSymbolRef.
        attrs.push_back(FlatSymbolRefAttr::get(InvokableOp));
        --numInvokables;
      }
    }

    // If this operation has more than one operand but only requires one to be
    // available before executing the primitive, then it is a nonstrict
    // primitive.
    mlir::UnitAttr nonStrict;
    if (operands.size() > 1 && primitiveTable[i] == 1)
      nonStrict = mlir::UnitAttr::get(loc.getContext());

    Operation *resultOp;
    if (!hasMLIRTypes || isOpaqueOp) {
      assert(regionsBlocksToTake.empty() &&
             "FIXME: cannot convert ops with regions to opaque yet!");
      // If we lack type information, create a bef.opaque operation.  We cannot
      // necessarily create the correct operation when we're lacking type info
      // and attribute names because we'll trip the op verifier.
      resultOp = bodyBuilder.create<BEF::OpaqueOp>(
          loc, resultTypes, primitiveName, bodyBuilder.getArrayAttr(attrs),
          nonStrict, operandValues, numInvokables);
    } else {
      OperationState state(loc, primitiveName, operandValues, resultTypes,
                           /*attrs*/ {});
      for (auto nameAndAttr : llvm::zip(attrNames, attrs))
        state.addAttribute(std::get<0>(nameAndAttr), std::get<1>(nameAndAttr));

      // Add all the regions.
      for (Block *block : regionsBlocksToTake)
        state.addRegion()->push_back(block);

      resultOp = bodyBuilder.createOperation(state);
    }

    // Add all the results to the register table so subsequent ops may use it.
    for (auto result : resultOp->getResults()) {
      assert(resultRegisterNo++ == registerValues.size() &&
             "result register mismatch");
      registerValues.push_back(result);
    }
  }

  // Make bef.output that produces the result regs for the invokable.
  SmallVector<Value> operandValues;
  for (auto resultReg : resultRegs) {
    if (resultReg >= registerValues.size())
      return bodyReader.emitError("invalid result register #");
    operandValues.push_back(registerValues[resultReg]);
  }
  auto outputOp = bodyBuilder.create<BEF::OutputOp>(loc, operandValues);

  // If we have MLIR type metadata, we can update our invokable signature to
  // have the correct types now that we know them.
  if (hasMLIRTypes) {
    auto type =
        FunctionType::get(invokable.getContext(), block.getArgumentTypes(),
                          outputOp.getOperandTypes());
    invokable->setAttr(mlir::FunctionOpInterface::getTypeAttrName(),
                       TypeAttr::get(type));
  }

  return false;
}

bool BEFToMLIRConverter::parseInvokableBody(BEF::InvokableOp invokable,
                                            BEFReader &bodyReader,
                                            BEFReader &attrNameReader,
                                            BEFReader &mlirTypesReader) {
  return InvokableBodyParser(invokable, bodyReader, attrNameReader,
                             mlirTypesReader, *this)
      .parse();
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
  BEFReader reader = converter.getReaderFor(buffer);
  // Read/validate the header and section table.
  if (converter.parseHeader(reader) || converter.findSections(reader) ||
      // Parse sections in a specific order, unbound by their order in the file.
      converter.parseLocationPositions() || converter.parseAttributes() ||
      converter.parseTypes() || converter.parsePrimitiveNames() ||
      converter.parseInvokables())
    return {};

  return converter.takeModule();
}
