//===- MLIRToBEF.cpp - MLIR to BEF file format converter ------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main entrypoints for the MLIRToBEF library.
// The converter is implemented in three phases.  The first phase identifies all
// of the strings and attributes that need to be emitted to the string/attribute
// pool.  The second phase optimizes and emits the strings and attributes to
// the file and remembers their offsets.  The third phase emits all of the
// regions in the MLIR program.
//
// MLIR ops are converted to primitive info and stored in BEF.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/MLIRToBEF.h"
#include "GraphRT/BEF/BEFEncoding.h"
#include "GraphRT/BEFDialect/BEFOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringSet.h"
using namespace M;

/// This holds an offset into a BEF section.
struct BEFOffset {
  size_t value;
};

static FunctionType getRegionFunctionType(Region *region) {
  // Emit information about the type of the function.
  auto *block = &region->getBlocks().front();

  // Arguments.
  SmallVector<Type, 4> inputs(block->getArgumentTypes());

  // Results: MLIR Regions don't have an easy way to identify results in
  // regions, so we just hard code the "bef.output" instruction.
  assert(isa<BEF::OutputOp>(&block->back()));
  SmallVector<Type, 4> results(block->back().getOperandTypes());
  return FunctionType::get(region->getContext(), inputs, results);
}

//===----------------------------------------------------------------------===//
// Attribute Processing
//===----------------------------------------------------------------------===//

namespace {
enum class WellKnownAttrs {
  Unknown,   // This isn't a well-known attribute.
  NonStrict, // This is the bef.nonstrict attribute on a primitive.
};
} // end anonymous namespace

/// Classify this attribute, so the rest of the code can know if it gets
/// special treatment.
static WellKnownAttrs classifyAttribute(StringRef attrName) {
  return llvm::StringSwitch<WellKnownAttrs>(attrName)
      .Case("bef.nonstrict", WellKnownAttrs::NonStrict)
      .Default(WellKnownAttrs::Unknown);
}

// Return true if this is a supported attribute that can be emitted as a
// attribute reference in a primitive.
static bool isSupportedAttribute(Attribute attr) {
  // We support references to BEF functions.
  if (attr.isa<SymbolRefAttr>())
    return true;

  // FIXME: Generalize
  return false;
}

//===----------------------------------------------------------------------===//
// BEFEmitter
//===----------------------------------------------------------------------===//

namespace {

// This is the emitter that builds a BEF into an std::vector.  This class
// contains low-level routines used by the various specific emitters.  In
// addition to collecting the bytes contained in this piece of the BEF file,
// this tracks the alignment requirement of the contents.  If this is a
// subsection of the file, then the enclosing container is required to provide
// at least this alignment.
class BEFEmitter {
public:
  BEFEmitter() {}
  BEFEmitter(const BEFEmitter &) = delete;
  BEFEmitter &operator=(const BEFEmitter &) = delete;

  // Return the alignment required by this chunk of a BEF file.
  unsigned getRequiredAlignment() const { return requiredAlignment; }

  /// Return the current position as a BEFOffset.
  BEFOffset getOffset() const { return BEFOffset{result.size()}; }

  /// Return a raw pointer into the result buffer for the specified offset.
  uint8_t *getRawPointer(BEFOffset offset) {
    assert(offset.value <= result.size() && "invalid offset");
    return result.data() + offset.value;
  }

  void emitByte(uint8_t byte) { result.push_back(byte); }

  void emitBytes(ArrayRef<uint8_t> bytes) {
    result.insert(result.end(), bytes.begin(), bytes.end());
  }

  void emitStringData(StringRef str) {
    emitBytes({reinterpret_cast<const uint8_t *>(str.data()), str.size()});
  }

  void emitNulTerminatedString(StringRef str) {
    emitStringData(str);
    emitByte(0);
  }

  void alignTo(unsigned alignment) {
    // Alignment of 0 and 1 is a noop.
    if (alignment < 2)
      return;

    assert(llvm::isPowerOf2_32(alignment));

    // We need attributes to have proper alignment in the file, so figure out
    // whether we need padding before this to make sure it ends up at the right
    // address.
    size_t curOffset = result.size();
    size_t neededPadding = llvm::alignTo(curOffset, alignment) - curOffset;

    // Emit dummy padding bytes to get up to the right offset.
    while (neededPadding--)
      emitByte(BEF::ALIGNMENT_BYTE);

    // Keep track of the maximum required alignment.
    requiredAlignment = std::max(requiredAlignment, alignment);
  }

  // Emit a guaranteed 4-byte integer aligned to 4 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void EmitInt4(uint32_t value) {
    alignTo(4);
    uint8_t data[] = {uint8_t(value & 0xFF), uint8_t((value >> 8) & 0xFF),
                      uint8_t((value >> 16) & 0xFF),
                      uint8_t((value >> 24) & 0xFF)};
    emitBytes(data);
  }

  // Emit a guaranteed 8-byte integer aligned to 8 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void emitInt8(uint64_t value) {
    alignTo(8);
    uint8_t data[] = {
        uint8_t(value & 0xFF),         uint8_t((value >> 8) & 0xFF),
        uint8_t((value >> 16) & 0xFF), uint8_t((value >> 24) & 0xFF),
        uint8_t((value >> 32) & 0xFF), uint8_t((value >> 40) & 0xFF),
        uint8_t((value >> 48) & 0xFF), uint8_t((value >> 56) & 0xFF)};
    emitBytes(data);
  }

  // Emit a vbr encoded integer of arbitrary width.
  void emitVBR(size_t value) { emitVBRImpl(value, false); }
  void emitInvertedVBR(size_t value);

  /// Emit an offset within a known BEF section.
  void emitOffset(BEFOffset offset) { emitVBR(offset.value); }

  // Many parts of the emitter logic includes forward references into stuff
  // that hasn't been emitted and has variable size.  This is handled by making
  // nested emitters.  This helper function emits the subpieces once they are
  // constructed, ensuring that alignment requirements of the nested emitter
  // are maintained correctly.
  void emitEmitter(const BEFEmitter &emitter) {
    alignTo(emitter.getRequiredAlignment());
    emitBytes(emitter.result);
  }

  void emitSection(BEF::SectionID sectionID, ArrayRef<uint8_t> sectionData,
                   unsigned alignment = 1) {
    // Section start with an identifier.
    result.push_back(static_cast<uint8_t>(sectionID));

    // Emit a length.  The low bit indicates whether alignment padding is
    // present.  We fill it in later.
    size_t needsAlignmentPaddingOffset = result.size();
    emitVBR(sectionData.size() << 1);

    // If the section body requires alignment padding, then we need to emit
    // the alignment and the padding bytes.
    if (alignment > 1) {
      if (result.size() & (alignment - 1)) {
        emitByte(alignment);
        alignTo(alignment);
        result[needsAlignmentPaddingOffset] |= 1;
      } else {
        // Otherwise, if we happen to be at a compatible offset, we just
        // remember that we need this alignment.
        requiredAlignment = std::max(requiredAlignment, alignment);
      }
    }

    // Emit the payload data itself.
    emitBytes(sectionData);
  }

  void emitSection(BEF::SectionID sectionID, const BEFEmitter &emitter) {
    emitSection(sectionID, emitter.result, emitter.getRequiredAlignment());
  }

  std::vector<uint8_t> takeResult() { return std::move(result); }

private:
  void emitVBRImpl(size_t value, bool isHighPart);
  // Keep track of the alignment required for the start of this object.
  unsigned requiredAlignment = 1;
  std::vector<uint8_t> result;
};

} // namespace

/// Our fundamental unit is a bytestream, but we want to be able to emit large
/// values as well.  We use a VBR encoding, where the high bit set indicates
/// that this is only a portion of the value.
void BEFEmitter::emitVBRImpl(size_t value, bool isHighPart) {
  if ((value >> 7) != 0)
    emitVBRImpl(value >> 7, /*isHighPart=*/true);

  emitByte(static_cast<uint8_t>((value & 127) | (isHighPart ? 128 : 0)));
}

/// Emit the specified value as a "backwards" VBR, which allows clients to
/// decode/parse it from the end of the value back to the front.  This is used
/// for things like string/array lengths, where we want a pointer to the string
/// data to indicate both the data and the length, allowing the executor to
/// reconstruct a StringRef/ArrayRef.
void BEFEmitter::emitInvertedVBR(size_t value) {
  emitByte(uint8_t(value & 0x7F));
  value >>= 7;
  while (value != 0) {
    emitByte(uint8_t((value & 0x7F) | 0x80));
    value >>= 7;
  }
}

//===----------------------------------------------------------------------===//
// EntityTable
//===----------------------------------------------------------------------===//

namespace {

/// This table keeps track of the interesting entities (attributes, types, other
/// strings) that we care about.  This is built in the first pass.
class EntityTable {
public:
  /// Walk the IR to build the symbol tables, this returns true on failure.
  bool collect(ModuleOp module, bool disableOptionalSections);
  ssize_t getFunctionNamed(StringAttr name) const;

  unsigned getTypeIndex(Type type) const {
    auto it = typeIDs.find(type);
    assert(it != typeIDs.end() && "unregistered type");
    return it->second;
  }

  unsigned getFunctionID(const Region &region) const {
    auto it = regionFunctionIDs.find(&region);
    assert(it != regionFunctionIDs.end() && "region not added to entity table");
    return it->second;
  }

  unsigned getPrimitiveID(Operation *primitive) const {
    auto it = primitiveIDs.find(primitive->getName().getIdentifier());
    assert(it != primitiveIDs.end() && "Unknown primitive");
    return it->second;
  }

private:
  void addString(StringRef string) { strings.insert(string); }
  void addType(Type type);
  LogicalResult addBEFFunction(Region *region, StringAttr name);
  void addPrimitive(Operation *primitive);
  void addLocation(Location loc);
  void addAttributeType(Attribute attr);

public:
  /// Uniquing set of attributes we need to emit, kept in order so we always
  /// produce a determinstic output file.  This only maintains the attributes
  /// that are references by the MLIR operations, it doesn't hold the
  /// recursive elements.
  llvm::SetVector<Attribute> attributes;
  SmallVector<Location> attributeLocs;

  /// After the attributes section is emitted, this holds the offset for
  /// each attribute that is emitted.
  DenseMap<Attribute, BEFOffset> attributeOffsets;

  /// Uniquing set of the primitives that we need to emit.
  std::vector<StringRef> primitives;
  DenseMap<StringAttr, unsigned> primitiveIDs;

  /// BEFFunctionEntry keeps track of each of the top level `func`'s and the
  /// regions attached to control flow statements.  These all turn into BEF
  /// functions.
  struct BEFFunctionEntry {
    BEFFunctionEntry(StringAttr name, FunctionType type, Region *region)
        : name(name), type(type), region(region) {}

    StringAttr name;
    FunctionType type;
    Region *region;
  };

  // List of functions that we need to emit, along with a name if they came
  // from a top level function.
  std::vector<BEFFunctionEntry> functions;
  DenseMap<Region *, unsigned> regionFunctionIDs;
  DenseMap<StringAttr, unsigned> namedFunctionIDs;

  // Types we've seen so far.
  std::vector<Type> types;
  DenseMap<Type, unsigned> typeIDs;

  // All of the strings we need to emit the BEF file, an unordered
  // collection that we sort before emitting.
  llvm::StringSet<> strings;

  /// This is the offset of each known string in the STRINGS section.
  DenseMap<StringRef, BEFOffset> stringOffsets;

  // This is all of the filenames referred to by locations in the file.
  llvm::SetVector<StringAttr> locationFilenames;

  // This is a list of locations within the file, the first element of the
  // tuple is a index into locationFilenames, the second and third are line/col
  // information.
  using LocationTuple = std::tuple<StringAttr, unsigned, unsigned>;
  llvm::SetVector<LocationTuple> locations;

  /// This is the offset of a given location record in the LOCATION_POSITIONS
  /// Section of the BEF file.
  DenseMap<LocationTuple, BEFOffset> locationOffsets;
};

} // namespace

// Add a type to our table, checking it by pointer to reduce string
// conversions.
void EntityTable::addType(Type type) {
  // Ignore the type if we've seen it before.
  if (!typeIDs.insert({type, types.size()}).second)
    return;
  types.push_back(type);

#if 0 // TODO: Remove this.
  // If it is new, remember the type name as a string.
  SmallVector<char, 64> resultStr;
  llvm::raw_svector_ostream os(resultStr);
  type.print(os);
  addString(os.str());
#endif
}

LogicalResult EntityTable::addBEFFunction(Region *region, StringAttr name) {
  // Check to see if we support this region kind.
  if (region->getBlocks().size() != 1) {
    mlir::emitError(region->getLoc())
        << "multi-block regions cannot be emitted to BEF files";
    return failure();
  }

  for (auto argTy : region->front().getArgumentTypes())
    addType(argTy);

  // Remember this function.
  if (name) {
    addString(name.getValue());
    namedFunctionIDs[name] = functions.size();
  }

  regionFunctionIDs[region] = functions.size();
  functions.push_back(
      BEFFunctionEntry(name, getRegionFunctionType(region), region));
  return success();
}

// Return the index of the specified function name, returning -1 if the
// function name cannot be found.
ssize_t EntityTable::getFunctionNamed(StringAttr name) const {
  auto iter = namedFunctionIDs.find(name);
  if (iter == namedFunctionIDs.end())
    return -1;
  return iter->second;
}

void EntityTable::addPrimitive(Operation *primitive) {
  StringAttr name = primitive->getName().getIdentifier();
  // Remember the primitive.
  if (!primitiveIDs.insert({name, primitives.size()}).second)
    return;
  primitives.push_back(name);

  // If we haven't seen it already, add it to the string table.
  addString(name);
}

void EntityTable::addLocation(Location loc) {
  StringAttr filename;
  unsigned line = 0, col = 0;
  // TODO: could support more powerful location approach.
  if (auto fileLineCol = loc.dyn_cast<FileLineColLoc>()) {
    filename = fileLineCol.getFilename();
    line = fileLineCol.getLine();
    col = fileLineCol.getColumn();
  } else {
    // TODO Could look through fancy locations to get a FileLineCol in most
    // cases.
    filename = StringAttr::get(loc.getContext());
  }

  locationFilenames.insert(filename);
  locations.insert(LocationTuple{filename, line, col});
}

void EntityTable::addAttributeType(Attribute attr) {
  if (auto intAttr = attr.dyn_cast<IntegerAttr>())
    addType(intAttr.getType());

  if (auto floatAttr = attr.dyn_cast<FloatAttr>())
    addType(floatAttr.getType());

  if (auto arrAttr = attr.dyn_cast<ArrayAttr>()) {
    for (auto attr : arrAttr.getValue())
      addAttributeType(attr);
  }
}

/// Walk the IR to build the symbol tables, this returns true on failure.
bool EntityTable::collect(ModuleOp module, bool disableOptionalSections) {
  std::vector<std::pair<SymbolRefAttr, Location>> symbolRefAttrs;

  auto opWalker = [&](Operation *op) -> WalkResult {
    // Ignore the module itself.
    if (op == module.getOperation())
      return WalkResult::advance();

    // Ignore the OutputOp: it gets special handling, instead of being treated
    // like a primitive.
    if (isa<BEF::OutputOp>(op))
      return WalkResult::advance();

    addLocation(op->getLoc());

    auto *curBlock = op->getBlock();

    // Notice the result and argument types of the ops.
    for (auto result : op->getResults())
      addType(result.getType());

    for (auto operand : op->getOperands()) {
      // Verify that the operand is defined inside the current region.  We
      // don't support references to outer regions.
      if (operand.getParentBlock() != curBlock)
        return op->emitError()
               << "BEF executor only supports references to primitives within"
               << " the current region";
    }

    // We treat functions specially, putting them into the symbol table and
    // ignoring their attributes.
    if (auto fn = dyn_cast<BEF::FuncOp>(op)) {
      if (failed(addBEFFunction(&fn.getBody(), fn.getNameAttr())))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    addPrimitive(op);

    // Keep track of any attributes used by this op.
    for (auto attr : op->getAttrs()) {
      // If this is a special attribute, ignore it.
      if (classifyAttribute(attr.getName()) != WellKnownAttrs::Unknown)
        continue;

      // Check to make sure that this is a supported attribute, if not,
      // reject it.
      if (!isSupportedAttribute(attr.getValue()))
        return op->emitError() << "BEF files cannot encode the '"
                               << attr.getName() << "' attribute";

      if (auto fn_attr = attr.getValue().dyn_cast<SymbolRefAttr>()) {
        // Keep track of function attributes specially so we can diagnose
        // them.
        symbolRefAttrs.push_back({fn_attr, op->getLoc()});

      } else {
        if (!disableOptionalSections) {
          // Add attribute names and types for attribute types section and
          // attribute names section. These will be ignored by executor.
          addString(attr.getName());
          addAttributeType(attr.getValue());
        }

        // We just keep track of the top level attributes in the "attributes"
        // list, we do not recursively process them.  This is because we want to
        // emit arrays of scalars as "flat" attributes where all the values are
        // inline.
        if (attributes.insert(attr.getValue()))
          // If this attribute is new, remember what used it.
          attributeLocs.push_back(op->getLoc());
      }
    }

    // Keep add any regions used by this op as BEF functions.
    for (auto &region : op->getRegions()) {
      if (failed(addBEFFunction(&region, StringAttr())))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };

  // Walk the whole module finding all the things.
  if (module.walk(opWalker).wasInterrupted())
    return true;

  // If we're successful, check to make sure that all functions can be
  // resolved.
  for (auto attrAndLoc : symbolRefAttrs) {
    if (getFunctionNamed(attrAndLoc.first.getRootReference()) == -1) {
      mlir::emitError(attrAndLoc.second)
          << "function " << attrAndLoc.first << " not defined";
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// BEFModuleEmitter
//===----------------------------------------------------------------------===//

namespace {
/// This emitter generates the top level structure of the BEF file.
class BEFModuleEmitter : public BEFEmitter {
public:
  explicit BEFModuleEmitter(ModuleOp module) : module(module) {}

  /// Walk the IR to build the symbol tables, this returns true on failure.
  bool collectEntities(bool disableOptionalSections) {
    return entities.collect(module, disableOptionalSections);
  }
  void emitLocationInfo();
  void emitStrings();
  void emitAttributes(BEFEmitter *attrTypesSection);

  bool hadError = false;

private:
  ModuleOp module;
  EntityTable entities;
};
} // end anonymous namespace

void BEFModuleEmitter::emitLocationInfo() {
  DenseMap<StringAttr, BEFOffset> filenameOffsets;
  BEFEmitter filenamesSection;
  for (StringAttr filename : entities.locationFilenames) {
    filenameOffsets[filename] = filenamesSection.getOffset();
    filenamesSection.emitNulTerminatedString(filename.getValue());
  }

  emitSection(BEF::SectionID::LOCATION_FILENAMES, filenamesSection);

  // Emit each of the positions and remember the offsets within the section.
  BEFEmitter positionsSection;
  for (auto position : entities.locations) {
    entities.locationOffsets[position] = positionsSection.getOffset();
    positionsSection.emitOffset(filenameOffsets[std::get<0>(position)]);
    positionsSection.emitVBR(std::get<1>(position));
    positionsSection.emitVBR(std::get<2>(position));
  }

  emitSection(BEF::SectionID::LOCATION_POSITIONS, positionsSection);
}

void BEFModuleEmitter::emitStrings() {
  // We have an ordered collection of strings: sort them alphabetically to make
  // them stable.
  std::vector<StringRef> stringsInOrder;
  stringsInOrder.reserve(entities.strings.size());
  for (const auto &str : entities.strings)
    stringsInOrder.push_back(str.getKey());
  llvm::sort(stringsInOrder.begin(), stringsInOrder.end());

  // Now that we have all the strings in order, emit them and remember their
  // offsets in the string section.
  BEFEmitter stringSection;
  for (StringRef string : stringsInOrder) {
    entities.stringOffsets[string] = stringSection.getOffset();
    stringSection.emitNulTerminatedString(string);
  }

  emitSection(BEF::SectionID::STRINGS, stringSection);
}

//===----------------------------------------------------------------------===//
// Attribute Emission
//===----------------------------------------------------------------------===//

/// This is the emitter that builds the attributes section of a BEF.
///
/// This is a bit trickier than it may seem because we don't always recursively
/// process all elements - we leave flat attributes inline, and don't
/// redundantly emit the scalar values.
class AttributesEmitter : public BEFEmitter {
public:
  explicit AttributesEmitter(EntityTable &entities, bool &hadError)
      : entities(entities), hadError(hadError) {}

  BEFOffset emitAttribute(Attribute attr, Location loc);

  size_t getNumAttributesEmitted() const { return numAttributesEmitted; }

private:
  BEFOffset emitAttributeInternal(Attribute attr, Location loc);
  BEFOffset emitIntegerAttribute(const APInt &value, Location loc);
  BEFOffset emitStringAttribute(StringRef value);
  BEFOffset emitArrayAttribute(ArrayAttr value, Location loc);
  EntityTable &entities;
  size_t numAttributesEmitted = 0;
  bool &hadError;
};

BEFOffset AttributesEmitter::emitIntegerAttribute(const APInt &value,
                                                  Location loc) {
  size_t bytes = value.getBitWidth() / 8;
  alignTo(bytes);
  auto offset = getOffset();

  switch (value.getBitWidth()) {
  default:
    mlir::emitError(loc) << "unsupported integer width " << value.getBitWidth();
    hadError = true;
    break;
  case 1:
    emitByte(static_cast<uint8_t>(value.getLimitedValue()));
    break;
  case 8:
  case 16:
  case 32:
  case 64:
    uint64_t v = value.getLimitedValue();
    for (unsigned i = 0; i != bytes; ++i) {
      emitByte(static_cast<uint8_t>(v & 255));
      v >>= 8;
    }
    break;
  }
  return offset;
}

BEFOffset AttributesEmitter::emitStringAttribute(StringRef value) {
  // In order to support strings of data with NUL characters in them, we emit
  // strings as modified Pascal-style strings, that start with a VBR length. The
  // string passed to the function points to the start of the data, but the
  // bytes before it specify how long the string is.  To enforce this notion, we
  // don't emit a zero byte at the end of the string data.
  emitInvertedVBR(value.size());
  auto offset = getOffset();
  emitStringData(value);
  return offset;
}

BEFOffset AttributesEmitter::emitArrayAttribute(ArrayAttr array, Location loc) {
  // We emit dense arrays of integers and floating point values.
  auto elts = array.getValue();

  BEFOffset sizeStart = getOffset();

  // Emit the number of elements first and then emit all of the attributes
  // consecutively, returning the offset of the first element in the attribute
  // array.
  emitInvertedVBR(array.size());

  BEFOffset sizeEnd = getOffset();

  // The attribute array is empty. We just return the offset past the size.
  if (elts.empty())
    return sizeEnd;

  // Emit the first element of the array, this could insert padding if it needs
  // larger alignment then the current offset.
  auto firstEltOffset = emitAttributeInternal(elts.front(), loc);
  for (auto elt : elts.drop_front())
    (void)emitAttributeInternal(elt, loc);

  // If there is a gap between the size and the first attribute value, we
  // move the size to be immediately before the first attribute value to
  // remove the gap.
  if (sizeEnd.value != firstEltOffset.value) {
    size_t sizeOfArraySizeInBytes = sizeEnd.value - sizeStart.value;
    memmove(getRawPointer(firstEltOffset) - sizeOfArraySizeInBytes,
            getRawPointer(sizeStart), sizeOfArraySizeInBytes);
    // Set the unused bytes to the alignment filler.
    memset(getRawPointer(sizeStart), BEF::ALIGNMENT_BYTE,
           firstEltOffset.value - sizeEnd.value);
  }
  return firstEltOffset;
}

BEFOffset AttributesEmitter::emitAttributeInternal(Attribute attr,
                                                   Location loc) {
  if (auto intAttr = attr.dyn_cast<IntegerAttr>())
    return emitIntegerAttribute(intAttr.getValue(), loc);

  if (auto fpAttr = attr.dyn_cast<FloatAttr>()) {
    const APFloat &value = fpAttr.getValue();
    auto size = APFloat::getSizeInBits(value.getSemantics());
    if (size == 16 || size == 32 || size == 64)
      return emitIntegerAttribute(value.bitcastToAPInt(), loc);
  }

  if (auto string = attr.dyn_cast<StringAttr>())
    return emitStringAttribute(string.getValue());

  if (auto array = attr.dyn_cast<ArrayAttr>())
    return emitArrayAttribute(array, loc);

  mlir::emitError(loc) << "unsupported attribute " << attr;
  hadError = true;
  return {0};
}

BEFOffset AttributesEmitter::emitAttribute(Attribute attr, Location loc) {
  // Since there might be nested array attributes, they might have been already
  // emitted.
  auto attrIt = entities.attributeOffsets.find(attr);
  if (attrIt != entities.attributeOffsets.end())
    return attrIt->second;

  auto offset = emitAttributeInternal(attr, loc);
  entities.attributeOffsets[attr] = offset;

  // We emitted an attribute.
  numAttributesEmitted++;
  return offset;
}

/// Emit the ATTRIBUTES section.  If attrTypesSection is non-null, we emit type
/// information into it.
void BEFModuleEmitter::emitAttributes(BEFEmitter *attrTypesSection) {
  // Emit attributes and record them in EntityIndex. Nested array attributes
  // will be traversed recursively and their elements will be emitted and
  // recorded before the top level offsets array is emitted.
  AttributesEmitter attributesSection(entities, hadError);
  for (auto attrAndLoc :
       llvm::zip(entities.attributes, entities.attributeLocs)) {
    (void)attributesSection.emitAttribute(std::get<0>(attrAndLoc),
                                          std::get<1>(attrAndLoc));
  }

  if (attrTypesSection) {
    // TODO: Emit attrTypesSection if it is present.
  }

  emitSection(BEF::SectionID::ATTRIBUTES, attributesSection);
}

//===----------------------------------------------------------------------===//
// Top level emission logic
//===----------------------------------------------------------------------===//

/// This function converts the specified MLIR module containing a executor
/// compatible program to the BinaryExecutableFormat (BEF) format, which is
/// the binary format that the executor takes.  When
/// `disableOptionalSections` is true, this omits metadata sections required
/// to reverse a BEF file back into a .mlir file.
///
/// On error, this emits the error message through the MLIR error handler,
/// and returns an empty vector.
std::vector<uint8_t> M::translateMLIRToBEF(ModuleOp module,
                                           bool disableOptionalSections) {
  BEFModuleEmitter emitter(module);

  // Build the entities table.
  if (emitter.collectEntities(disableOptionalSections))
    return {};

  // If optional section emission is enabled, these collect the data to emit
  // into the respective sections.
  BEFEmitter attrTypesSection, attrNamesSection, registerTypesSection;

  // Emit magic numbers and format version.
  emitter.emitBytes({BEF::MAGIC1, BEF::MAGIC2 + BEF::VERSION0});
  emitter.emitLocationInfo();
  emitter.emitStrings();

  emitter.emitAttributes(disableOptionalSections ? nullptr : &attrTypesSection);

#if 0 // TODO
  emitter.emitPrimitives();
   emitter.emitTypes();
   emitter.emitFunctions(disableOptionalSections ? nullptr, &attrNamesSection,
                        disableOptionalSections ? nullptr,
                         &registerTypesSection);
   emitter.emitFunctionIndex();
#endif

  // Emit all the optional sections at the end if they're requested.
  if (!disableOptionalSections) {
    emitter.emitSection(BEF::SectionID::ATTRIBUTE_TYPES, attrTypesSection);
    emitter.emitSection(BEF::SectionID::ATTRIBUTE_NAMES, attrNamesSection);
    emitter.emitSection(BEF::SectionID::REGISTER_TYPES, registerTypesSection);
  }

  // If emission diagnosed an error along the way and emitted a diagnostic,
  // then we should propagate the failure up.
  if (emitter.hadError)
    return {};

  return emitter.takeResult();
}
