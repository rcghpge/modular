//===- MLIRToBEF.cpp - MLIR to BEF file format converter ------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the entrypoint for the MLIRToBEF library.  The converter
// is implemented in three phases.  The first phase identifies all of the
// strings and attributes that need to be emitted to the string/attribute pool.
// The second phase optimizes and emits the strings and attributes to the file
// and remembers their offsets.  The third phase emits all of the regions in the
// MLIR program.
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

struct BEFFunctionSignature {
  size_t numArguments, numResults;
};

static BEFFunctionSignature getBEFFunctionSignature(Region *region) {
  // Emit information about the type of the function.
  auto *block = &region->getBlocks().front();

  // BEF functions produce results with the "bef.output" operation.
  auto output = dyn_cast<BEF::OutputOp>(&block->back());
  size_t numResults = output ? output->getNumOperands() : 0;
  return BEFFunctionSignature{block->getNumArguments(), numResults};
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

  // Emit a guaranteed 2-byte integer aligned to 2 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void emitInt2(uint16_t value) {
    alignTo(2);
    emitByte(value & 0xFF);
    emitByte(value >> 8);
  }

  // Emit a guaranteed 4-byte integer aligned to 4 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void emitInt4(uint32_t value) {
    alignTo(4);
    uint8_t data[] = {uint8_t(value & 0xFF), uint8_t((value >> 8) & 0xFF),
                      uint8_t((value >> 16) & 0xFF),
                      uint8_t((value >> 24) & 0xFF)};
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
    emitByte(static_cast<uint8_t>(sectionID));

    // Emit a length.  The low bit indicates whether alignment padding is
    // present.  We fill it in later.
    emitVBR(sectionData.size() << 1);
    size_t needsAlignmentPaddingOffset = result.size() - 1;

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
    StringAttr name;
    if (auto opaque = dyn_cast<BEF::OpaqueOp>(primitive))
      name = opaque.opnameAttr();
    else
      name = primitive->getName().getIdentifier();

    auto it = primitiveIDs.find(name);
    assert(it != primitiveIDs.end() && "Unknown primitive");
    return it->second;
  }

  BEFOffset getStringOffset(StringAttr str) const {
    auto it = stringOffsets.find(str);
    assert(it != stringOffsets.end() && "unknown string");
    return it->second;
  }

  BEFOffset getAttributeOffset(Attribute attr) const {
    auto it = attributeOffsets.find(attr);
    assert(it != attributeOffsets.end() && "unknown attribute");
    return it->second;
  }

  void addBEFFunctionOffset(StringAttr name, BEFOffset offset, Location loc,
                            BEFFunctionSignature signature) {
    functionIndexEntries.push_back(
        {getStringOffset(name), offset, getLocationOffset(loc), signature});
  }

  /// Return the BEFOffset for the specified location entry.
  BEFOffset getLocationOffset(Location loc) const {
    auto locInfo = decomposeLocation(loc);
    auto it = locationOffsets.find(locInfo);
    assert(it != locationOffsets.end() && "unknown location");
    return it->second;
  }

  mlir::InFlightDiagnostic emitError(Location loc) {
    hadError = true;
    return mlir::emitError(loc);
  }

private:
  using LocationTuple = std::tuple<StringAttr, unsigned, unsigned>;
  static LocationTuple decomposeLocation(Location loc);
  void addString(StringAttr string) { strings.insert(string); }
  void addType(Type type);
  LogicalResult addBEFFunction(Region *region, StringAttr name);
  void addPrimitiveOpcode(StringAttr opcode);
  void addLocation(Location loc);
  void addAttributeType(Attribute attr);

public:
  bool hadError = false;

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
  std::vector<StringAttr> primitives;
  DenseMap<StringAttr, unsigned> primitiveIDs;

  /// BEFFunctionEntry keeps track of each of the top level `func`'s and the
  /// regions attached to control flow statements.  These all turn into BEF
  /// functions.
  struct BEFFunctionEntry {
    BEFFunctionEntry(StringAttr name, BEFFunctionSignature signature,
                     Region *region)
        : name(name), signature(signature), region(region) {}

    StringAttr name;
    BEFFunctionSignature signature;
    Region *region;
  };

  // List of functions that we need to emit, along with a name if they came
  // from a top level function.
  std::vector<BEFFunctionEntry> functions;
  DenseMap<Region *, unsigned> regionFunctionIDs;
  DenseMap<StringAttr, unsigned> namedFunctionIDs;

  struct FunctionIndexEntry {
    BEFOffset nameOffset;
    BEFOffset functionOffset;
    BEFOffset locationOffset;
    BEFFunctionSignature signature;
  };

  // This follows the format of the FunctionIndex section.
  std::vector<FunctionIndexEntry> functionIndexEntries;

  // Types we've seen so far.
  std::vector<Type> types;
  DenseMap<Type, unsigned> typeIDs;

  // All of the strings we need to emit the BEF file.
  llvm::SetVector<StringAttr> strings;

  /// This is the offset of each known string in the STRINGS section.
  DenseMap<StringAttr, BEFOffset> stringOffsets;

  // This is all of the filenames referred to by locations in the file.
  llvm::SetVector<StringAttr> locationFilenames;

  // This is a list of locations within the file, the first element of the
  // tuple is a index into locationFilenames, the second and third are line/col
  // information.
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
    addString(name);
    namedFunctionIDs[name] = functions.size();
  }

  regionFunctionIDs[region] = functions.size();
  functions.push_back(
      BEFFunctionEntry(name, getBEFFunctionSignature(region), region));
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

void EntityTable::addPrimitiveOpcode(StringAttr opcode) {
  // Remember the primitive.
  if (!primitiveIDs.insert({opcode, primitives.size()}).second)
    return;
  primitives.push_back(opcode);

  // If we haven't seen it already, add it to the string table.
  addString(opcode);
}

auto EntityTable::decomposeLocation(Location loc) -> LocationTuple {
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
  return LocationTuple{filename, line, col};
}

void EntityTable::addLocation(Location loc) {
  auto locationInfo = decomposeLocation(loc);

  locationFilenames.insert(std::get<0>(locationInfo));
  locations.insert(locationInfo);
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

    auto processAttr = [&](StringAttr name, Attribute attr) {
      // If this is a special bef.nonstrict attribute, ignore it.
      if (name.getValue() == BEF::NONSTRICT_ATTR)
        return;

      if (auto fn_attr = attr.dyn_cast<SymbolRefAttr>()) {
        // Keep track of function attributes specially so we can diagnose
        // them.
        symbolRefAttrs.push_back({fn_attr, op->getLoc()});
      } else {
        if (!disableOptionalSections) {
          // Add attribute names and types for attribute types section and
          // attribute names section. These will be ignored by executor.
          addString(name);
          addAttributeType(attr);
        }

        // We just keep track of the top level attributes in the "attributes"
        // list: do not recursively process them.  This is because we want to
        // emit arrays of scalars as "flat" attributes where all the values are
        // inline.
        if (attributes.insert(attr))
          // If this attribute is new, remember what used it.
          attributeLocs.push_back(op->getLoc());
      }
    };

    // Opaque BEF operations encode some fields in a special way.
    if (auto opaque = dyn_cast<BEF::OpaqueOp>(op)) {
      addPrimitiveOpcode(opaque.opnameAttr());

      // Encode the attributes in the attrs array, not all the attributes on the
      // OpaqueOp itself.
      for (auto attr : opaque.attrs())
        processAttr(StringAttr::get(op->getContext()), attr);
    } else {
      addPrimitiveOpcode(op->getName().getIdentifier());
      // Keep track of any attributes used by this op.
      for (auto attr : op->getAttrs())
        processAttr(attr.getName(), attr.getValue());
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
// Attribute Emission
//===----------------------------------------------------------------------===//

namespace {
struct OffsetAndAttrKind {
  BEFOffset offset;
  BEF::AttributeKind attrKind;
};
} // namespace

/// This is the emitter that builds the attributes section of a BEF.
///
/// This is a bit trickier than it may seem because we don't always recursively
/// process all elements - we leave flat attributes inline, and don't
/// redundantly emit the scalar values.
namespace {
class AttributesEmitter : public BEFEmitter {
public:
  explicit AttributesEmitter(EntityTable &entities) : entities(entities) {}

  OffsetAndAttrKind emitAttribute(Attribute attr, Location loc);

private:
  OffsetAndAttrKind emitAttributeInternal(Attribute attr, Location loc);
  OffsetAndAttrKind emitIntegerAttribute(const APInt &value, Location loc);
  OffsetAndAttrKind emitStringAttribute(StringRef value);
  OffsetAndAttrKind emitArrayAttribute(ArrayAttr value, Location loc);
  EntityTable &entities;
};
} // end anonymous namespace.

OffsetAndAttrKind AttributesEmitter::emitIntegerAttribute(const APInt &value,
                                                          Location loc) {
  size_t bytes = (value.getBitWidth() + 7) / 8;
  alignTo(bytes);
  auto offset = getOffset();
  BEF::AttributeKind kind;

  switch (value.getBitWidth()) {
  default:
    entities.emitError(loc)
        << "unsupported integer width " << value.getBitWidth();
    return {offset, BEF::AttributeKind::INVALID};
  case 1:
    kind = BEF::AttributeKind::I1;
    break;
  case 8:
    kind = BEF::AttributeKind::I8;
    break;
  case 16:
    kind = BEF::AttributeKind::I16;
    break;
  case 32:
    kind = BEF::AttributeKind::I32;
    break;
  case 64:
    kind = BEF::AttributeKind::I64;
    break;
  }

  uint64_t v = value.getLimitedValue();
  for (unsigned i = 0; i != bytes; ++i) {
    emitByte(static_cast<uint8_t>(v & 255));
    v >>= 8;
  }

  return {offset, kind};
}

OffsetAndAttrKind AttributesEmitter::emitStringAttribute(StringRef value) {
  // In order to support strings of data with NUL characters in them, we emit
  // strings as modified Pascal-style strings, that start with a VBR length. The
  // string passed to the function points to the start of the data, but the
  // bytes before it specify how long the string is.  To enforce this notion, we
  // don't emit a zero byte at the end of the string data.
  emitInvertedVBR(value.size());
  auto offset = getOffset();
  emitStringData(value);
  return {offset, BEF::AttributeKind::STRING};
}

OffsetAndAttrKind AttributesEmitter::emitArrayAttribute(ArrayAttr array,
                                                        Location loc) {
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
    return {sizeEnd, BEF::AttributeKind::ARRAY_TYPE};

  // Emit the first element of the array, this could insert padding if it needs
  // larger alignment then the current offset.
  auto firstEltOffsetAndKind = emitAttributeInternal(elts.front(), loc);

  // Reject unsupported elements of an array.
  switch (firstEltOffsetAndKind.attrKind) {
  default:
    entities.emitError(loc)
        << "unsupported array element attribute " << elts.front();
    return {sizeEnd, BEF::AttributeKind::INVALID};
  case BEF::AttributeKind::I1:
  case BEF::AttributeKind::I8:
  case BEF::AttributeKind::I16:
  case BEF::AttributeKind::I32:
  case BEF::AttributeKind::I64:
  case BEF::AttributeKind::F32:
  case BEF::AttributeKind::F64:
    break;
  }

  for (auto elt : elts.drop_front()) {
    auto nextEltKind = emitAttributeInternal(elt, loc).attrKind;
    if (nextEltKind != firstEltOffsetAndKind.attrKind) {
      entities.emitError(loc)
          << "unsupported array with heterogenous elements " << elt;
      return {sizeEnd, BEF::AttributeKind::INVALID};
    }
  }

  auto startOffset = firstEltOffsetAndKind.offset;

  // If there is a gap between the size and the first attribute value, we
  // move the size to be immediately before the first attribute value to
  // remove the gap.
  if (sizeEnd.value != startOffset.value) {
    size_t sizeOfArraySizeInBytes = sizeEnd.value - sizeStart.value;
    memmove(getRawPointer(startOffset) - sizeOfArraySizeInBytes,
            getRawPointer(sizeStart), sizeOfArraySizeInBytes);
    // Set the unused bytes to the alignment filler.
    memset(getRawPointer(sizeStart), BEF::ALIGNMENT_BYTE,
           startOffset.value - sizeEnd.value);
  }

  unsigned kindValue = (unsigned)firstEltOffsetAndKind.attrKind |
                       (unsigned)BEF::AttributeKind::ARRAY_TYPE;
  return {startOffset, BEF::AttributeKind(kindValue)};
}

OffsetAndAttrKind AttributesEmitter::emitAttributeInternal(Attribute attr,
                                                           Location loc) {
  if (auto intAttr = attr.dyn_cast<IntegerAttr>())
    return emitIntegerAttribute(intAttr.getValue(), loc);

  if (auto fpAttr = attr.dyn_cast<FloatAttr>()) {
    OffsetAndAttrKind intResult =
        emitIntegerAttribute(fpAttr.getValue().bitcastToAPInt(), loc);
    switch (intResult.attrKind) {
    default:
      mlir::emitError(loc) << "unsupported float type" << attr;
      return {{}, BEF::AttributeKind::INVALID};
    case BEF::AttributeKind::INVALID:
      return intResult; // Error already emitted.
    case BEF::AttributeKind::I32:
      return {intResult.offset, BEF::AttributeKind::F32};
    case BEF::AttributeKind::I64:
      return {intResult.offset, BEF::AttributeKind::F64};
    }
  }

  if (auto string = attr.dyn_cast<StringAttr>())
    return emitStringAttribute(string.getValue());

  if (auto array = attr.dyn_cast<ArrayAttr>())
    return emitArrayAttribute(array, loc);

  entities.emitError(loc) << "unsupported attribute " << attr;
  return {{0}, BEF::AttributeKind::INVALID};
}

OffsetAndAttrKind AttributesEmitter::emitAttribute(Attribute attr,
                                                   Location loc) {
  assert(!entities.attributeOffsets.count(attr) && "Attr already emitted?");
  auto offsetAndKind = emitAttributeInternal(attr, loc);
  entities.attributeOffsets[attr] = offsetAndKind.offset;
  return offsetAndKind;
}

//===----------------------------------------------------------------------===//
// BEFFunctionEmitter
//===----------------------------------------------------------------------===//

namespace {
// This is the emitter that builds the function entry of a BEF.
class BEFFunctionEmitter : public BEFEmitter {
public:
  BEFFunctionEmitter(EntityTable &entities, BEFEmitter *attributeNamesSection,
                     BEFEmitter *registerTypesSection)
      : entities(entities) {}

  void emitFunction(Region *region);

private:
  void emitRegisterTable(Block *block);
  unsigned getNumNonOutputUsers(Value result) const;
  void emitPrimitiveResultUsers(Value result, BEFEmitter &primitiveBody) const;
  void emitArgumentsPseudoOp(Block *block, BEFEmitter &primitiveList) const;
  void emitPrimitive(Operation *op, BEFEmitter &primitiveList) const;

  unsigned getRegisterNumber(Value reg) const {
    auto it = registerNumbers.find(reg);
    assert(it != registerNumbers.end() && "Unknown register");
    return it->second;
  }

  DenseMap<Value, unsigned> registerNumbers;
  DenseMap<Operation *, unsigned> primitiveIndices;

  EntityTable &entities;
};
} // end anonymous namespace.

void BEFFunctionEmitter::emitFunction(Region *region) {
  assert(region->getBlocks().size() == 1 && "should have a single block");
  auto &block = region->getBlocks().front();

  // Emit the register table.
  emitRegisterTable(&block);

  // Get a dense numbering of primitives.
  unsigned numPrimitives = 0;

  // If the function has arguments, we emit a pseudo-op that provides the
  // argument values.
  if (block.getNumArguments() != 0)
    ++numPrimitives;

  for (auto &op : block.getOperations()) {
    if (!isa<BEF::OutputOp>(op))
      primitiveIndices[&op] = numPrimitives++;
  }

  // Emit a count of primitives, then the offset of each primitive (from the
  // start of the primitive list) then each primitive is emitted in turn.
  emitVBR(numPrimitives);

  BEFEmitter primitiveList;

  // If we have arguments, emit a pseudo op (with no opcode) that produces the
  // registers on entry to the function.
  if (block.getNumArguments() != 0) {
    // Offset of the primitive in the list.
    emitOffset(primitiveList.getOffset());
    // Pseudo has zero operands that need to be available.
    emitVBR(0);
    emitArgumentsPseudoOp(&block, primitiveList);
  }

  for (auto &op : block) {
    // Output primitive gets special processing.
    if (isa<BEF::OutputOp>(&op))
      continue;

    bool isNonStrict = false;
    if (auto opaque = dyn_cast<BEF::OpaqueOp>(op))
      isNonStrict = opaque.nonstrict();
    else
      isNonStrict = op.getAttr(BEF::NONSTRICT_ATTR) != Attribute();

    // Offset of the primitive in the list.
    emitOffset(primitiveList.getOffset());
    // Number of operands that need to be available before it is ready to go.
    auto numOperandsBeforeRunning = op.getNumOperands();

    // We set the number to 1 for non-strict primitives so they get kicked off
    // as soon as any argument is avaiable.  We use 1 instead of zero because we
    // primitives with no operands ready are likely to just wait anyway.
    if (isNonStrict && numOperandsBeforeRunning)
      numOperandsBeforeRunning = 1;

    emitVBR(numOperandsBeforeRunning);
    emitPrimitive(&op, primitiveList);
  }

  // Emit the result registers list at the end, if present.
  if (auto outputOp = dyn_cast<BEF::OutputOp>(block.getTerminator())) {
    for (auto operand : outputOp.getOperands())
      emitVBR(getRegisterNumber(operand));
  }

  // Once we're done, we can emit the primitive data after the primitive index
  // list. Note that primitive entries are fixed32 integers with 4-byte
  // alignment.
  alignTo(BEF::PRIMITIVE_ENTRY_ALIGNMENT);
  emitEmitter(primitiveList);

  primitiveIndices.clear();
}

void BEFFunctionEmitter::emitRegisterTable(Block *block) {
  BEFEmitter regTable;
  unsigned numRegisters = 0;

  auto emitRegister = [&](Value reg) {
    // Then the use-count.
    regTable.emitVBR(std::distance(reg.use_begin(), reg.use_end()));
    registerNumbers[reg] = numRegisters++;
  };

  for (auto arg : block->getArguments())
    emitRegister(arg);

  for (auto &op : *block)
    for (auto result : op.getResults())
      emitRegister(result);

  // Emit the number of registers, then the register table.
  emitVBR(numRegisters);
  emitEmitter(regTable);
}

unsigned BEFFunctionEmitter::getNumNonOutputUsers(Value result) const {
  int numUsers = 0;
  for (auto *user : result.getUsers()) {
    // Ignore the 'bef.output' op, it gets special handling.
    if (!isa<BEF::OutputOp>(user))
      ++numUsers;
  }
  return numUsers;
}

void BEFFunctionEmitter::emitPrimitiveResultUsers(
    Value result, BEFEmitter &primitiveBody) const {
  for (auto *user : result.getUsers()) {
    // Ignore the 'bef.output' op, it gets special handling.
    if (isa<BEF::OutputOp>(user))
      continue;

    auto it = primitiveIndices.find(user);
    assert(it != primitiveIndices.end() && "Invalid user");
    primitiveBody.emitInt4(it->second);
  }
}

void BEFFunctionEmitter::emitArgumentsPseudoOp(
    Block *block, BEFEmitter &primitiveList) const {
  // This primitive starts with a dummy code and a dummy location. And this
  // primitive only has results and UsedBys in its body.
  primitiveList.emitInt4(0);                        // code
  primitiveList.emitInt4(0);                        // location
  primitiveList.emitInt2(0);                        // # operands
  primitiveList.emitInt2(0);                        // # attributes
  primitiveList.emitInt2(0);                        // # functions
  primitiveList.emitInt2(block->getNumArguments()); // # results

  for (auto arg : block->getArguments())
    primitiveList.emitInt4(getNumNonOutputUsers(arg));
  for (auto arg : block->getArguments())
    primitiveList.emitInt4(getRegisterNumber(arg));
  for (auto arg : block->getArguments())
    emitPrimitiveResultUsers(arg, primitiveList);
}

void BEFFunctionEmitter::emitPrimitive(Operation *op,
                                       BEFEmitter &primitiveList) const {
  // Collect the regions and attributes to emit.
  SmallVector<uint32_t> functionsToEmit, attributesToEmit;
  auto processAttr = [&](StringAttr name, Attribute value) {
    if (auto fnAttr = value.dyn_cast<FlatSymbolRefAttr>()) {
      // Function references are emitted as regions.
      functionsToEmit.push_back(entities.getFunctionNamed(fnAttr.getAttr()));
    } else {
      attributesToEmit.push_back(entities.getAttributeOffset(value).value);
    }
  };

  // bef.opaque operands store attributes in an explicit way since we don't know
  // the attribute names.
  if (auto opaque = dyn_cast<BEF::OpaqueOp>(op)) {
    // All encoded attributes are stored in a single array.
    for (auto attr : opaque.attrs())
      processAttr(StringAttr(), attr);
  } else {
    for (auto attrNamePair : op->getAttrs()) {
      // Don't encode the bef.nonstrict attribute explicitly.
      if (attrNamePair.getName().getValue() == BEF::NONSTRICT_ATTR)
        continue;
      processAttr(attrNamePair.getName(), attrNamePair.getValue());
    }
  }

  // Regions on the op are also considered to be BEF functions.
  for (auto &region : op->getRegions())
    functionsToEmit.push_back(entities.getFunctionID(region));

  // Ensure this primitive isn't too big to encode.
  if (op->getNumOperands() > 65535 || attributesToEmit.size() > 65535 ||
      functionsToEmit.size() > 65535 || op->getNumResults() > 65535)
    entities.emitError(op->getLoc()) << "operation too large to encode in BEF";

  // Emit the PRIMITIVE_HEADER records.
  primitiveList.emitInt4(entities.getPrimitiveID(op));
  primitiveList.emitInt4(entities.getLocationOffset(op->getLoc()).value);
  primitiveList.emitInt2(op->getNumOperands());
  primitiveList.emitInt2(attributesToEmit.size());
  primitiveList.emitInt2(functionsToEmit.size());
  primitiveList.emitInt2(op->getNumResults());

  // Emit PRIMITIVE_RESULT_TABLE, results with the primitives that use them.
  for (auto result : op->getResults())
    primitiveList.emitInt4(getNumNonOutputUsers(result));

  // Emit the PRIMITIVE_BODY contents.
  for (auto operand : op->getOperands())
    primitiveList.emitInt4(getRegisterNumber(operand));
  for (auto attrOffset : attributesToEmit)
    primitiveList.emitInt4(attrOffset);
  for (auto fnOffset : functionsToEmit)
    primitiveList.emitInt4(fnOffset);
  for (auto result : op->getResults())
    primitiveList.emitInt4(getRegisterNumber(result));
  // Then results with the primitives that use them.
  for (auto result : op->getResults())
    emitPrimitiveResultUsers(result, primitiveList);
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
  void emitLocationInfo(BEFEmitter &locationFilenames,
                        BEFEmitter &locationPositions);
  void emitStrings();
  void emitAttributes(BEFEmitter &attrKindSection);
  void emitPrimitiveNames();
  void emitBEFFunctions(BEFEmitter *attributeNamesSection,
                        BEFEmitter *registerTypesSection);
  void emitFunctionIndex();

  bool hadError() const { return entities.hadError; }

private:
  ModuleOp module;
  EntityTable entities;
};
} // end anonymous namespace

void BEFModuleEmitter::emitLocationInfo(BEFEmitter &locationFilenames,
                                        BEFEmitter &locationPositions) {
  DenseMap<StringAttr, BEFOffset> filenameOffsets;
  for (StringAttr filename : entities.locationFilenames) {
    filenameOffsets[filename] = locationFilenames.getOffset();
    locationFilenames.emitNulTerminatedString(filename.getValue());
  }

  // Emit each of the positions and remember the offsets within the section.
  for (auto position : entities.locations) {
    entities.locationOffsets[position] = locationPositions.getOffset();
    locationPositions.emitOffset(filenameOffsets[std::get<0>(position)]);
    locationPositions.emitVBR(std::get<1>(position));
    locationPositions.emitVBR(std::get<2>(position));
  }
}

void BEFModuleEmitter::emitStrings() {
  // Emit the strings to the string section and remember their offsets.
  BEFEmitter stringSection;
  for (StringAttr string : entities.strings) {
    entities.stringOffsets[string] = stringSection.getOffset();
    stringSection.emitNulTerminatedString(string);
  }

  emitSection(BEF::SectionID::STRINGS, stringSection);
}

/// Emit the ATTRIBUTES section, and the ATTRIBUTE_KIND_MD sections.
void BEFModuleEmitter::emitAttributes(BEFEmitter &attrKindsSection) {
  // Emit attributes and record them in EntityIndex. Nested array attributes
  // will be traversed recursively and their elements will be emitted and
  // recorded before the top level offsets array is emitted.
  AttributesEmitter attributesSection(entities);
  for (auto attrAndLoc :
       llvm::zip(entities.attributes, entities.attributeLocs)) {
    auto offsetAndKind = attributesSection.emitAttribute(
        std::get<0>(attrAndLoc), std::get<1>(attrAndLoc));

    // Emit metadata for each attribute we put out so BEF2MLIR can interpret the
    // data correctly.
    attrKindsSection.emitOffset(offsetAndKind.offset);
    attrKindsSection.emitByte((unsigned)offsetAndKind.attrKind);
  }

  emitSection(BEF::SectionID::ATTRIBUTES, attributesSection);
}

void BEFModuleEmitter::emitPrimitiveNames() {
  // The primitives are already in a stable order, so just emit them in the
  // order they were found.
  BEFEmitter primitivesNamesSection;
  for (auto op : entities.primitives)
    primitivesNamesSection.emitOffset(entities.getStringOffset(op));

  emitSection(BEF::SectionID::PRIMITIVE_NAMES, primitivesNamesSection);
}

/// Emit the BEF_FUNCTIONS section.  If optional metadata is enabled, the
/// attributeNamesSection/registerTypesSection will be non-null and should be
/// filled in.
void BEFModuleEmitter::emitBEFFunctions(BEFEmitter *attributeNamesSection,
                                        BEFEmitter *registerTypesSection) {
  BEFFunctionEmitter functionsSection(entities, attributeNamesSection,
                                      registerTypesSection);

  for (auto function : entities.functions) {
    // Remember that we emitted this region to this offset.
    entities.addBEFFunctionOffset(function.name, functionsSection.getOffset(),
                                  function.region->getLoc(),
                                  function.signature);
    functionsSection.emitFunction(function.region);
  }

  emitSection(BEF::SectionID::BEF_FUNCTIONS, functionsSection);
}

void BEFModuleEmitter::emitFunctionIndex() {
  BEFEmitter functionIndexSection;
  for (const auto &entry : entities.functionIndexEntries) {
    functionIndexSection.emitOffset(entry.functionOffset);
    functionIndexSection.emitOffset(entry.nameOffset);
    functionIndexSection.emitOffset(entry.locationOffset);
    functionIndexSection.emitVBR(entry.signature.numArguments);
    functionIndexSection.emitVBR(entry.signature.numResults);
  }

  emitSection(BEF::SectionID::BEF_FUNCTION_INDEX, functionIndexSection);
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

  // Sections describing location information.  We collect this early, but emit
  // it near the end of the file since it isn't typically needed by the
  // BEFExecutor.
  BEFEmitter locationFilenames, locationPositions;

  // If optional section emission is enabled, these collect the data to emit
  // into the respective sections.
  BEFEmitter attrKindsSection, attrNamesSection, registerTypesSection;

  // Emit magic numbers and format version.
  emitter.emitBytes({BEF::MAGIC1, BEF::MAGIC2 + BEF::VERSION0});
  emitter.emitLocationInfo(locationFilenames, locationPositions);
  emitter.emitStrings();

  emitter.emitAttributes(attrKindsSection);

  emitter.emitPrimitiveNames();
  emitter.emitBEFFunctions(
      disableOptionalSections ? nullptr : &attrNamesSection,
      disableOptionalSections ? nullptr : &registerTypesSection);
  emitter.emitFunctionIndex();

  // Now that we've emitted all of the critical sections used in the common case
  // of the BEF emitter, emit metadata sections.
  emitter.emitSection(BEF::SectionID::LOCATION_FILENAMES, locationFilenames);
  emitter.emitSection(BEF::SectionID::LOCATION_POSITIONS, locationPositions);

  // Unconditionally emit metadata that describes the attribute section, this
  // is the minimal metadata needed by BEF2MLIR.
  emitter.emitSection(BEF::SectionID::ATTRIBUTE_KIND_MD, attrKindsSection);

  // Emit all the optional sections at the end if they're requested.
  if (!disableOptionalSections) {
    emitter.emitSection(BEF::SectionID::ATTRIBUTE_NAME_MD, attrNamesSection);
    emitter.emitSection(BEF::SectionID::REGISTER_TYPE_MD, registerTypesSection);
  }

  // If emission diagnosed an error along the way and emitted a diagnostic,
  // then we should propagate the failure up.
  if (emitter.hadError())
    return {};

  return emitter.takeResult();
}
