//===- MLIRToBEF.cpp ------------------------------------------------------===//
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
#include "GraphRT/BEFDialect/BEFNonStrict.h"
#include "GraphRT/BEFDialect/BEFOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringSet.h"
using namespace M;

/// This holds an offset into a BEF section.
struct BEFOffset {
  size_t value;
};

/// This keeps track of the signature for an invokable (a bef.invokable or
/// region in the MLIR file).
struct InvokableSignature {
  size_t numArguments, numResults;
};

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
      emitByte(BEF::kAlignmentByte);

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

  /// Emit an offset to a primitive, which is known to be aligned, so we can
  /// emit a smaller number.
  void emitPrimitiveOffsetNumber(BEFOffset offset) {
    assert((offset.value & (BEF::kPrimitiveEntryAlignment - 1)) == 0 &&
           "offset to primitive isn't correctly aligned");
    emitVBR(offset.value / BEF::kPrimitiveEntryAlignment);
  }

  // Many parts of the emitter logic includes forward references into stuff
  // that hasn't been emitted and has variable size.  This is handled by making
  // nested emitters.  This helper emits the subpieces once they are
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
  bool collect(ModuleOp module, bool enableBEFMetadata);
  ssize_t getInvokableNamed(StringAttr name) const;

  unsigned getTypeIndex(Type type) const {
    auto it = typeIDs.find(type);
    assert(it != typeIDs.end() && "unregistered type");
    return it->second;
  }

  unsigned getInvokableID(const Region &region) const {
    auto it = invokableRegionIDs.find(&region);
    assert(it != invokableRegionIDs.end() &&
           "region not added to entity table");
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

  BEFOffset getMetadataStringOffset(StringAttr str) const {
    auto it = metadataStringOffsets.find(str);
    assert(it != metadataStringOffsets.end() && "unknown metadata string");
    return it->second;
  }

  BEFOffset getAttributeOffset(Attribute attr) const {
    auto it = attributeOffsets.find(attr);
    assert(it != attributeOffsets.end() && "unknown attribute");
    return it->second;
  }

  void addInvokableOffset(StringAttr name, BEFOffset offset,
                          InvokableSignature signature) {
    invokableIndexEntries.push_back({getStringOffset(name), offset, signature});
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
  void addMetadataString(StringAttr string) { metadataStrings.insert(string); }
  void addType(Type type);
  LogicalResult addInvokable(Region *region, StringAttr name);
  void addPrimitiveOpcode(StringAttr opcode);
  void addLocation(Location loc);

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

  /// InvokableEntry keeps track of each of the top level `bef.invokable`'s
  /// and the regions attached to control flow statements.  These all turn into
  /// invokables in the BEF file.
  struct InvokableEntry {
    InvokableEntry(StringAttr name, InvokableSignature signature,
                   Region *region)
        : name(name), signature(signature), region(region) {}

    StringAttr name;
    InvokableSignature signature;
    Region *region;
  };

  // List of invokables that we need to emit, along with a name if they came
  // from a bef.invokable.
  std::vector<InvokableEntry> invokables;
  DenseMap<Region *, unsigned> invokableRegionIDs;
  DenseMap<StringAttr, unsigned> namedInvokableIDs;

  struct InvokableIndexEntry {
    BEFOffset nameOffset;
    BEFOffset invokableOffset;
    InvokableSignature signature;
  };

  // This follows the format of the InvokableIndex section.
  std::vector<InvokableIndexEntry> invokableIndexEntries;

  // All of the strings we need to emit the BEF file.
  llvm::SetVector<StringAttr> strings;

  // All of the strings we need to emit for metadata in the BEF file, e.g.
  // attribute names.
  llvm::SetVector<StringAttr> metadataStrings;

  // All of the types we need when emiting metadata in the BEF file.
  std::vector<Type> types;
  DenseMap<Type, unsigned> typeIDs;

  /// This is the offset of each known string in the STRINGS and STRINGS_MD
  /// section.
  DenseMap<StringAttr, BEFOffset> stringOffsets, metadataStringOffsets;

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

void EntityTable::addType(Type type) {
  // Ignore the type if we've seen it before.
  if (!typeIDs.insert({type, types.size()}).second)
    return;
  types.push_back(type);
}

static InvokableSignature getInvokableSignature(Region *region) {
  // Emit information about the type of the invokable.
  auto *block = &region->getBlocks().front();

  // BEF regions produce results with the "bef.output" operation.
  auto output = dyn_cast<BEF::OutputOp>(&block->back());
  size_t numResults = output ? output->getNumOperands() : 0;
  return InvokableSignature{block->getNumArguments(), numResults};
}

LogicalResult EntityTable::addInvokable(Region *region, StringAttr name) {
  // Check to see if we support this region kind.
  if (region->getBlocks().size() != 1) {
    mlir::emitError(region->getLoc())
        << "multi-block regions cannot be emitted to BEF files";
    return failure();
  }

  for (auto argTy : region->front().getArgumentTypes())
    addType(argTy);

  // Remember this invokable.
  addString(name);
  namedInvokableIDs[name] = invokables.size();

  invokableRegionIDs[region] = invokables.size();
  invokables.push_back(
      InvokableEntry(name, getInvokableSignature(region), region));
  return success();
}

/// Return the index of the specified invokable name, returning -1 if the
/// invokable name cannot be found.
ssize_t EntityTable::getInvokableNamed(StringAttr name) const {
  auto iter = namedInvokableIDs.find(name);
  if (iter == namedInvokableIDs.end())
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

/// Walk the IR to build the symbol tables, this returns true on failure.
bool EntityTable::collect(ModuleOp module, bool enableBEFMetadata) {
  std::vector<std::pair<SymbolRefAttr, Location>> symbolRefAttrs;

  auto opWalker = [&](Operation *op) -> WalkResult {
    // Ignore the module itself.
    if (op == module.getOperation())
      return WalkResult::advance();

    // Ignore the OutputOp: it gets special handling, instead of being treated
    // like a primitive.  Similarly, ignore bef.invokable.
    if (isa<BEF::OutputOp>(op))
      return WalkResult::advance();

    // We treat invokables specially, putting them into the symbol table and
    // ignoring their attributes and location.
    if (auto fn = dyn_cast<BEF::InvokableOp>(op)) {
      if (failed(addInvokable(&fn.getBody(), fn.getNameAttr())))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    addLocation(op->getLoc());

    auto *curBlock = op->getBlock();

    // Notice the result types of the ops.  Argument types can only be arguments
    // to the invokables and results of other ops.
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

    llvm::unique_function<void(StringAttr, Attribute, bool)> processAttr =
        [&](StringAttr name, Attribute attr, bool insideDict) {
          // Add attribute names if requested.
          if (enableBEFMetadata && !insideDict)
            addMetadataString(name);

          if (auto fn_attr = attr.dyn_cast<SymbolRefAttr>()) {
            // Keep track of invokable references specially so we can diagnose
            // them.
            symbolRefAttrs.emplace_back(fn_attr, op->getLoc());
          } else {
            // Recurse into a DictionaryAttr only. Perform a post-order
            // traversal so that we can use the offsets from the values inside
            // the dictionary when emitting the dictionary.
            if (auto dictAttr = attr.dyn_cast<DictionaryAttr>())
              for (NamedAttribute namedAttr : dictAttr)
                processAttr(namedAttr.getName(), namedAttr.getValue(), true);

            // If we're handling a dictionary, then insert the name into the
            // attributes list.
            if (insideDict && attributes.insert(name))
              attributeLocs.push_back(op->getLoc());

            // We just keep track of the top level attributes in the
            // "attributes" list: do not recursively process them.  This is
            // because we want to emit arrays of scalars as "flat" attributes
            // where all the values are inline.
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
        processAttr(StringAttr::get(op->getContext()), attr, false);
    } else {
      addPrimitiveOpcode(op->getName().getIdentifier());
      // Keep track of any attributes used by this op.
      for (auto attr : op->getAttrs())
        processAttr(attr.getName(), attr.getValue(), false);
    }

    // Keep add any regions used by this op as BEF invokables.
    for (auto &region : op->getRegions()) {
      if (failed(addInvokable(&region, StringAttr::get(op->getContext()))))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };

  // Walk the whole module finding all the things.
  if (module.walk(opWalker).wasInterrupted())
    return true;

  // If we're successful, check to make sure that all invokables can be
  // resolved.
  for (auto attrAndLoc : symbolRefAttrs) {
    if (getInvokableNamed(attrAndLoc.first.getRootReference()) == -1) {
      mlir::emitError(attrAndLoc.second)
          << "invokable " << attrAndLoc.first << " not defined";
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Attribute Emission
//===----------------------------------------------------------------------===//

/// This is the emitter that builds the attributes section of a BEF.
///
/// This is a bit trickier than it may seem because we don't always recursively
/// process all elements - we leave flat attributes inline, and don't
/// redundantly emit the scalar values.
namespace {
class AttributesEmitter : public BEFEmitter {
public:
  explicit AttributesEmitter(EntityTable &entities) : entities(entities) {}

  BEFOffset emitAttribute(Attribute attr, Location loc);

private:
  FailureOr<BEF::AttributeKind>
  deriveAttributeKind(unsigned bitwidth, bool isFloat, Location loc) const;
  FailureOr<BEF::AttributeKind> getAttributeKind(Attribute attr,
                                                 Location loc) const;
  BEFOffset emitAttributeInternal(Attribute attr, Location loc,
                                  bool emitAsArrayElement = false);
  BEFOffset emitIntegerAttribute(bool emitAsArrayElement, const APInt &value,
                                 bool isFloat, Location loc);
  BEFOffset emitStringAttribute(StringRef value);
  BEFOffset emitDenseElementsAttribute(DenseElementsAttr value, Location loc);
  BEFOffset emitDictionaryAttribute(DictionaryAttr value, Location loc);
  EntityTable &entities;
};
} // end anonymous namespace.

/// Given a valid bitwidth and a boolean that tells us if this attribute is a
/// float or not, construct the correct attribute kind.
FailureOr<BEF::AttributeKind>
AttributesEmitter::deriveAttributeKind(unsigned bitwidth, bool isFloat,
                                       Location loc) const {
  if (!llvm::isPowerOf2_64(bitwidth) || bitwidth > 64)
    return LogicalResult(entities.emitError(loc)
                         << "invalid bitwidth " << bitwidth);

  return BEF::AttributeKind((bitwidth & 0xff) | (isFloat & 0b1));
}

FailureOr<BEF::AttributeKind>
AttributesEmitter::getAttributeKind(Attribute attr, Location loc) const {
  if (auto intAttr = attr.dyn_cast<IntegerAttr>())
    return deriveAttributeKind(intAttr.getValue().getBitWidth(),
                               /*isFloat=*/false, loc);

  if (auto floatAttr = attr.dyn_cast<FloatAttr>())
    return deriveAttributeKind(
        floatAttr.getValue().bitcastToAPInt().getBitWidth(), /*isFloat=*/true,
        loc);

  return LogicalResult(entities.emitError(loc)
                       << "Unknown attribute in getAttributeKind " << attr);
}

BEFOffset AttributesEmitter::emitIntegerAttribute(bool emitAsArrayElement,
                                                  const APInt &value,
                                                  bool isFloat, Location loc) {
  if (!llvm::isPowerOf2_64(value.getBitWidth()) || value.getBitWidth() > 64) {
    entities.emitError(loc)
        << "unsupported integer width " << value.getBitWidth();
    return {0};
  }

  size_t bytes = (value.getBitWidth() + 7) / 8;
  alignTo(bytes);

  // If we are not being emitted as an array element, we should emit our kind.
  if (!emitAsArrayElement) {
    auto kindOr = deriveAttributeKind(value.getBitWidth(), isFloat, loc);
    if (failed(kindOr))
      return {0};
    emitByte((uint8_t)*kindOr);
  }

  auto offset = getOffset();

  uint64_t v = value.getLimitedValue();
  for (unsigned i = 0; i != bytes; ++i) {
    emitByte(static_cast<uint8_t>(v & 255));
    v >>= 8;
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
  emitByte((uint8_t)BEF::AttributeKind::kString);
  auto offset = getOffset();
  emitStringData(value);
  return offset;
}

BEFOffset AttributesEmitter::emitDenseElementsAttribute(DenseElementsAttr value,
                                                        Location loc) {
  Type elementType = value.getElementType();

  // We only support integer and FP element types.
  if (!elementType.isIntOrFloat()) {
    entities.emitError(loc) << "unsupported element type";
    return {0};
  }

  // Get the attribute kind for the elements of the array.
  auto kindOr = deriveAttributeKind(elementType.getIntOrFloatBitWidth(),
                                    !elementType.isIntOrIndex(), loc);
  if (failed(kindOr))
    return {0};

  auto elements = value.getValues<Attribute>();

  BEFOffset sizeStart = getOffset();

  // Emit the number of elements first and then emit all of the attributes
  // consecutively, returning the offset of the first element in the attribute
  // array.
  emitInvertedVBR(elements.size());

  // Emit the byte for the attribute kind just after the inverted VBR for the
  // elements.
  emitByte((uint8_t)BEF::getArrayOf(*kindOr));

  BEFOffset sizeAndKindEnd = getOffset();

  // The attribute array is empty. We just return the offset past the size.
  if (elements.empty())
    return sizeAndKindEnd;

  // Emit the first element of the array, this could insert padding if it needs
  // larger alignment then the current offset.
  auto startOffset = emitAttributeInternal(*elements.begin(), loc, true);
  for (auto elt : llvm::drop_begin(elements))
    (void)emitAttributeInternal(elt, loc, true);

  // If there is a gap between the size and the first attribute value, we
  // move the size to be immediately before the first attribute value to
  // remove the gap.
  if (sizeAndKindEnd.value != startOffset.value) {
    size_t sizeOfArraySizeInBytes = sizeAndKindEnd.value - sizeStart.value;
    memmove(getRawPointer(startOffset) - sizeOfArraySizeInBytes,
            getRawPointer(sizeStart), sizeOfArraySizeInBytes);
    // Set the unused bytes to the alignment filler.
    memset(getRawPointer(sizeStart), BEF::kAlignmentByte,
           startOffset.value - sizeAndKindEnd.value);
  }

  return startOffset;
}

BEFOffset AttributesEmitter::emitDictionaryAttribute(DictionaryAttr value,
                                                     Location loc) {
  if (value.empty()) {
    emitInvertedVBR(value.size());
    emitByte((uint8_t)BEF::AttributeKind::kDictionary);
    return getOffset();
  }

  // Emit the size of the dict and the attribute kind.
  emitInvertedVBR(value.size());
  emitByte((uint8_t)BEF::AttributeKind::kDictionary);

  // This is the offset into the dict we want to provide - essentially a table
  // of name:offset pairs that we can use when parsing.
  BEFOffset nameAndOffsetStart = getOffset();

  for (NamedAttribute namedAttr : value) {
    // Emit the offsets to the name and data as an offset relative to the
    // dictionary offset. They are therefore implicitly negative!
    size_t relativeNameOffset =
        nameAndOffsetStart.value -
        entities.getAttributeOffset(namedAttr.getName()).value;

    size_t relativeAttrOffset =
        nameAndOffsetStart.value -
        entities.getAttributeOffset(namedAttr.getValue()).value;

    emitOffset(BEFOffset{relativeNameOffset});
    emitOffset(BEFOffset{relativeAttrOffset});
  }

  return nameAndOffsetStart;
}

BEFOffset AttributesEmitter::emitAttributeInternal(Attribute attr, Location loc,
                                                   bool emitAsArrayElement) {
  if (auto intAttr = attr.dyn_cast<IntegerAttr>())
    return emitIntegerAttribute(emitAsArrayElement, intAttr.getValue(), false,
                                loc);

  if (auto fpAttr = attr.dyn_cast<FloatAttr>())
    return emitIntegerAttribute(emitAsArrayElement,
                                fpAttr.getValue().bitcastToAPInt(), true, loc);

  // Non-primitive (i.e. non-int or non-float) elements cannot be emitted as
  // arrays, so if this function proceeds past here there has been an error.
  if (emitAsArrayElement) {
    entities.emitError(loc)
        << "cannot have aggregates of non-primitive attrs " << attr;
    return {0};
  }

  if (auto string = attr.dyn_cast<StringAttr>())
    return emitStringAttribute(string.getValue());

  if (auto denseElements = attr.dyn_cast<DenseElementsAttr>())
    return emitDenseElementsAttribute(denseElements, loc);

  if (auto dict = attr.dyn_cast<DictionaryAttr>())
    return emitDictionaryAttribute(dict, loc);

  entities.emitError(loc) << "unsupported attribute " << attr;
  return {0};
}

BEFOffset AttributesEmitter::emitAttribute(Attribute attr, Location loc) {
  assert(!entities.attributeOffsets.count(attr) && "Attr already emitted?");
  auto offset = emitAttributeInternal(attr, loc);
  entities.attributeOffsets[attr] = offset;
  return offset;
}

//===----------------------------------------------------------------------===//
// InvokableBodyEmitter
//===----------------------------------------------------------------------===//

namespace {
// This is the emitter that builds the invokable body of a BEF.
class InvokableBodyEmitter : public BEFEmitter {
public:
  InvokableBodyEmitter(EntityTable &entities, BEFEmitter *attributeNamesSection,
                       BEFEmitter *mlirTypesSection)
      : entities(entities), attributeNamesSection(attributeNamesSection),
        mlirTypesSection(mlirTypesSection) {}

  void emitInvokable(Region *region);

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

  // If optional metadata is enabled, this section collects information about
  // the names that correspond to primitive attributes.
  BEFEmitter *attributeNamesSection;
  // If optional metadata is enabled, this section collects information about
  // the MLIR type for each argument and primitive result.
  BEFEmitter *mlirTypesSection;
};
} // end anonymous namespace.

void InvokableBodyEmitter::emitInvokable(Region *region) {
  assert(region->getBlocks().size() == 1 && "should have a single block");
  auto &block = region->getBlocks().front();

  // Get a dense numbering of primitives.
  unsigned numPrimitives = 0;

  // If the invokable has arguments, we emit a pseudo-op that provides the
  // argument values.
  if (block.getNumArguments() != 0)
    ++numPrimitives;

  for (auto &op : block.getOperations()) {
    if (!isa<BEF::OutputOp>(op))
      primitiveIndices[&op] = numPrimitives++;
  }

  // Emit a count of primitives.
  emitVBR(numPrimitives);

  // Emit the register table.
  emitRegisterTable(&block);

  // Next is an offset for each primitive (from the start of the primitive
  // list) then each primitive is emitted in turn.
  BEFEmitter primitiveList;

  // If we have arguments, emit a pseudo op (with no opcode) that produces the
  // registers on entry to the invokable.
  if (block.getNumArguments() != 0) {
    // Offset of the primitive in the list.
    emitPrimitiveOffsetNumber(primitiveList.getOffset());
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
      isNonStrict = isa<BEF::NonStrictOp>(op);

    // Offset of the primitive in the list.
    emitPrimitiveOffsetNumber(primitiveList.getOffset());
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
  alignTo(BEF::kPrimitiveEntryAlignment);
  emitEmitter(primitiveList);

  primitiveIndices.clear();
}

void InvokableBodyEmitter::emitRegisterTable(Block *block) {
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

unsigned InvokableBodyEmitter::getNumNonOutputUsers(Value result) const {
  int numUsers = 0;
  for (auto *user : result.getUsers()) {
    // Ignore the 'bef.output' op, it gets special handling.
    if (!isa<BEF::OutputOp>(user))
      ++numUsers;
  }
  return numUsers;
}

void InvokableBodyEmitter::emitPrimitiveResultUsers(
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

void InvokableBodyEmitter::emitArgumentsPseudoOp(
    Block *block, BEFEmitter &primitiveList) const {
  // This primitive starts with a dummy code and a dummy location. And this
  // primitive only has results and UsedBys in its body.
  primitiveList.emitInt4(0);                        // code
  primitiveList.emitInt4(0);                        // location
  primitiveList.emitInt2(0);                        // # operands
  primitiveList.emitInt2(0);                        // # attributes
  primitiveList.emitInt2(0);                        // # invokables
  primitiveList.emitInt2(block->getNumArguments()); // # results

  for (auto arg : block->getArguments())
    primitiveList.emitInt4(getNumNonOutputUsers(arg));

  // Only emit a single register number for the decoder for consistency.  It
  // should always be zero since this is the first result of the first
  // primitive.
  assert(getRegisterNumber(block->getArguments().front()) == 0);
  primitiveList.emitInt4(0);

  for (auto arg : block->getArguments()) {
    emitPrimitiveResultUsers(arg, primitiveList);
    // If extra metadata is enabled, remember the type for each result.
    if (mlirTypesSection)
      mlirTypesSection->emitVBR(entities.getTypeIndex(arg.getType()));
  }
}

void InvokableBodyEmitter::emitPrimitive(Operation *op,
                                         BEFEmitter &primitiveList) const {
  // Collect the regions and attributes to emit.
  SmallVector<std::pair<uint32_t, BEFOffset>> invokablesToEmit;
  SmallVector<uint32_t> attributesToEmit;
  auto processAttr = [&](StringAttr name, Attribute value) {
    if (auto fnAttr = value.dyn_cast<FlatSymbolRefAttr>()) {
      // bef.invokable references are emitted as invokable.
      BEFOffset nameIdx = {0};
      if (attributeNamesSection)
        nameIdx = entities.getMetadataStringOffset(name);
      invokablesToEmit.push_back(
          {entities.getInvokableNamed(fnAttr.getAttr()), nameIdx});
    } else {
      attributesToEmit.push_back(entities.getAttributeOffset(value).value);
      if (attributeNamesSection)
        attributeNamesSection->emitOffset(
            entities.getMetadataStringOffset(name));
    }
  };

  // bef.opaque operands store attributes in an explicit way since we don't know
  // the attribute names.
  if (auto opaque = dyn_cast<BEF::OpaqueOp>(op)) {
    // All encoded attributes are stored in a single array.
    for (auto attr : opaque.attrs())
      processAttr(StringAttr::get(op->getContext()), attr);
  } else {
    for (auto attrNamePair : op->getAttrs())
      processAttr(attrNamePair.getName(), attrNamePair.getValue());
  }

  // Regions on the op are also considered to be BEF invokables.
  for (auto &region : op->getRegions())
    invokablesToEmit.push_back({entities.getInvokableID(region), BEFOffset{0}});

  // Ensure this primitive isn't too big to encode.
  if (op->getNumOperands() > 65535 || attributesToEmit.size() > 65535 ||
      invokablesToEmit.size() > 65535 || op->getNumResults() > 65535)
    entities.emitError(op->getLoc()) << "operation too large to encode in BEF";

  // Emit the PRIMITIVE_HEADER records.
  primitiveList.emitInt4(entities.getPrimitiveID(op));
  primitiveList.emitInt4(entities.getLocationOffset(op->getLoc()).value);
  primitiveList.emitInt2(op->getNumOperands());
  primitiveList.emitInt2(attributesToEmit.size());
  primitiveList.emitInt2(invokablesToEmit.size());
  primitiveList.emitInt2(op->getNumResults());

  // Emit PRIMITIVE_RESULT_TABLE, results with the primitives that use them.
  for (auto result : op->getResults())
    primitiveList.emitInt4(getNumNonOutputUsers(result));

  // Emit the PRIMITIVE_BODY contents.
  for (auto operand : op->getOperands())
    primitiveList.emitInt4(getRegisterNumber(operand));
  for (auto attrOffset : attributesToEmit)
    primitiveList.emitInt4(attrOffset);
  for (auto fnOffsetAndName : invokablesToEmit) {
    primitiveList.emitInt4(fnOffsetAndName.first);
    if (attributeNamesSection)
      attributeNamesSection->emitOffset(fnOffsetAndName.second);
  }

  // Emit a single result number of we have one or more results.  They are
  // numbered consecutively, so we don't need to emit them all.
  if (!op->getResults().empty())
    primitiveList.emitInt4(getRegisterNumber(op->getResults()[0]));

  // Then results with the primitives that use them.
  for (auto result : op->getResults()) {
    emitPrimitiveResultUsers(result, primitiveList);

    // If extra metadata is enabled, remember the type for each result.
    if (mlirTypesSection)
      mlirTypesSection->emitVBR(entities.getTypeIndex(result.getType()));
  }
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
  bool collectEntities(bool enableBEFMetadata) {
    return entities.collect(module, enableBEFMetadata);
  }
  void emitLocationInfo(BEFEmitter &locationFilenames,
                        BEFEmitter &locationPositions);
  void emitStrings();
  void emitStringsMD(BEFEmitter &stringsMDSection);
  void emitTypesMD(BEFEmitter &typeStringsMDSection);
  void emitAttributes(BEFEmitter &attrMDSection);
  void emitPrimitiveNames();
  void emitInvokables(BEFEmitter *attributeNamesSection,
                      BEFEmitter *mlirTypesSection);
  void emitInvokableIndex();

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
    stringSection.emitNulTerminatedString(string ? string.getValue() : "");
  }

  emitSection(BEF::SectionID::kStrings, stringSection);
}

void BEFModuleEmitter::emitStringsMD(BEFEmitter &stringsMDSection) {
  // Emit the strings to the metadata string section and remember their offsets.
  for (StringAttr string : entities.metadataStrings) {
    entities.metadataStringOffsets[string] = stringsMDSection.getOffset();
    stringsMDSection.emitNulTerminatedString(string);
  }
}

void BEFModuleEmitter::emitTypesMD(BEFEmitter &typeStringsMDSection) {
  // Emit the type strings to the type string section.
  SmallVector<char, 64> result;
  for (Type type : entities.types) {
    result.clear();
    llvm::raw_svector_ostream os(result);
    type.print(os);
    typeStringsMDSection.emitNulTerminatedString(os.str());
  }
}

/// Emit the ATTRIBUTES and the ATTRIBUTE_INDEX_MD sections.
void BEFModuleEmitter::emitAttributes(BEFEmitter &attrMDSection) {
  // Emit attributes and record them in EntityIndex. Nested array attributes
  // will be traversed recursively and their elements will be emitted and
  // recorded before the top level offsets array is emitted.
  AttributesEmitter attributesSection(entities);
  for (auto attrAndLoc :
       llvm::zip(entities.attributes, entities.attributeLocs)) {
    auto offset = attributesSection.emitAttribute(std::get<0>(attrAndLoc),
                                                  std::get<1>(attrAndLoc));
    // Emit metadata for each attribute we put out so BEF2MLIR can interpret the
    // data correctly.
    attrMDSection.emitOffset(offset);
  }

  emitSection(BEF::SectionID::kAttributes, attributesSection);
}

void BEFModuleEmitter::emitPrimitiveNames() {
  // The primitives are already in a stable order, so just emit them in the
  // order they were found.
  BEFEmitter primitivesNamesSection;
  primitivesNamesSection.emitVBR(entities.primitives.size());
  for (auto op : entities.primitives)
    primitivesNamesSection.emitOffset(entities.getStringOffset(op));

  emitSection(BEF::SectionID::kPrimitiveNames, primitivesNamesSection);
}

/// Emit the INVOKABLES_BODIES section.  If optional metadata is enabled,
/// the attributeNamesSection/mlirTypesSection will be non-null and should be
/// filled in.
void BEFModuleEmitter::emitInvokables(BEFEmitter *attributeNamesSection,
                                      BEFEmitter *mlirTypesSection) {
  InvokableBodyEmitter invokablesSection(entities, attributeNamesSection,
                                         mlirTypesSection);

  for (auto invokable : entities.invokables) {
    // Remember that we emitted this region to this offset.
    entities.addInvokableOffset(invokable.name, invokablesSection.getOffset(),
                                invokable.signature);
    invokablesSection.emitInvokable(invokable.region);
  }

  emitSection(BEF::SectionID::kInvokablesBodies, invokablesSection);
}

void BEFModuleEmitter::emitInvokableIndex() {
  BEFEmitter invokableIndexSection;
  invokableIndexSection.emitVBR(entities.invokableIndexEntries.size());
  for (const auto &entry : entities.invokableIndexEntries) {
    invokableIndexSection.emitOffset(entry.invokableOffset);
    invokableIndexSection.emitOffset(entry.nameOffset);
    invokableIndexSection.emitVBR(entry.signature.numArguments);
    invokableIndexSection.emitVBR(entry.signature.numResults);
  }

  emitSection(BEF::SectionID::kInvokablesIndex, invokableIndexSection);
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
                                           bool enableBEFMetadata) {
  BEFModuleEmitter emitter(module);

  // Build the entities table.
  if (emitter.collectEntities(enableBEFMetadata))
    return {};

  // Sections describing location information.  We collect this early, but emit
  // it near the end of the file since it isn't typically needed by the
  // BEFExecutor.
  BEFEmitter locationFilenames, locationPositions;

  // If optional section emission is enabled, these collect the data to emit
  // into the respective sections.
  BEFEmitter attrIndexMDSection;
  BEFEmitter stringsMDSection, typeStringsMDSection, attrNamesSection,
      mlirTypesSection;

  // Emit magic numbers and format version.
  emitter.emitBytes({BEF::kMagic1, BEF::kMagic2 + BEF::kVersion0});
  emitter.emitLocationInfo(locationFilenames, locationPositions);
  emitter.emitStrings();
  if (enableBEFMetadata) {
    emitter.emitStringsMD(stringsMDSection);
    emitter.emitTypesMD(typeStringsMDSection);
  }

  // Emitting the attributes can fail if there's an unsupported attribute.
  emitter.emitAttributes(attrIndexMDSection);

  emitter.emitPrimitiveNames();
  emitter.emitInvokables(enableBEFMetadata ? &attrNamesSection : nullptr,
                         enableBEFMetadata ? &mlirTypesSection : nullptr);
  emitter.emitInvokableIndex();

  // Now that we've emitted all of the critical sections used in the common case
  // of the BEF emitter, emit metadata sections.
  emitter.emitSection(BEF::SectionID::kLocationFilenames, locationFilenames);
  emitter.emitSection(BEF::SectionID::kLocationPositions, locationPositions);

  // Unconditionally emit metadata that describes the attribute section, this
  // is the minimal metadata needed by BEF2MLIR.
  emitter.emitSection(BEF::SectionID::kAttributeIndexMD, attrIndexMDSection);

  // Emit all the optional sections at the end if they're requested.
  if (enableBEFMetadata) {
    emitter.emitSection(BEF::SectionID::kMLIRTypeMD, mlirTypesSection);
    emitter.emitSection(BEF::SectionID::kStringsMD, stringsMDSection);
    emitter.emitSection(BEF::SectionID::kAttributeNamesMD, attrNamesSection);
    emitter.emitSection(BEF::SectionID::kMLIRTypeStringsMD,
                        typeStringsMDSection);
  }

  // If emission diagnosed an error along the way and emitted a diagnostic,
  // then we should propagate the failure up.
  if (emitter.hadError())
    return {};

  return emitter.takeResult();
}
