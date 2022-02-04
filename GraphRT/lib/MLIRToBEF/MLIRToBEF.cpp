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

static FunctionType getRegionFunctionType(Region *region) {
  // Emit information about the type of the function.
  auto *block = &region->getBlocks().front();

  // Arguments.
  SmallVector<Type, 4> inputs(block->getArgumentTypes());

  // Results: MLIR Regions don't have an easy way to identify results in
  // regions, so we just hard code the "bef.return" instruction.
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

  size_t size() const { return result.size(); }

  void emitByte(uint8_t byte) { result.push_back(byte); }

  void emitBytes(ArrayRef<uint8_t> bytes) {
    result.insert(result.end(), bytes.begin(), bytes.end());
  }

  void alignTo(unsigned alignment) {
    // Alignment of 0 and 1 is a noop.
    if (alignment < 2)
      return;

    assert(llvm::isPowerOf2_32(alignment));

    // We need attributes to have proper alignment in the file, so figure out
    // whether we need padding before this to make sure it ends up at the right
    // address.
    size_t curOffset = size();
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

  // Many parts of the emitter logic includes forward references into stuff
  // that hasn't been emitted and has variable size.  This is handled by making
  // nested emitters.  This helper function emits the subpieces once they are
  // constructed, ensuring that alignment requirements of the nested emitter
  // are maintained correctly.
  void EmitEmitter(const BEFEmitter &emitter) {
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

// Our fundamental unit is a bytestream, but we want to be able to emit large
// values as well.  We use a VBR encoding, where the high bit set indicates
// that this is only a portion of the value.
void BEFEmitter::emitVBRImpl(size_t value, bool isHighPart) {
  if ((value >> 7) != 0)
    emitVBRImpl(value >> 7, /*isHighPart=*/true);

  emitByte(static_cast<uint8_t>((value & 127) | (isHighPart ? 128 : 0)));
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
  ssize_t getFunctionNamed(StringRef name) const;

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
    auto it = primitiveIDs.find(primitive->getName().getStringRef());
    assert(it != primitiveIDs.end() && "Unknown primitive");
    return it->second;
  }

private:
  void addString(StringRef string) { strings.insert(string); }
  void addType(Type type);
  LogicalResult addBEFFunction(Region *region, StringRef name);
  void addPrimitive(Operation *primitive);
  void addLocation(Location loc);
  void addAttributeType(Attribute attr);

private:
  // Uniquing set of attributes we need to emit, kept in order so we always
  // produce a determinstic output file.
  llvm::SetVector<Attribute> attributes;

  // Uniquing set of the primitives that we need to emit.
  std::vector<StringRef> primitives;
  llvm::StringMap<unsigned> primitiveIDs;

  /// BEFFunctionEntry keeps track of each of the top level `func`'s and the
  /// regions attached to control flow statements.  These all turn into BEF
  /// functions.
  struct BEFFunctionEntry {
    BEFFunctionEntry(StringRef name, FunctionType type, Region *region)
        : name(name), type(type), region(region) {}

    StringRef name;
    FunctionType type;
    Region *region;
  };

  // List of functions that we need to emit, along with a name if they came
  // from a top level function.
  std::vector<BEFFunctionEntry> functions;
  DenseMap<Region *, unsigned> regionFunctionIDs;
  llvm::StringMap<unsigned> namedFunctionIDs;

  // Types we've seen so far.
  std::vector<Type> types;
  DenseMap<Type, unsigned> typeIDs;

  // All of the strings we need to emit the BEF file, an unordered
  // collection that we sort before emitting.
  llvm::StringSet<> strings;

  // This is all of the filenames referred to by locations in the file.
  SmallVector<StringRef, 4> locationFilenames;
  llvm::StringMap<unsigned> locationFilenamesIndices;

  // This is a list of locations within the file, the first element of the
  // tuple is a index into locationFilenames, the second and third are line/col
  // information.
  typedef std::tuple<unsigned, unsigned, unsigned> LocationTuple;
  llvm::SetVector<LocationTuple> locationPositions;
};

} // namespace

// Add a type to our table, checking it by pointer to reduce string
// conversions.
void EntityTable::addType(Type type) {
  // Ignore the type if we've seen it before.
  if (!typeIDs.insert({type, types.size()}).second)
    return;
  types.push_back(type);

  // If it is new, remember the type name as a string.
  // FIXME: Why is this needed?
  SmallVector<char, 64> resultStr;
  llvm::raw_svector_ostream os(resultStr);
  type.print(os);
  addString(os.str());
}

LogicalResult EntityTable::addBEFFunction(Region *region, StringRef name) {
  // Check to see if we support this region kind.
  if (region->getBlocks().size() != 1) {
    mlir::emitError(region->getLoc())
        << "multi-block regions cannot be emitted to BEF files";
    return failure();
  }

  for (auto argTy : region->front().getArgumentTypes())
    addType(argTy);

  // Remember this function.
  addString(name);
  regionFunctionIDs[region] = functions.size();
  namedFunctionIDs[name] = functions.size();
  functions.push_back(
      BEFFunctionEntry(name, getRegionFunctionType(region), region));
  return success();
}

// Return the index of the specified function name, returning -1 if the
// function name cannot be found.
ssize_t EntityTable::getFunctionNamed(StringRef name) const {
  auto iter = namedFunctionIDs.find(name);
  if (iter == namedFunctionIDs.end())
    return -1;
  return iter->second;
}

void EntityTable::addPrimitive(Operation *primitive) {
  StringRef name = primitive->getName().getStringRef();
  // Remember the primitive.
  if (!primitiveIDs.insert({name, primitives.size()}).second)
    return;
  primitives.push_back(name);

  // If we haven't seen it already, add it to the string table.
  addString(name);
}

void EntityTable::addLocation(Location loc) {
  StringRef filename = "";
  unsigned line = 0, col = 0;
  // TODO: mainline TFRT has a richer location model we could adopt.
  if (auto fileLineCol = loc.dyn_cast<FileLineColLoc>()) {
    filename = fileLineCol.getFilename();
    line = fileLineCol.getLine();
    col = fileLineCol.getColumn();
  }

  auto nextFilenameIndex = locationFilenames.size();
  auto it =
      locationFilenamesIndices.insert({filename, nextFilenameIndex}).first;
  if (it->second == nextFilenameIndex)
    locationFilenames.push_back(filename);

  locationPositions.insert(LocationTuple{it->second, line, col});
}

void EntityTable::addAttributeType(Attribute attr) {
  if (auto int_attr = attr.dyn_cast<IntegerAttr>())
    addType(int_attr.getType());

  if (auto float_attr = attr.dyn_cast<FloatAttr>())
    addType(float_attr.getType());

  if (auto arr_attr = attr.dyn_cast<ArrayAttr>()) {
    for (auto attr : arr_attr.getValue()) {
      addAttributeType(attr);
    }
  }
}

/// Walk the IR to build the symbol tables, this returns true on failure.
bool EntityTable::collect(ModuleOp module, bool disableOptionalSections) {
  std::vector<std::pair<SymbolRefAttr, Location>> symbolRefAttrs;

  auto opWalker = [&](Operation *op) -> WalkResult {
    // Ignore the module itself.
    if (op == module.getOperation())
      return WalkResult::advance();

    // The return op gets special handling, ensure it is at the end of its
    // enclosing block.
    if (isa<BEF::OutputOp>(op)) {
      if (&op->getBlock()->back() != op) {
        op->emitError() << "return op must be at the end of its block";
        return WalkResult::interrupt();
      }
      // Ignore it, return gets special handling.
      return WalkResult::advance();
    }

    addLocation(op->getLoc());

    auto *curBlock = op->getBlock();

    // Notice the result and argument types of the ops.
    for (auto result : op->getResults())
      addType(result.getType());

    for (auto operand : op->getOperands()) {
      // Verify that the operand is defined inside the current region.  We
      // don't support references to outer regions.
      if (operand.getParentBlock() != curBlock) {
        op->emitError()
            << "BEF executor only supports references to primitives within"
            << " the current region";
        return WalkResult::interrupt();
      }
    }

    // We treat functions specially, putting them into the symbol table and
    // ignoring their attributes.
    if (auto fn = dyn_cast<BEF::FuncOp>(op)) {
      // Verify that all functions end with a return to catch a common
      // error.
      auto &lastOp = fn.getBlocks().front().back();
      if (!isa<BEF::OutputOp>(&lastOp)) {
        lastOp.emitError() << "all functions need to have a bef.return";
        return WalkResult::interrupt();
      }

      if (failed(addBEFFunction(&fn.getBody(), fn.getName())))
        return WalkResult::interrupt();
    } else {
      addPrimitive(op);

      // Keep track of any attributes used by this op.
      for (auto attr : op->getAttrs()) {
        // If this is a special attribute, ignore it.
        if (classifyAttribute(attr.getName()) != WellKnownAttrs::Unknown)
          continue;

        // Check to make sure that this is a supported attribute, if not,
        // reject it.
        if (!isSupportedAttribute(attr.getValue())) {
          op->emitError() << "BEF files cannot encode the '" << attr.getName()
                          << "' attribute";
          return WalkResult::interrupt();
        }

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

          // We ignore the name of attributes, they just get passed as
          // arguments.
          attributes.insert(attr.getValue());
        }
      }

      // Keep add any regions used by this op as BEF functions.
      for (auto &region : op->getRegions()) {
        if (failed(addBEFFunction(&region, "")))
          return WalkResult::interrupt();
      }
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

private:
  ModuleOp module;
  EntityTable entities;
};
} // end anonymous namespace

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

  // Emit magic numbers and format version.
  emitter.emitBytes({BEF::MAGIC1, BEF::MAGIC2 + BEF::VERSION0});

  return emitter.takeResult();
}
