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
#include "mlir/IR/BuiltinOps.h"
using namespace M;

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
// BEFModuleEmitter
//===----------------------------------------------------------------------===//

namespace {
/// This emitter generates the top level structure of the BEF file.
class BEFModuleEmitter : public BEFEmitter {
public:
  explicit BEFModuleEmitter(mlir::ModuleOp module) : module(module) {}

  bool collectEntities(bool disableOptionalSections) {
    // TODO.
    return false;
  }

private:
  mlir::ModuleOp module;
};
} // end anonymous namespace

/// This function converts the specified MLIR module containing a executor
/// compatible program to the BinaryExecutableFormat (BEF) format, which is the
/// binary format that the executor takes.  When `disableOptionalSections` is
/// true, this omits metadata sections required to reverse a BEF file back into
/// a .mlir file.
///
/// On error, this emits the error message through the MLIR error handler, and
/// returns an empty vector.
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
