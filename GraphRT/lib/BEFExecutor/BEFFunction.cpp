//===- BEFFunction.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements logic required to execute a function.  This is the core
// execution logic for the BEF executor runtime.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/Function.h"

#include "BEFFileImpl.h"
#include "GraphRT/BEF/BEFReader.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/RCRef.h"
#include "llvm/Support/TrailingObjects.h"
using namespace M;

//===----------------------------------------------------------------------===//
// Function implementation
//===----------------------------------------------------------------------===//

// NOTE: This could be moved out to somewhere non-BEF specific if there are
// other kinds of functions someday.
void Function::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// RegisterInfo and PrimitiveInfo
//===----------------------------------------------------------------------===//

namespace {
struct RegisterInfo {
  /// This is the number of uses of the register in the function.  The
  /// AsyncValue may be deallocated when this number of uses are complete.
  const size_t userCount;

  /// This is a pointer to the AsyncValue for a register as the function
  /// executes.  It is generally filled in when the primitive that generates it
  /// executes.
  std::atomic<AsyncValue *> value{nullptr};

  explicit RegisterInfo(unsigned userCount) : userCount(userCount) {}
};
} // end anonymous namespace

namespace {
/// One entry of this record is kept for each primitive in a BEFFunction.  This
/// allows us to efficiently map from a primitive index to a primitive offset.
struct PrimitiveInfo {
  std::atomic<int> numOperandsNotReady;
  unsigned offset;

  /// We initialize the ready list to "numOperands + 1" so we can drop the
  /// last count in the executor constructor.
  PrimitiveInfo(unsigned offset, unsigned numOperandsBeforeExecuting)
      : numOperandsNotReady(numOperandsBeforeExecuting + 1), offset(offset) {}
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BEFFunctionDecoder
//===----------------------------------------------------------------------===//

namespace {
/// This class is a helper used to handle initial function decoding before the
/// BEFExecutor can be allocated.
struct BEFFunctionDecoder {
public:
  BEFFunctionDecoder(const BEFFunction &fn)
      : fn(fn), funcReader(fn.owner->befFunctions, [&](Error message) {
          befFormatError = std::move(message);
        }) {}

  /// Decoding the function record.
  bool decodeHeader() {
    return funcReader.dropBytes(fn.functionOffset) ||
           funcReader.parseVBR(numPrimitives) ||
           funcReader.parseVBR(numRegisters);
  }

public:
  const BEFFunction &fn;
  Optional<Error> befFormatError;
  BEFReader funcReader;

  /// These are the fields we decode.
  size_t numPrimitives = 0, numRegisters = 0;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BEFExecutor implementation
//===----------------------------------------------------------------------===//

namespace {
/// A BEFExecutor runs a BEF function containing a stream of asynchronous
/// primitives.  Multiple executors can be active at one time, even for the same
/// function on disk.
///
/// The layout of this class is the BEFExecutor itself, following by a trailing
/// array of "N" RegisterInfo records, following by "M" PrimitiveInfo records,
/// where N and M are encoded into the BEF function header.
class BEFExecutor final
    : public ReferenceCounted<BEFExecutor>,
      public llvm::TrailingObjects<BEFExecutor, RegisterInfo, PrimitiveInfo> {
public:
  BEFExecutor(BEFFileImpl *file, BEFFunctionDecoder &decoder)
      : befFile(copyRCRef(file)), numRegisters(decoder.numRegisters),
        numPrimitives(decoder.numPrimitives) {}
  ~BEFExecutor() {
    // NOTE: we don't bother destructing the ResultInfo/PrimitiveInfo records.
  }

  /// Decode the register info and primitive info tables.  This returns true on
  /// failure.
  bool decodeTables(BEFFunctionDecoder &decoder);

  /// Execute the function once the tables are decoded.
  void execute(const BEFFunction &fn, ArrayRef<AsyncValue *> arguments,
               MutableArrayRef<RCRef<AsyncValue>> results);

  /// When the last reference to the BEFExecutor is dropped, we deallocate
  /// ourself.  The memory for this class is managed by the current Runtime.
  void destroy() const {
    Runtime &runtime = befFile->runtime;
    this->~BEFExecutor();
    auto sizeInBytes = totalSizeToAlloc<RegisterInfo, PrimitiveInfo>(
        numRegisters, numPrimitives);
    runtime.deallocateBytes(const_cast<BEFExecutor *>(this), sizeInBytes);
  }

  /// Provide access to the inline-allocated RegisterInfo table.
  MutableArrayRef<RegisterInfo> getRegisterInfoTable() {
    return {getTrailingObjects<RegisterInfo>(), numRegisters};
  }
  /// Provide access to the inline-allocated PrimitiveInfo table.
  MutableArrayRef<PrimitiveInfo> getPrimitiveInfoTable() {
    return {getTrailingObjects<PrimitiveInfo>(), numPrimitives};
  }
  /// This method is used by TrailingObjects to calculate offsets.
  size_t numTrailingObjects(OverloadToken<RegisterInfo>) const {
    return numRegisters;
  }

private:
  /// Make sure the BEF file doesn't get deallocated while we're
  /// asynchronously running stuff.
  RCRef<BEFFileImpl> befFile;

  const unsigned numRegisters, numPrimitives;

  /// These are the indices of the result registers for the function.
  SmallVector<unsigned, 4> resultRegs;

  // All of the primitive entry records for this function.
  ArrayRef<uint32_t> funcBody;
};
} // end anonymous namespace

/// Decode the register info and primitive info tables.  This returns true on
/// failure.
bool BEFExecutor::decodeTables(BEFFunctionDecoder &decoder) {
  auto &funcReader = decoder.funcReader;

  // Parse the register table.
  MutableArrayRef<RegisterInfo> regTable = getRegisterInfoTable();
  for (size_t i = 0, e = numRegisters; i != e; ++i) {
    size_t numUses;
    if (funcReader.parseVBR(numUses))
      return true;
    new (&regTable[i]) RegisterInfo(numUses);
  }

  // Parse the primitive table.
  MutableArrayRef<PrimitiveInfo> primTable = getPrimitiveInfoTable();
  for (size_t i = 0, e = numPrimitives; i != e; ++i) {
    size_t offset, numOperandsBeforeExecuting;
    if (funcReader.parseVBR(offset) ||
        funcReader.parseVBR(numOperandsBeforeExecuting))
      return true;
    new (&primTable[i]) PrimitiveInfo(offset, numOperandsBeforeExecuting);
  }

  // Parse the ResultRegs
  auto numResults = decoder.fn.getNumResults();
  resultRegs.reserve(numResults);
  for (size_t i = 0; i != numResults; ++i) {
    size_t registerIndex;
    if (funcReader.parseVBR(registerIndex))
      return true;
    resultRegs.push_back(registerIndex);
  }

  // Align to where the function body is.
  if (funcReader.alignTo(BEF::PRIMITIVE_ENTRY_ALIGNMENT))
    return true;

  funcBody = llvm::makeArrayRef((const uint32_t *)funcReader.data(),
                                funcReader.getContents().size() / 4);
  return false;
}

 /// Execute the function once the tables are decoded.
 void BEFExecutor::execute(const BEFFunction &fn,
                          ArrayRef<AsyncValue *> arguments,
                          MutableArrayRef<RCRef<AsyncValue>> results) {
  // TODO: Do we need `fn`?
  printf("execute %s\n", fn.getName().data());

  // TODO: Implement this.
}

//===----------------------------------------------------------------------===//
// BEFFunction
//===----------------------------------------------------------------------===//

/// Add a new reference to this object.  The lifetime of BEF functions are
/// completely tied to BEFFileImpl, so just extend its reference count.
void BEFFunction::addRef() const { owner->addRef(); }

/// Drop a reference to this object, potentially deallocating it.
void BEFFunction::dropRef() const { owner->dropRef(); }

/// Execute this function, passing the specified arguments. This returns one
/// AsyncValue for each result.
void BEFFunction::execute(ArrayRef<AsyncValue *> arguments,
                          MutableArrayRef<RCRef<AsyncValue>> results) const {
  assert(arguments.size() == getNumArguments() &&
         "incorrect number of argument to a BEF function");
  assert(results.size() == getNumResults() &&
         "incorrect number of results for a BEF function");

  BEFFunctionDecoder decoder(*this);
  bool headerParseFailure = decoder.decodeHeader();

  // We allocate a single slab of memory for the BEFExecutor object itself,
  // along with the register and primitive entries.  This give good locality as
  // well as reducing #allocations.
  auto sizeInBytes = BEFExecutor::totalSizeToAlloc<RegisterInfo, PrimitiveInfo>(
      decoder.numRegisters, decoder.numPrimitives);
  void *rawPtr =
      owner->runtime.allocateBytes(sizeInBytes, alignof(BEFExecutor));
  auto executor = takeRCRef(new (rawPtr) BEFExecutor(owner, decoder));

  // Parse the rest of the tables now that we allocated memory.
  if (headerParseFailure || executor->decodeTables(decoder)) {
    // TODO: Fill in error async value results.
    assert(0 && "FIXME: Errors not implemented yet");
    abort();
  }

  // Finally, start execution of any primitives that are ready to go.
  executor->execute(*this, arguments, results);

  // The RCRef drops the reference to `executor` since it is running.
}
