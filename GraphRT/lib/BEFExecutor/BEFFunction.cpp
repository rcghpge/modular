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
#include "LLCL/Runtime/AsyncValue.h"
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
// RegisterInfo
//===----------------------------------------------------------------------===//

namespace {
struct RegisterInfo {
  /// This is the number of uses of the register in the function.  The
  /// AsyncValue may be deallocated when this number of uses are complete.  This
  /// field allows us to allocate the register with the correct reference count
  /// up front.
  const size_t userCount;

  explicit RegisterInfo(unsigned userCount) : userCount(userCount) {}

  /// Return the AsyncValue for this register, this is used when the client
  /// knows it already exists or is prepared to deal with a missing value.
  AsyncValue *getValue() { return value.load(std::memory_order_acquire); }

  /// Return the AsyncValue for this register, creating it on demand as
  /// necessary.
  AsyncValue *getOrCreateIndirectValue();

private:
  /// This is a pointer to the AsyncValue for a register as the function
  /// executes.  It is generally filled in when the primitive that generates it
  /// executes.
  std::atomic<AsyncValue *> value{nullptr};
};
} // end anonymous namespace

/// Return the AsyncValue for this register, creating it on demand as
/// necessary.
AsyncValue *RegisterInfo::getOrCreateIndirectValue() {
  // In the normal case, just load the pointer and return it.
  AsyncValue *value = getValue();
  if (value)
    return value;

  // If it doesn't exist, we create an IndirectAsyncValue for this.  We have to
  // be a bit careful though because a concurrent task could swap in the actual
  // result while we're working on this.
  assert(0 && "indirect value isn't implemented yet");
  abort();
}

//===----------------------------------------------------------------------===//
// PrimitiveInfo
//===----------------------------------------------------------------------===//

namespace {
/// One entry of this record is kept for each primitive in a BEFFunction.  This
/// allows us to efficiently map from a primitive index to a primitive offset.
struct PrimitiveInfo {
  /// This is the number of operands that still need to be resolved before the
  /// primitive can be executed.
  std::atomic<int> numOperandsNotReady;

  /// This is the offset of the primitive from the start of the function body.
  /// It is stored in units of 32-bit entries, not bytes.
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

  /// These are the indices of the result registers for the function.
  SmallVector<unsigned, 8> resultRegs;
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
               MutableArrayRef<RCRef<AsyncValue>> results,
               ArrayRef<unsigned> resultRegIndex);

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
  /// Set up registers for input arguments to a function.
  void processArgumentsPseudo(SmallVectorImpl<unsigned> &primitiveIDs,
                              ArrayRef<AsyncValue *> arguments);

  /// Decrement the `numOperandsNotReady` counters for the specified primitives
  /// by one, executing them if they are now ready to run. This processes the
  /// entries from the end of the vector to the start - worklist style.
  void decrementOperandsNotReadyCounts(SmallVectorImpl<unsigned> &primitiveIDs);

private:
  /// Make sure the BEF file doesn't get deallocated while we're
  /// asynchronously running stuff.
  RCRef<BEFFileImpl> befFile;

  const unsigned numRegisters, numPrimitives;

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
  for (size_t i = 0; i != regTable.size(); ++i) {
    size_t numUses;
    if (funcReader.parseVBR(numUses))
      return true;
    new (&regTable[i]) RegisterInfo(numUses);
  }

  // Parse the primitive table.
  MutableArrayRef<PrimitiveInfo> primTable = getPrimitiveInfoTable();
  for (size_t i = 0; i != primTable.size(); ++i) {
    size_t offset, numOperandsBeforeExecuting;
    if (funcReader.parseVBR(offset) ||
        funcReader.parseVBR(numOperandsBeforeExecuting))
      return true;
    new (&primTable[i]) PrimitiveInfo(offset, numOperandsBeforeExecuting);
  }

  // Parse the ResultRegs
  auto numResults = decoder.fn.getNumResults();
  decoder.resultRegs.reserve(numResults);
  for (size_t i = 0; i != numResults; ++i) {
    size_t registerIndex;
    if (funcReader.parseVBR(registerIndex))
      return true;
    if (registerIndex >= regTable.size())
      return funcReader.emitError("invalid result register #");
    decoder.resultRegs.push_back(registerIndex);
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
                          MutableArrayRef<RCRef<AsyncValue>> results,
                          ArrayRef<unsigned> resultRegIndex) {
  // Now that the executor is all set up and ready to go, kick off the ready
  // primitives.

  // We initialized each PrimitiveInfo::numOperandsNotReady to one plus the
  // number of arguments in the primitive. This means that as we walk the list
  // to drop the argument count, if we hit zero then it is time for us to
  // trigger the computation. This arrangement is nice because any primitive
  // that immediately produces results will immediately unblock subsequent
  // primitives to be run by the current thread, which results in zero thread
  // hops, clean top-down execution semantics (very cache friendly and good for
  // more predictable unit tests), and results in all the atomics staying in
  // that cores' cache.
  SmallVector<unsigned, 16> primitiveIDToVisit;

  // If a kernel's result has multiple uses, we will get multiple entries added
  // on the worklist. We reserve some extra space to accommodate this growth.
  primitiveIDToVisit.reserve(numPrimitives + 4);
  // Reverse indices in primitiveIDToVisit because
  // decrementOperandsNotReadyCounts processes its argument from back to front.
  for (unsigned i = 0, e = numPrimitives; i != e; ++i)
    primitiveIDToVisit.push_back(e - i - 1);

  // If we have arguments, then the first primitive is a pseudo kernel that gets
  // special handling.
  if (!arguments.empty())
    processArgumentsPseudo(primitiveIDToVisit, arguments);

  // Burn through the worklist of primitives to execute.
  decrementOperandsNotReadyCounts(primitiveIDToVisit);

  // Populate the function result AsyncValues.
  //
  // The primitives may have resolved some of the register values (e.g. if you
  // return an argument register) but typically they are not yet available.
  // Handle this by creating indirect values as necessary.  When the primitive
  // that computes the register completes, the IndirectAsyncValue is resolved.
  assert(results.size() == resultRegIndex.size());
  MutableArrayRef<RegisterInfo> regTable = getRegisterInfoTable();
  for (size_t i = 0, e = results.size(); i != e; ++i) {
    assert(!results[i] && "result AsyncValue is not nullptr");
    // We already checked the resultRegIndex numbers are inrange when parsing
    // that array.  This 'takes' the reference because the refcount was
    // initialized to know that a reference from this result exists.
    results[i] =
        takeRCRef(regTable[resultRegIndex[i]].getOrCreateIndirectValue());
  }
}

/// Set up registers for input arguments to a function.
void BEFExecutor::processArgumentsPseudo(
    SmallVectorImpl<unsigned> &primitiveIDs, ArrayRef<AsyncValue *> arguments) {
  /// TODO: InitializeArgumentRegisters
  assert(0 && "arguments not implemented yet!");
  abort();
}

/// Decrement the `numOperandsNotReady` counters for the specified primitives by
/// one, executing them if they are now ready to run. This processes the entries
/// from the end of the vector to the start - worklist style.
void BEFExecutor::decrementOperandsNotReadyCounts(
    SmallVectorImpl<unsigned> &primitiveIDs) {

  MutableArrayRef<PrimitiveInfo> primitives = getPrimitiveInfoTable();

  while (!primitiveIDs.empty()) {
    auto primitiveID = primitiveIDs.pop_back_val();
    assert(primitiveID < primitives.size() && "invalid primitive ID");

    // Decrement the count and see if we're ready to run.  If not, then the
    // primitive isn't ready to run yet.
    if (primitives[primitiveID].numOperandsNotReady.fetch_sub(1) != 1)
      continue;

    const uint32_t *befPrimitiveEntry =
        &funcBody[primitives[primitiveID].offset];
    uint32_t opcode = *befPrimitiveEntry++;
    uint32_t locOffset = *befPrimitiveEntry++;
    uint32_t numOperandsAndAttrs = *befPrimitiveEntry++;
    uint16_t numOperands = uint16_t(numOperandsAndAttrs);
    uint16_t numAttrs = uint16_t(numOperandsAndAttrs >> 16);

    uint32_t numFunctionsAndResults = *befPrimitiveEntry++;
    uint16_t numFunctions = uint16_t(numFunctionsAndResults);
    uint16_t numResults = uint16_t(numFunctionsAndResults >> 16);

    // Find the implementation of this primitive.
    auto fnImpl = befFile->primitives[opcode];
    assert(fnImpl != nullptr);

    auto numResultUsers = llvm::makeArrayRef(befPrimitiveEntry, numResults);
    befPrimitiveEntry += numResults;
    auto operands = llvm::makeArrayRef(befPrimitiveEntry, numOperands);
    befPrimitiveEntry += numOperands;
    auto attrOffsets = llvm::makeArrayRef(befPrimitiveEntry, numAttrs);
    befPrimitiveEntry += numAttrs;

    // Keep track of whether we saw any error arguments. If so, we propagate the
    // error to the results automatically.
    // TODO(cancelation): Initialize it with the cancel async value: if the
    // execution has been canceled we'll stop running primitives.
    AsyncValue *anyErrorArgument = nullptr;

    // Handle operands.
    assert(operands.empty() && "operands not supported yet");

    // Set up attributes.
    assert(attrOffsets.empty() && "attributes not supported yet");

    // Set up functions.
    assert(numFunctions == 0);

    // If all arguments are good then run the function.
    if (!anyErrorArgument) {
      // Set the location for the primitive so they can report errors.
      (void)locOffset;
      // Run it.
      fnImpl();
    } else {
      // Otherwise, automatically propagate errors to the result values.
      assert(0 && "Error propagation not implemented");
    }

    // Now that the primitive had a chance to look at the arguments, we're done
    // with them, so they can potentially be deallocated if this was the last
    // user of them.
    assert(operands.empty() && "operands not supported yet");

    // Iterate over all results of the primitive. If a result has no users, it
    // will be skipped. Otherwise, if the primitive immediately completed a
    // result, then we can mark all users as ready to go.  If not, we need to
    // enqueue them on their unavailable operands.
    auto resultNos = llvm::makeArrayRef(befPrimitiveEntry, numResults);
    befPrimitiveEntry += numResults;

    for (size_t resultNo = 0, e = resultNos.size(); resultNo != e;
         ++resultNo) {
      assert(0 && "results not implemented yet");
    }
  }
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

  // llvm::errs() << "Executing BEF function " << getName() << "\n";

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
  executor->execute(*this, arguments, results, decoder.resultRegs);

  // The RCRef drops the reference to `executor` since it is running.
}
