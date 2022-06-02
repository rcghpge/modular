//===- BEFInvokable.cpp ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements logic required to execute a invokable.  This is the core
// execution logic for the BEF executor runtime.
//
//===----------------------------------------------------------------------===//

#include "GraphRT/BEFExecutor/BEFInvokable.h"

#include "BEFFileImpl.h"
#include "GraphRT/BEF/BEFReader.h"
#include "GraphRT/BEFExecutor/PrimitiveDescriptor.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/Runtime.h"
#include "llvm/Support/TrailingObjects.h"
using namespace M;

//===----------------------------------------------------------------------===//
// RegisterInfo
//===----------------------------------------------------------------------===//

namespace {
/// Each instance of this struct is an entry in the register file for the BEF
/// executor.  The executor keeps one of these for each SSA value in the .mlir
/// file.
struct RegisterInfo {
  /// This is the number of uses of the register in the invokable.  The
  /// AsyncValue may be deallocated when this number of users have completed
  /// execution.  This field allows us to allocate the register with the correct
  /// reference count up front.
  const size_t userCount;

  explicit RegisterInfo(unsigned userCount) : userCount(userCount) {}

  /// Return the AsyncValue for this register, this is used when the client
  /// knows it already exists or is prepared to deal with a missing value.
  AsyncValue *getValue() { return value.load(std::memory_order_acquire); }

  /// Return the AsyncValue for this register, creating it on demand as
  /// necessary.
  AsyncValue *getOrCreateIndirectValue(Runtime &runtime);

  /// Set this register to the specified value when the primitive that
  /// constructs it has completed.  This returns the actual value held by the
  /// register file, which may be an IndirectAsyncValue if someone needed the
  /// result register before it was computed.
  AsyncValue *setRegister(AsyncValue *newValue);

  /// Set this register to the specified value when initializing arguments. This
  /// is equivalent but slightly cheaper than setRegister.
  void setArgumentRegister(AsyncValue *newValue);

private:
  /// This is a pointer to the AsyncValue for a register as the invokable
  /// executes.
  ///
  /// It is set to a non-null AsyncValue in two cases:
  ///   1) when the primitive that generates it executes, or
  ///   2) when the result is demanded early (e.g. due to a non-strict primitive
  ///      that uses the register as an operand, or due to the results of the
  ///      invokable).  In this case it is filled with IndirectAsyncValue.
  ///
  /// Once filled in, we bias the reference count with the number of unexecuted
  /// primitive operands left to execute (this is what is stored in userCount).
  /// Each of these RC's get dropped when the using primitive executes.
  ///
  /// In the case of an IndirectAsyncValue (case #2 above), an extra +1 RC is
  /// present which gets dropped when the indirect value is resolved.
  std::atomic<AsyncValue *> value{nullptr};
};
} // end anonymous namespace

/// Return the AsyncValue for this register, creating it on demand as
/// necessary.
AsyncValue *RegisterInfo::getOrCreateIndirectValue(Runtime &runtime) {
  // In the normal case, just load the pointer and return it.
  AsyncValue *valuePtr = getValue();
  if (valuePtr)
    return valuePtr;

  // If it doesn't exist, we create an IndirectAsyncValue for this.  We have to
  // be a bit careful though because a concurrent task could swap in the actual
  // result while we're working on this.
  AsyncValue *indirect = AsyncValue::createIndirect(runtime).release();
  AsyncValue *existing = nullptr;

  // Speculatively set refcount to `userCount` refs with the expectation that
  // the compare_exchange succeeds.  indirect starts with 1 reference for the
  // RCRef, and setting this register will count as an additional use (+1), so
  // bringing its refcount to (userCount + 1).
  //
  // We keep an extra +1 on IndirectAsyncValue's in the register file until they
  // are resolved.  That way they don't get deallocated in the case when all
  // the primitives using the register result get run (e.g. when nonstrict)
  // ahead of the value being resolved.
  assert(userCount && "shouldn't build an IndirectAsyncValue without users");
  AnyAsyncValueRef::lowLevelAddRef(indirect, userCount);
  if (value.compare_exchange_strong(existing, indirect,
                                    std::memory_order_release,
                                    std::memory_order_acquire))
    return indirect; // Success: old value = nullptr!

  // If someone beat us to setting a value then we don't need the
  // IndirectAsyncValue after all. Decrease refcount back to 0 to free it.
  AnyAsyncValueRef::lowLevelDropRef(indirect, userCount + 1);
  return existing;
}

/// Set this register to the specified value when the primitive that
/// constructs it has completed.  This returns the actual value held by the
/// register file, which may be an IndirectAsyncValue if someone needed the
/// result register before it was computed.
AsyncValue *RegisterInfo::setRegister(AsyncValue *newValue) {
  // If this result is unused, save some atomic operations by not storing the
  // value in the register.
  if (userCount == 0) {
    // We know there must not have been an indirect reference to resolve.
    assert(!getValue());
    return newValue;
  }

  // Atomically set value to newValue.
  AsyncValue *existing = nullptr;
  if (value.compare_exchange_strong(existing, newValue,
                                    std::memory_order_release,
                                    std::memory_order_acquire))
    return newValue; // Success: old value = nullptr!

  // If there was already a value in it, it must be a IndirectAsyncValue. We
  // set the IndirectAsyncValue to point to the result.
  existing->resolveIndirect(copyRCRef(newValue));
  return existing;
}

/// Set this register to the specified value when initializing arguments.
void RegisterInfo::setArgumentRegister(AsyncValue *newValue) {
  // If this result is unused, save some atomic operations by not storing the
  // value in the register.
  if (userCount != 0)
    value = newValue;
}

//===----------------------------------------------------------------------===//
// PrimitiveInfo
//===----------------------------------------------------------------------===//

namespace {
/// One entry of this record is kept for each primitive in a BEFInvokable.  This
/// allows us to efficiently map from a primitive index to a primitive offset.
struct PrimitiveInfo {
  /// This is the number of operands that still need to be resolved before the
  /// primitive can be executed.
  std::atomic<int> numOperandsNotReady;

  /// This is the offset of the primitive from the start of the invokable body.
  /// It is stored in units of 32-bit entries, not bytes.
  unsigned offset;

  /// We initialize the ready list to "numOperands + 1" so we can drop the
  /// last count in the executor constructor.
  PrimitiveInfo(unsigned offset, unsigned numOperandsBeforeExecuting)
      : numOperandsNotReady(numOperandsBeforeExecuting + 1), offset(offset) {}
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BEFInvokableDecoder
//===----------------------------------------------------------------------===//

namespace {
/// This class is a helper used to handle initial invokable decoding before the
/// BEFExecutor can be allocated.
struct BEFInvokableDecoder {
public:
  BEFInvokableDecoder(const BEFInvokableImpl &inv)
      : inv(inv), bodyReader(inv.owner->invokableBodies, [&](Error message) {
          befFormatError = std::move(message);
        }) {}

  /// Decoding the invokable record.
  bool decodeHeader() {
    return bodyReader.dropBytes(inv.bodyOffset) ||
           bodyReader.parseVBR(numPrimitives) ||
           bodyReader.parseVBR(numRegisters);
  }

public:
  const BEFInvokableImpl &inv;
  Optional<Error> befFormatError;
  BEFReader bodyReader;

  /// These are the fields we decode.
  size_t numPrimitives = 0, numRegisters = 0;

  /// These are the indices of the result registers for the invokable.
  SmallVector<unsigned, 8> resultRegs;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BEFExecutor implementation
//===----------------------------------------------------------------------===//

namespace {
/// A BEFExecutor runs an invokable containing a stream of asynchronous
/// primitives.  Multiple executors can be active at one time, even for the same
/// invokable on disk.
///
/// The layout of this class is the BEFExecutor itself, following by a trailing
/// array of "N" RegisterInfo records, following by "M" PrimitiveInfo records,
/// where N and M are encoded into the invokable header.
class BEFExecutor final
    : public ReferenceCounted<BEFExecutor>,
      public LocationDecoder,
      public llvm::TrailingObjects<BEFExecutor, RegisterInfo, PrimitiveInfo> {
public:
  BEFExecutor(BEFFileImpl *file, Runtime &runtime, BEFInvokableDecoder &decoder)
      : befFile(copyRCRef(file)), runtime(runtime),
        numRegisters(decoder.numRegisters),
        numPrimitives(decoder.numPrimitives) {}
  ~BEFExecutor() {
    // NOTE: we don't bother destructing the ResultInfo/PrimitiveInfo records.
  }

  /// Decode the register info and primitive info tables.  This returns true on
  /// failure.
  bool decodeTables(BEFInvokableDecoder &decoder);

  /// Execute the invokable once the tables are decoded.
  void execute(const BEFInvokableImpl &invokable,
               MutableArrayRef<AnyAsyncValueRef> arguments,
               MutableArrayRef<AnyAsyncValueRef> results,
               ArrayRef<unsigned> resultRegIndex);

  /// When the last reference to the BEFExecutor is dropped, we deallocate
  /// ourself.  The memory for this class is managed by the current Runtime.
  void destroy() const {
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

  // Implement LocationDecoder hooks.
  void addRef() override {
    RCRef<ReferenceCounted<BEFExecutor>>::lowLevelAddRef(this);
  }
  void dropRef() override {
    RCRef<ReferenceCounted<BEFExecutor>>::lowLevelDropRef(this);
  }
  LLCL::DecodedLocation decode(const EncodedLocation &loc) const override;

private:
  /// Set up registers for input arguments to a invokable.
  void processArgumentsPseudo(SmallVectorImpl<uint32_t> &primitiveIDWorklist,
                              MutableArrayRef<AnyAsyncValueRef> arguments);

  /// Decrement the `numOperandsNotReady` counters for the specified primitives
  /// by one, executing them if they are now ready to run. This processes the
  /// entries from the end of the vector to the start - worklist style.
  void decrementOperandsNotReadyCounts(
      SmallVectorImpl<uint32_t> &primitiveIDWorklist);
  void processUsedByRecords(AnyAsyncValueRef result, AsyncValue *registerVal,
                            ArrayRef<uint32_t> users,
                            uint32_t totalUserRefCount,
                            SmallVectorImpl<uint32_t> &primitiveIDWorklist);

private:
  /// Make sure the BEF file doesn't get deallocated while we're
  /// asynchronously running stuff.
  RCRef<BEFFileImpl> befFile;
  Runtime &runtime;

  const unsigned numRegisters, numPrimitives;

  // All of the primitive entry records for this invokable.
  ArrayRef<uint32_t> funcBody;
};
} // end anonymous namespace

/// This is the core method that implements the `LocationDecoder` abstract base
/// class.  This turns the data payload in EncodedLocation (which, in our case,
/// is an offset into the locations section) into a decoded location.
LLCL::DecodedLocation BEFExecutor::decode(const EncodedLocation &loc) const {
  BEFReader reader(befFile->locationPositions, [&](Error message) {});

  size_t filenameOffset, lineNo, colNo;
  if (reader.dropBytes(loc.getData()) || reader.parseVBR(filenameOffset) ||
      reader.parseVBR(lineNo) || reader.parseVBR(colNo))
    return LLCL::DecodedLocation{"<<BEF file encoding error>>", 0, -1};

  reader.setContents(befFile->locationFilenames);
  StringRef filename;
  if (reader.parseNULTerminatedStringAt(filenameOffset, filename))
    return LLCL::DecodedLocation{"<<BEF file encoding error>>", 0, -1};
  return {filename.str(), (int)lineNo, (int)colNo};
}

/// Decode the register info and primitive info tables.  This returns true on
/// failure.
bool BEFExecutor::decodeTables(BEFInvokableDecoder &decoder) {
  auto &bodyReader = decoder.bodyReader;

  // Parse the register table.
  MutableArrayRef<RegisterInfo> regTable = getRegisterInfoTable();
  for (size_t i = 0, e = regTable.size(); i != e; ++i) {
    size_t numUses;
    if (bodyReader.parseVBR(numUses))
      return true;
    new (&regTable[i]) RegisterInfo(numUses);
  }

  // Parse the primitive table.
  MutableArrayRef<PrimitiveInfo> primTable = getPrimitiveInfoTable();
  for (size_t i = 0; i != primTable.size(); ++i) {
    size_t offset, numOperandsBeforeExecuting;
    if (bodyReader.parseVBR(offset) ||
        bodyReader.parseVBR(numOperandsBeforeExecuting))
      return true;
    new (&primTable[i]) PrimitiveInfo(offset, numOperandsBeforeExecuting);
  }

  // Parse the ResultRegs
  auto numResults = decoder.inv.getNumResults();
  decoder.resultRegs.reserve(numResults);
  for (size_t i = 0; i != numResults; ++i) {
    size_t registerIndex;
    if (bodyReader.parseVBR(registerIndex))
      return true;
    if (registerIndex >= regTable.size())
      return bodyReader.emitError("invalid result register #");
    decoder.resultRegs.push_back(registerIndex);
  }

  // Align to where the invokable body is.
  if (bodyReader.alignTo(BEF::kPrimitiveEntryAlignment))
    return true;

  funcBody = llvm::makeArrayRef((const uint32_t *)bodyReader.data(),
                                bodyReader.getContents().size() / 4);
  return false;
}

/// Execute the invokable once the tables are decoded.
void BEFExecutor::execute(const BEFInvokableImpl &invokable,
                          MutableArrayRef<AnyAsyncValueRef> arguments,
                          MutableArrayRef<AnyAsyncValueRef> results,
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
  SmallVector<uint32_t, 16> primitiveIDToVisit;

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

  // Populate the invokable result AsyncValues.
  //
  // The primitives may have resolved some of the register values (e.g. if you
  // return an argument register) but typically they are not yet available.
  // Handle this by creating indirect values as necessary.  When the primitive
  // that computes the register completes, the IndirectAsyncValue is resolved.
  assert(results.size() == resultRegIndex.size());
  MutableArrayRef<RegisterInfo> regTable = getRegisterInfoTable();
  for (size_t i = 0, e = results.size(); i != e; ++i) {
    assert(!results[i] && "result AsyncValue is not nullptr");
    // We already checked the resultRegIndex numbers are in-range when parsing
    // that array.  This 'takes' the reference because the refcount was
    // initialized to know that a reference from this result exists.
    results[i] = takeRCRef(
        regTable[resultRegIndex[i]].getOrCreateIndirectValue(runtime));
  }
}

namespace {
/// This struct holds the data decoded for a primitive in an invokable.
struct PrimitiveDecoder {
  const uint32_t *dataPtr;

  uint32_t opcode, locOffset;
  uint16_t numOperands, numAttrs;
  uint16_t numInvokables, numResults;
  ArrayRef<uint32_t> perResultUsersCount;
  ArrayRef<uint32_t> operandIndices;
  ArrayRef<uint32_t> attrOffsets;
  ArrayRef<uint32_t> invokableRefIndices;
  uint32_t firstResultRegNumber;

  explicit PrimitiveDecoder(const uint32_t *dataPtr) {
    // Decode the fixed header.
    opcode = *dataPtr++;
    locOffset = *dataPtr++;
    uint32_t numOperandsAndAttrs = *dataPtr++;
    numOperands = uint16_t(numOperandsAndAttrs);
    numAttrs = uint16_t(numOperandsAndAttrs >> 16);
    uint32_t numInvokablesAndResults = *dataPtr++;
    numInvokables = uint16_t(numInvokablesAndResults);
    numResults = uint16_t(numInvokablesAndResults >> 16);

    // Decode the variable size fields.
    perResultUsersCount = llvm::makeArrayRef(dataPtr, numResults);
    dataPtr += numResults;
    operandIndices = llvm::makeArrayRef(dataPtr, numOperands);
    dataPtr += numOperands;
    attrOffsets = llvm::makeArrayRef(dataPtr, numAttrs);
    dataPtr += numAttrs;
    invokableRefIndices = llvm::makeArrayRef(dataPtr, numInvokables);
    dataPtr += numInvokables;
    if (numResults)
      firstResultRegNumber = *dataPtr++;
    else
      firstResultRegNumber = ~0U;
    this->dataPtr = dataPtr;
  }

  ArrayRef<uint32_t> getEncodedElements(size_t numElements) {
    auto result = llvm::makeArrayRef(dataPtr, numElements);
    dataPtr += numElements;
    return result;
  }
};
} // end anonymous namespace

/// Set up registers for input arguments to an invokable.
void BEFExecutor::processArgumentsPseudo(
    SmallVectorImpl<uint32_t> &primitiveIDWorklist,
    MutableArrayRef<AnyAsyncValueRef> arguments) {
  // We're handling the 0th primitive, so remove it from the worklist.
  assert(primitiveIDWorklist.back() == 0);
  primitiveIDWorklist.pop_back();

  // Decode the first primitive, which is the pseudo-op for the argument
  // block.
  PrimitiveDecoder decoder(&funcBody[0]);
  assert(decoder.numOperands == 0 && decoder.numAttrs == 0 &&
         decoder.numInvokables == 0 && decoder.numResults == arguments.size() &&
         "malformed argument record");

  MutableArrayRef<RegisterInfo> registers = getRegisterInfoTable();

  // Process each argument in turn, capturing its value into the register file,
  // and (if the value is ready) adding all the users of it to the worklist for
  // processing.
  for (size_t i = 0, e = arguments.size(); i != e; ++i) {
    AsyncValue *argPtr = arguments[i].getPointer();
    registers[i].setArgumentRegister(argPtr);

    // Get the array of primitive ID's that use this result.
    uint32_t numResultUsers = decoder.perResultUsersCount[i];
    auto users = decoder.getEncodedElements(numResultUsers);

    // Process users of this result, adding them to the worklist as needed. This
    // consumes the argument pointer's +1 value.
    processUsedByRecords(std::move(arguments[i]), argPtr, users,
                         registers[i].userCount, primitiveIDWorklist);
  }
}

/// Decrement the `numOperandsNotReady` counters for the specified primitives by
/// one, executing them if they are now ready to run. This processes the entries
/// from the end of the vector to the start - worklist style.
void BEFExecutor::decrementOperandsNotReadyCounts(
    SmallVectorImpl<uint32_t> &primitiveIDWorklist) {
  RunnablePrimitiveDescriptorBuilder frame(runtime, befFile->attributes, *this);

  MutableArrayRef<RegisterInfo> registers = getRegisterInfoTable();
  MutableArrayRef<PrimitiveInfo> primitives = getPrimitiveInfoTable();

  while (!primitiveIDWorklist.empty()) {
    auto primitiveID = primitiveIDWorklist.pop_back_val();
    assert(primitiveID < primitives.size() && "invalid primitive ID");

    // Decrement the count and see if we're ready to run.  If not, then the
    // primitive isn't ready to run yet.
    if (primitives[primitiveID].numOperandsNotReady.fetch_sub(1) != 1)
      continue;

    // Reset the frame back to its initial state so we can fill it in for this
    // invocation.
    frame.reset();

    // Decode the primitive record.
    PrimitiveDecoder decoder(&funcBody[primitives[primitiveID].offset]);

    // Find the implementation of this primitive.
    auto invokableImpl = befFile->primitives[decoder.opcode];
    assert(invokableImpl != nullptr);

    // Keep track of whether we saw any error arguments. If so, we propagate the
    // error to the results automatically.  We initialize with the runtime's
    // cancelation setting so cancellation quashes primitive execution.
    AsyncValue *anyErrorArgument = runtime.getCancelValue();

    // Reserve space in the frame to avoid vector reallocation.
    frame.reserveArgsAndResults(decoder.numOperands, decoder.numResults);

    // Handle operands.
    for (auto registerIdx : decoder.operandIndices) {
      RegisterInfo &reg = registers[registerIdx];

      // The operand register may not be available for non-strict primitives
      // that are started before all operands are available. In that case, we
      // use an IndirectAsyncValue so it can be resolved later.
      AsyncValue *value = reg.getOrCreateIndirectValue(runtime);
      frame.addArg(takeRCRef(value));

      // If any operand is an error, we implicitly propagate it through to all
      // results and short-circuit execution of the primitive.  Note that we
      // still process all the operands here, even if we find an error, because
      // we need to decrement the use count of each register to avoid leaks.
      if (value->isError())
        anyErrorArgument = value;
    }

    // Handle results.
    frame.setNumResults(decoder.numResults);

    // If all arguments are good then run the invokable.
    if (!anyErrorArgument) {
      // Set up attributes, the frame object takes the offsets directly.
      frame.setAttrs(decoder.attrOffsets);

      // Set up invokables.
      for (size_t invokableID : decoder.invokableRefIndices)
        frame.addInvokable(befFile->getInvokableByID(invokableID));

      // Set the location for the primitive so they can report errors.
      frame.setLocationData(decoder.locOffset);

      // Run it.
      invokableImpl(frame);
    } else {
      // Otherwise, automatically propagate errors to the result values.
      for (size_t i = 0; i != decoder.numResults; ++i)
        frame.setResult(i, copyRCRef(anyErrorArgument));
    }

    // Now that the primitive had a chance to look at the operand registers,
    // we're done with them.  Drop their reference so they can potentially be
    // deallocated if this was the last user of them.
    for (AnyAsyncValueRef &arg : frame.getArguments())
      arg.reset();

    // Iterate over all results of the primitive.  If the primitive immediately
    // completed a result (its AsyncValue is "ready"), then we can add all users
    // to our worklist.  If not, we `andThen` on their unavailable result.
    for (size_t resultNo = 0, e = decoder.numResults; resultNo != e;
         ++resultNo) {
      AnyAsyncValueRef &result = frame.getResult(resultNo);
      assert(result && "Primitive impl did not set a result!");

      // Store the value in the register table, and get the final AsyncValue to
      // be used by the register. This could be an IndirectAsyncValue.
      RegisterInfo &resultReg =
          registers[decoder.firstResultRegNumber + resultNo];
      auto *registerVal = resultReg.setRegister(result.getPointer());

      // Get the array of primitive ID's that use this result.
      uint32_t numResultUsers = decoder.perResultUsersCount[resultNo];
      auto users = decoder.getEncodedElements(numResultUsers);

      // Process users of this result, adding them to the worklist as needed.
      processUsedByRecords(std::move(result), registerVal, users,
                           resultReg.userCount, primitiveIDWorklist);
    }
  }
}

/// After setting a result into the register file, process all the "used by"
/// records for this result.  If the result is 'ready' (typical in synchronous
/// kernels) then we want to add users to the worklist.  If not, we have to do
/// `andThen` logic to process them when they become available.
///
/// This method also adjusts the reference count of `result` and `registerVal`
/// to prepare for the users, who will each drop a reference as they go.  This
/// count of users is specified by `totalUserRefCount`.  It may be larger than
/// `users.size()` because the later doesn't account for bef.output.
void BEFExecutor::processUsedByRecords(
    AnyAsyncValueRef result, AsyncValue *registerVal, ArrayRef<uint32_t> users,
    uint32_t totalUserRefCount,
    SmallVectorImpl<uint32_t> &primitiveIDWorklist) {
  assert(users.size() <= totalUserRefCount &&
         "totalUserRefCount is primitive users + bef.output");

  auto *resultPtr = result.getPointer();

  // In the degenerate case of no users, this drops the last reference to the
  // AsyncValue and it is deallocated.
  if (totalUserRefCount == 0) {
    assert(resultPtr == registerVal &&
           "cannot have an indirectvalue with no users");
    // Just allow `result` to go out of scope and drop its reference.
    return;
  }

  // Make sure the result value has a +1 for each user.  It already has a +1
  // reference from the `result` RCRef, so we add '#users - 1' references.
  if (resultPtr == registerVal) { // IndirectValue RC already handled.
    if (totalUserRefCount != 1)   // Don't atomic add 0.
      AnyAsyncValueRef::lowLevelAddRef(resultPtr, totalUserRefCount - 1);
    // The reference count is handled manually above, don't drop another RC.
    result.release();
  } else {
    // If we are fulfilling an IndirectValue then the IndirectValue already took
    // a +1 to this AsyncValue.  We don't need another reference, so let
    // `result` do the dropRef when it goes out of scope.

    // We are however done with the extra reference on the register
    // IndirectAsyncValue itself, so we can drop it.
    (void)takeRCRef(registerVal);
  }

  // If the result is ready (typical case), then we can be efficient and avoid
  // "andThen" logic.
  if (resultPtr->isReady()) {
    primitiveIDWorklist.append(users.begin(), users.end());
  } else if (!users.empty()) {
    // Otherwise, the primitive is going to produce its result asynchronously -
    // process the user whenever the value becomes available.

    // Keep this executor alive as long as the result is running, so all the
    // state we maintain (e.g. attribute pointers) stay alive.
    RCRef<BEFExecutor> thisRef = copyRCRef(this);

    // Process the whole batch of users when this result becomes available. Note
    // that we capture `users` which is an ArrayRef instead of copying the
    // content. This is fine because the underlying BEF file is alive when the
    // BEF executor is.
    resultPtr->andThen([thisRef = std::move(thisRef), users]() {
      SmallVector<uint32_t, 16> primitiveIDToVisit(users.begin(), users.end());
      thisRef->decrementOperandsNotReadyCounts(primitiveIDToVisit);
    });
  }
}

//===----------------------------------------------------------------------===//
// BEFInvokable
//===----------------------------------------------------------------------===//

/// Add a new reference to this object.  The lifetime of BEFInvokables are
/// completely tied to BEFFileImpl, so just extend its reference count.
void BEFInvokableImpl::addRef() const {
  RCRef<BEFFileImpl>::lowLevelAddRef(owner);
}

/// Drop a reference to this object, potentially deallocating it.
void BEFInvokableImpl::dropRef() const {
  RCRef<BEFFileImpl>::lowLevelDropRef(owner);
}

/// Invoke this, passing the specified arguments. This returns one AsyncValue
/// for each result.
void BEFInvokableImpl::invoke(MutableArrayRef<AnyAsyncValueRef> arguments,
                              MutableArrayRef<AnyAsyncValueRef> results,
                              Runtime &runtime) const {
  assert(arguments.size() == getNumArgs() &&
         "incorrect number of argument to an invokable");
  assert(results.size() == getNumResults() &&
         "incorrect number of results for an invokable");

  BEFInvokableDecoder decoder(*this);
  bool headerParseFailure = decoder.decodeHeader();

  // We allocate a single slab of memory for the BEFExecutor object itself,
  // along with the register and primitive entries.  This give good locality as
  // well as reducing #allocations.
  auto sizeInBytes = BEFExecutor::totalSizeToAlloc<RegisterInfo, PrimitiveInfo>(
      decoder.numRegisters, decoder.numPrimitives);
  void *rawPtr = runtime.allocateBytes(sizeInBytes, alignof(BEFExecutor));
  auto executor = takeRCRef(new (rawPtr) BEFExecutor(owner, runtime, decoder));

  // Parse the rest of the tables now that we allocated memory.
  if (headerParseFailure || executor->decodeTables(decoder)) {
    // Fill in error async value results if there are any present.
    if (results.empty())
      return;

    // For our location, we can use anything in the file.  There isn't really a
    // good way to handle this.  We could encoded a location for each function
    // into the function index section, but it would be a shame to bloat out our
    // memory representation for this case.
    auto curLocation = EncodedLocation(/*some location=*/0, executor.copy());
    results[0] = AsyncValue::createError(
        runtime, EncodedDiagnostic{std::move(decoder.befFormatError.getValue()),
                                   std::move(curLocation)});
    // Fill in the rest with the same error.
    for (size_t i = 1, e = results.size(); i != e; ++i)
      results[i] = results[0].copy();
    return;
  }

  // Finally, start execution of any primitives that are ready to go.
  executor->execute(*this, arguments, results, decoder.resultRegs);

  // The RCRef drops the reference to `executor` since it is running.
}

/// Return the invokable with the specified Invokable ID.
BEFInvokable *BEFFileImpl::getInvokableByID(size_t number) const {
  assert(number < numInvokables && "invalid invokable");
  // invokables are stored consequtively in memory.
  return invokables + number;
}
