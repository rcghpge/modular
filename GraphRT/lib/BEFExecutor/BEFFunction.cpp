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
#include "GraphRT/BEFExecutor/PrimitiveFrame.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/Runtime.h"
#include "llvm/Support/TrailingObjects.h"
using namespace M;

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

  /// Set this register to the specified value when the primitive that
  /// constructs it has completed.  This returns the actual value held by the
  /// register file, which may be an IndirectAsyncValue if someone needed the
  /// result register before it was computed.
  AsyncValue *setRegister(AsyncValue *newValue);

  /// Set this register to the specified value when initializing arguments. This
  /// is equivalent but slightly cheaper than setRegister.
  void setArgumentRegister(AsyncValue *newValue);

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
                                    std::memory_order_acquire)) {
    // Success!
    return newValue;
  }

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
  void execute(const BEFFunction &fn,
               MutableArrayRef<RCRef<AsyncValue>> arguments,
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
  void processArgumentsPseudo(SmallVectorImpl<uint32_t> &primitiveIDWorklist,
                              MutableArrayRef<RCRef<AsyncValue>> arguments);

  /// Decrement the `numOperandsNotReady` counters for the specified primitives
  /// by one, executing them if they are now ready to run. This processes the
  /// entries from the end of the vector to the start - worklist style.
  void decrementOperandsNotReadyCounts(
      SmallVectorImpl<uint32_t> &primitiveIDWorklist);
  void processUsedByRecords(RCRef<AsyncValue> result, AsyncValue *registerVal,
                            ArrayRef<uint32_t> users,
                            uint32_t totalUserRefCount,
                            SmallVectorImpl<uint32_t> &primitiveIDWorklist);

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
                          MutableArrayRef<RCRef<AsyncValue>> arguments,
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

namespace {
/// This struct holds the data decoded for a primitive in a BEF function.
struct PrimitiveDecoder {
  const uint32_t *dataPtr;

  uint32_t opcode, locOffset;
  uint16_t numOperands, numAttrs;
  uint16_t numFunctions, numResults;
  llvm::ArrayRef<uint32_t> perResultUsersCount;
  llvm::ArrayRef<uint32_t> operandIndices;
  llvm::ArrayRef<uint32_t> attrOffsets;
  llvm::ArrayRef<uint32_t> functionIndices;
  uint32_t firstResultRegNumber;

  explicit PrimitiveDecoder(const uint32_t *dataPtr) {
    // Decode the fixed header.
    opcode = *dataPtr++;
    locOffset = *dataPtr++;
    uint32_t numOperandsAndAttrs = *dataPtr++;
    numOperands = uint16_t(numOperandsAndAttrs);
    numAttrs = uint16_t(numOperandsAndAttrs >> 16);
    uint32_t numFunctionsAndResults = *dataPtr++;
    numFunctions = uint16_t(numFunctionsAndResults);
    numResults = uint16_t(numFunctionsAndResults >> 16);

    // Decode the variable size fields.
    perResultUsersCount = llvm::makeArrayRef(dataPtr, numResults);
    dataPtr += numResults;
    operandIndices = llvm::makeArrayRef(dataPtr, numOperands);
    dataPtr += numOperands;
    attrOffsets = llvm::makeArrayRef(dataPtr, numAttrs);
    dataPtr += numAttrs;
    functionIndices = llvm::makeArrayRef(dataPtr, numFunctions);
    dataPtr += numFunctions;
    if (numResults)
      firstResultRegNumber = *dataPtr++;
    else
      firstResultRegNumber = ~0U;
    this->dataPtr = dataPtr;
  }

  llvm::ArrayRef<uint32_t> getEncodedElements(size_t numElements) {
    auto result = llvm::makeArrayRef(dataPtr, numElements);
    dataPtr += numElements;
    return result;
  }
};
} // end anonymous namespace

/// Set up registers for input arguments to a function.
void BEFExecutor::processArgumentsPseudo(
    SmallVectorImpl<uint32_t> &primitiveIDWorklist,
    MutableArrayRef<RCRef<AsyncValue>> arguments) {
  // We're handling the 0th primitive, so remove it from the worklist.
  assert(primitiveIDWorklist.back() == 0);
  primitiveIDWorklist.pop_back();

  // Decode the first primitive, which is the pseudo-op for the argument
  // block.
  PrimitiveDecoder decoder(&funcBody[0]);
  assert(decoder.numOperands == 0 && decoder.numAttrs == 0 &&
         decoder.numFunctions == 0 && decoder.numResults == arguments.size() &&
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

  PrimitiveFrameBuilder frame(befFile->runtime, befFile->attributes);

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
    auto fnImpl = befFile->primitives[decoder.opcode];
    assert(fnImpl != nullptr);

    // Keep track of whether we saw any error arguments. If so, we propagate the
    // error to the results automatically.
    // TODO(cancelation): Initialize it with the cancel async value: if the
    // execution has been canceled we'll stop running primitives.
    AsyncValue *anyErrorArgument = nullptr;

    // Reserve space in the frame to avoid vector reallocation.
    frame.reserveArgsAndResults(decoder.numOperands, decoder.numResults);

    // Handle operands.
    for (auto registerIdx : decoder.operandIndices) {
      RegisterInfo &reg = registers[registerIdx];

      // The operand register may not be available for non-strict primitives
      // that are started before all operands are available. In that case, we
      // use an IndirectAsyncValue so it can be resolved later.
      AsyncValue *value = reg.getOrCreateIndirectValue();
      frame.addArg(takeRCRef(value));

      // If any operand is an error, we implicitly propagate it through to all
      // results and short-circuit execution of the primitive.
      if (value->isError()) {
        anyErrorArgument = value;
        break;
      }
    }

    // Handle results.
    frame.setNumResults(decoder.numResults);

    // Set up attributes, the frame object takes the offsets directly.
    frame.setAttrs(decoder.attrOffsets);

    // Set up functions.
    for (size_t fnID : decoder.functionIndices)
      frame.addFunction(befFile->getFunctionByID(fnID));

    // If all arguments are good then run the function.
    if (!anyErrorArgument) {
      // Set the location for the primitive so they can report errors.
      (void)decoder.locOffset;
      // Run it.
      fnImpl(frame);
    } else {
      // Otherwise, automatically propagate errors to the result values.
      for (size_t i = 0; i != decoder.numResults; ++i)
        frame.setResult(i, copyRCRef(anyErrorArgument));
    }

    // Now that the primitive had a chance to look at the operand registers,
    // we're done with them.  Drop their reference so they can potentially be
    // deallocated if this was the last user of them.
    for (RCRef<AsyncValue> &arg : frame.getArguments())
      arg.reset();

    // Iterate over all results of the primitive.  If the primitive immediately
    // completed a result (its AsyncValue is "ready"), then we can add all users
    // to our worklist.  If not, we `andThen` on their unavailable result.
    for (size_t resultNo = 0, e = decoder.numResults; resultNo != e;
         ++resultNo) {
      RCRef<AsyncValue> &result = frame.getResult(resultNo);
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
/// This function also adjusts the reference count of `result` and `registerVal`
/// to prepare for the users, who will each drop a reference as they go.  This
/// count of users is specified by `totalUserRefCount`.  It may be larger than
/// `users.size()` because the later doesn't account for bef.output.
void BEFExecutor::processUsedByRecords(
    RCRef<AsyncValue> result, AsyncValue *registerVal, ArrayRef<uint32_t> users,
    uint32_t totalUserRefCount,
    SmallVectorImpl<uint32_t> &primitiveIDWorklist) {

  auto *resultPtr = result.getPointer();

  // If the result is ready (typical case), then we can be efficient and avoid
  // "andThen" logic.
  if (AsyncValue::isReady(result->getState())) {
    primitiveIDWorklist.append(users.begin(), users.end());
  } else {
    // Otherwise, the primitive is going to produce its result asynchronously -
    // process the user whenever the value becomes available.

    // Keep this executor alive as long as the result is running, so all the
    // state we maintain (e.g. attribute pointers) stay alive.
    addRef();

    // Process the whole batch of users when this result becomes available. Note
    // that we capture `users` which is an ArrayRef instead of copying the
    // content. This is fine because the underlying BEF file is alive when the
    // BEF executor is.
    resultPtr->andThen([this, users]() mutable {
      SmallVector<uint32_t, 16> primitiveIDToVisit(users.begin(), users.end());
      this->decrementOperandsNotReadyCounts(primitiveIDToVisit);
      this->dropRef();
    });
  }

  // If we are fulfilling an IndirectValue, then the IndirectValue takes the
  // +1 value from this AsyncValue.  Otherwise, we need to make sure the
  // result value has a +1 for each user.  It already has a +1 reference, so
  // we add '#users - 1' references.  In the degenerate case of no users, this
  // drops the last reference to the AsyncValue and it is deallocated.
  if (totalUserRefCount == 0) {
    assert(resultPtr == registerVal &&
           "cannot have an indirectvalue with no users");
    registerVal->dropRef();
  } else if (resultPtr == registerVal) { // IndirectValue RC already handled.
    if (totalUserRefCount != 1)          // Don't atomic add 0.
      registerVal->addRef(totalUserRefCount - 1);
  }
  // The reference count is handled manually, don't drop it.
  result.release();
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
void BEFFunction::execute(MutableArrayRef<RCRef<AsyncValue>> arguments,
                          MutableArrayRef<RCRef<AsyncValue>> results) const {
  assert(arguments.size() == getNumArgs() &&
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
    // Fill in error async value results if there are any present.
    if (results.empty())
      return;
    results[0] = AsyncValue::createError(
        owner->runtime, std::move(decoder.befFormatError.getValue()));
    // Fill in the rest with the same error.
    for (size_t i = 1; i != results.size(); ++i)
      results[i] = results[0].copy();
    return;
  }

  // Finally, start execution of any primitives that are ready to go.
  executor->execute(*this, arguments, results, decoder.resultRegs);

  // The RCRef drops the reference to `executor` since it is running.
}

/// Return the function with the specified BEF function ID.
const Function *BEFFileImpl::getFunctionByID(size_t number) const {
  assert(number < numFunctions && "invalid function");
  // Functions are stored consequtively in memory.
  return functions + number;
}
