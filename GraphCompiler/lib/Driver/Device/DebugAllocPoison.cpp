//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Device/DebugAllocPoison.h"
#include "Support/Configuration.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

using namespace M;
using M::GraphCompiler::Driver::AllocationPoisonDecision;
using M::GraphCompiler::Driver::PoisonPattern;

std::optional<PoisonPattern>
M::GraphCompiler::Driver::getPoisonPattern(const TensorSpec &spec) {
  DType dtype = spec.getEltType();

  if (const llvm::fltSemantics *sem = dtype.getFloatSemantics()) {
    if (llvm::APFloat::getSizeInBits(*sem) < 8)
      return std::nullopt;
    if (llvm::is_contained(
            std::initializer_list<DType>{DType::f8e3m4, DType::f8e8m0fnu,
                                         DType::f8e4m3fn, DType::f8e4m3fnuz,
                                         DType::f8e5m2, DType::f8e5m2fnuz},
            dtype))
      return std::nullopt;
    llvm::APFloat largest = llvm::APFloat::getLargest(*sem);
    return PoisonPattern{largest.bitcastToAPInt().getZExtValue(),
                         llvm::APFloat::getSizeInBits(*sem) / 8,
                         "largest-finite"};
  }

  // Integer types. Every float dtype we support has LLVM APFloat semantics
  // (see `Support/include/Support/ML/FloatTypes.def`), so a future float type
  // without semantics is the only way to land here as a float — at which
  // point this code needs to grow a fill strategy for it rather than
  // silently fall through to 0xCD.
  assert(!dtype.isFloat() && "float dtype lacks LLVM APFloat semantics");
  return PoisonPattern{0xCD, 1, "byte-fill"};
}

void M::GraphCompiler::Driver::fillHostMemory(void *ptr, size_t totalBytes,
                                              const PoisonPattern &pattern) {
  uint8_t *dst = static_cast<uint8_t *>(ptr);
  if (pattern.patternBytes == 1) {
    std::memset(dst, static_cast<uint8_t>(pattern.bits), totalBytes);
    return;
  }
  // Replicate the patternBytes-byte little-endian pattern across the
  // buffer. We avoid `std::fill_n` through a `uint{16,32,64}_t *` because
  // the caller's allocation does not guarantee `alignof(T)` for T larger
  // than a byte, and a misaligned typed store is UB. The `bits` argument to
  // `memcpy` is taken by address but read as a contiguous byte sequence, so
  // no alignment requirement applies.
  for (size_t off = 0; off + pattern.patternBytes <= totalBytes;
       off += pattern.patternBytes) {
    std::memcpy(dst + off, &pattern.bits, pattern.patternBytes);
  }
}

namespace {

constexpr PoisonPattern kZeroFill{0, 1, "zero"};

/// The parsed form of a selection spec. Empty `dtypes` means "any"; the
/// ordinal range defaults to everything.
struct PoisonSelection {
  enum class Mode {
    Select,   ///< Poison what the filters match, zero-fill the rest.
    None,     ///< Zero-fill everything.
    SyncOnly, ///< Write nothing.
  };
  Mode mode = Mode::Select;
  llvm::SmallVector<DType, 4> dtypes;
  uint64_t minOrdinal = 0;
  uint64_t maxOrdinal = std::numeric_limits<uint64_t>::max();
};

/// Parses `N` or `LO-HI` into an inclusive range.
std::optional<std::pair<uint64_t, uint64_t>>
parseOrdinalRange(llvm::StringRef value) {
  llvm::StringRef loStr = value, hiStr = value;
  if (size_t dash = value.find('-'); dash != llvm::StringRef::npos) {
    loStr = value.take_front(dash);
    hiStr = value.drop_front(dash + 1);
  }
  uint64_t lo, hi;
  if (loStr.trim().getAsInteger(10, lo) || hiStr.trim().getAsInteger(10, hi) ||
      lo > hi)
    return std::nullopt;
  return std::make_pair(lo, hi);
}

std::optional<llvm::SmallVector<DType, 4>> parseDTypes(llvm::StringRef value) {
  llvm::SmallVector<llvm::StringRef, 4> names;
  value.split(names, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (names.empty())
    return std::nullopt;
  llvm::SmallVector<DType, 4> dtypes;
  for (llvm::StringRef name : names) {
    auto dtype = DType::getFromString(name.trim());
    // `invalid` parses, but no allocation has it, so it would select nothing.
    if (failed(dtype) || !dtype->isValid())
      return std::nullopt;
    dtypes.push_back(*dtype);
  }
  return dtypes;
}

PoisonSelection parseSelection(llvm::StringRef spec) {
  PoisonSelection sel;
  spec = spec.trim();
  if (spec.empty() || spec == "all")
    return sel;
  if (spec == "none") {
    sel.mode = PoisonSelection::Mode::None;
    return sel;
  }
  if (spec == "sync-only") {
    sel.mode = PoisonSelection::Mode::SyncOnly;
    return sel;
  }

  // Not `;`: `Config` reads it as the start of a `modular.cfg` comment and
  // would silently drop every clause after the first.
  llvm::SmallVector<llvm::StringRef, 4> clauses;
  spec.split(clauses, '&', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  bool haveDTypes = false, haveOrdinals = false;
  for (llvm::StringRef clause : clauses) {
    clause = clause.trim();
    auto [key, value] = clause.split('=');
    key = key.trim();
    bool parsed = false;
    if (key == "dtype" && !haveDTypes) {
      if (auto dtypes = parseDTypes(value)) {
        sel.dtypes = std::move(*dtypes);
        haveDTypes = parsed = true;
      }
    } else if (key == "ord" && !haveOrdinals) {
      if (auto range = parseOrdinalRange(value)) {
        std::tie(sel.minOrdinal, sel.maxOrdinal) = *range;
        haveOrdinals = parsed = true;
      }
    }
    if (!parsed)
      llvm::errs()
          << "DEBUG_ALLOC_POISON: ignoring malformed selection clause '"
          << clause << "'\n";
  }
  return sel;
}

bool matches(const PoisonSelection &sel, const TensorSpec &spec,
             uint64_t ordinal) {
  if (ordinal < sel.minOrdinal || ordinal > sel.maxOrdinal)
    return false;
  return sel.dtypes.empty() ||
         llvm::is_contained(sel.dtypes, spec.getEltType());
}

AllocationPoisonDecision decide(const PoisonSelection &sel,
                                const TensorSpec &spec, uint64_t ordinal) {
  switch (sel.mode) {
  case PoisonSelection::Mode::SyncOnly:
    return {ordinal, std::nullopt};
  case PoisonSelection::Mode::None:
    return {ordinal, kZeroFill};
  case PoisonSelection::Mode::Select:
    if (!matches(sel, spec, ordinal))
      return {ordinal, kZeroFill};
    return {ordinal, M::GraphCompiler::Driver::getPoisonPattern(spec)};
  }
  llvm_unreachable("unknown poison selection mode");
}

void logDecision(const PoisonSelection &sel, const TensorSpec &spec,
                 bool hostAlloc, const AllocationPoisonDecision &decision) {
  llvm::raw_ostream &os = llvm::errs();
  os << "DEBUG_ALLOC_POISON: " << (decision.fill ? "Filling " : "Leaving ")
     << (hostAlloc ? "host " : "device ") << spec.getEltType()
     << " allocation (" << spec.getSizeInBytes() << " bytes) ordinal "
     << decision.ordinal;
  if (!decision.fill) {
    os << " untouched ("
       << (sel.mode == PoisonSelection::Mode::SyncOnly
               ? "sync-only"
               : "no poison pattern for this dtype")
       << ")\n";
    return;
  }
  const PoisonPattern &fill = *decision.fill;
  os << " with " << fill.patternKind << " pattern ";
  (os.changeColor(llvm::raw_ostream::RED)
   << "0x" << llvm::format_hex_no_prefix(fill.bits, fill.patternBytes * 2))
      .resetColor();
  os << "\n";
}

/// What identifies one version of the selection file. Content is deliberately
/// not part of it: rewriting the same selection must still start an epoch.
struct SelectionFileVersion {
  llvm::sys::fs::UniqueID id;
  llvm::sys::TimePoint<> mtime;
  uint64_t size = 0;

  bool operator==(const SelectionFileVersion &other) const {
    return id == other.id && mtime == other.mtime && size == other.size;
  }
};

SelectionFileVersion versionOf(const llvm::sys::fs::file_status &status) {
  return {status.getUniqueID(), status.getLastModificationTime(),
          status.getSize()};
}

struct SelectionState {
  std::mutex mutex;
  PoisonSelection selection;
  std::string path; ///< Empty when the selection is fixed for the process.
  /// The file version `selection` was read from; empty until the first read.
  std::optional<SelectionFileVersion> version;
  uint64_t epoch = 0;
  /// Reset whenever the selection file changes, so ordinals are relative to
  /// the epoch rather than to the boot.
  uint64_t nextOrdinal = 0;
};

std::string configValue(llvm::StringRef key) {
  ErrorOr<Config> config = Config::open();
  if (config.isError())
    return {};
  return config->getValue(key).str();
}

void resetState(SelectionState &st, std::string path) {
  st.selection = PoisonSelection{};
  st.path = std::move(path);
  st.version.reset();
  st.epoch = 0;
  st.nextOrdinal = 0;
}

void configure(SelectionState &st) {
  if (std::string path = configValue("debug.alloc_poison_select_file");
      !path.empty()) {
    llvm::errs() << "DEBUG_ALLOC_POISON: reading selection from " << path
                 << " on every allocation; each change resets ordinals\n";
    resetState(st, std::move(path));
    return;
  }
  resetState(st, {});
  std::string spec = configValue("debug.alloc_poison_select");
  st.selection = parseSelection(spec);
  if (!spec.empty())
    llvm::errs() << "DEBUG_ALLOC_POISON: selection '" << spec
                 << "'; unselected allocations are zero-filled\n";
}

SelectionState &state() {
  static SelectionState st;
  static std::once_flag once;
  std::call_once(once, [] { configure(st); });
  return st;
}

/// Re-reads the selection file if it changed, which starts a new epoch.
///
/// The file is stat'd on every allocation rather than on a timer. A timer
/// would let a request begin under the previous epoch's ordinals, and the
/// point of resetting per epoch is that a caller can address "the n-th
/// allocation of this request" -- true only if the reset lands after the
/// last allocation before the request and before the first one in it. A
/// stat costs a microsecond against a device allocation that synchronizes.
void refreshSelection(SelectionState &st) {
  if (st.path.empty())
    return;

  // Unreadable: keep the last good selection, which is `all` if the file has
  // never been read. Failing open beats failing clean, which would present as
  // a run with no uninitialized reads in it.
  llvm::sys::fs::file_status status;
  if (llvm::sys::fs::status(st.path, status))
    return;
  if (st.version == versionOf(status))
    return;

  // Take the version from the descriptor the contents are read through. A
  // second lookup by path lets a rename land in between, pairing the new
  // contents with the old version, and the next allocation would then start
  // the same selection's epoch over again.
  llvm::Expected<llvm::sys::fs::file_t> fd =
      llvm::sys::fs::openNativeFileForRead(st.path);
  if (!fd) {
    llvm::consumeError(fd.takeError());
    return;
  }
  llvm::scope_exit closeFd{[&] { llvm::sys::fs::closeFile(*fd); }};
  if (llvm::sys::fs::status(*fd, status))
    return;
  // The file is rewritten while the process runs, so it must not be mmapped.
  auto buffer = llvm::MemoryBuffer::getOpenFile(
      *fd, st.path, status.getSize(), /*RequiresNullTerminator=*/false,
      /*IsVolatile=*/true);
  if (!buffer)
    return;
  llvm::StringRef spec = (*buffer)->getBuffer().trim();

  // Recording the version taken before the read, not after, means an in-place
  // write landing during it shows up as another change on the next allocation.
  st.selection = parseSelection(spec);
  st.version = versionOf(status);
  st.epoch++;
  st.nextOrdinal = 0;

  llvm::errs() << "DEBUG_ALLOC_POISON: selection epoch " << st.epoch << " '"
               << spec << "'; ordinals reset\n";
}

} // namespace

AllocationPoisonDecision
M::GraphCompiler::Driver::decidePoison(const TensorSpec &spec, bool hostAlloc) {
  SelectionState &st = state();
  // One lock covers the refresh, the ordinal and the log line, so an
  // allocation can't pair one epoch's selection with the next epoch's
  // ordinal, and the log lists ordinals in order.
  std::lock_guard<std::mutex> lock(st.mutex);
  // Refresh before consuming an ordinal, not after: a refresh resets the
  // counter, so taking the ordinal first would hand the epoch's first
  // allocation the previous epoch's last number.
  refreshSelection(st);
  AllocationPoisonDecision decision =
      decide(st.selection, spec, st.nextOrdinal++);
  logDecision(st.selection, spec, hostAlloc, decision);
  return decision;
}

AllocationPoisonDecision M::GraphCompiler::Driver::decidePoisonForTesting(
    llvm::StringRef selectionSpec, const TensorSpec &spec, uint64_t ordinal) {
  return decide(parseSelection(selectionSpec), spec, ordinal);
}

void M::GraphCompiler::Driver::resetPoisonSelectionForTesting(
    llvm::StringRef path) {
  SelectionState &st = state();
  std::lock_guard<std::mutex> lock(st.mutex);
  if (path.empty())
    configure(st);
  else
    resetState(st, path.str());
}
