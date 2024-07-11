//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ArraySupport/Tensor.h"
#include "ArraySupport/TensorBufferRef.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"

#include <numeric>

using namespace M;

Tensor Tensor::createOwned(TensorSpec spec, AsyncRT::CompactRuntimePtr runtime,
                           std::optional<size_t> alignment) {
  return Tensor(
      TensorBufferRef::createOwned(runtime, spec.getSizeInBytes(), alignment),
      std::move(spec));
}

Tensor Tensor::takeOwned(TensorSpec spec, AsyncRT::CompactRuntimePtr runtime,
                         void *takeFrom) {
  return Tensor(TensorBufferRef::take(runtime, spec.getSizeInBytes(), takeFrom),
                std::move(spec));
}

Tensor Tensor::createBorrowed(void *buffer, TensorSpec spec,
                              std::optional<size_t> alignment,
                              TensorBufferRef::Lifetime lifetime) {
  return Tensor(TensorBufferRef::createWithNonTrackedMemory(
                    buffer, spec.getSizeInBytes(), alignment),
                std::move(spec));
}

Tensor Tensor::createBorrowed(const void *buffer, TensorSpec spec,
                              std::optional<size_t> alignment,
                              TensorBufferRef::Lifetime lifetime) {
  return Tensor(
      TensorBufferRef::createWithNonTrackedMemory(
          const_cast<void *>(buffer), spec.getSizeInBytes(), alignment),
      std::move(spec));
}

Tensor Tensor::borrow(TensorBufferRef::Lifetime borrowLifetime) const {
  return Tensor(bufferRef.borrow(), TensorSpec(spec));
}

void Tensor::printDebug(raw_ostream &os) const {
  os << "Tensor(";
  os << spec;
  os << ", ";
  os << bufferRef;
  os << ")";
}

//===----------------------------------------------------------------------===//
// Tensor printing functionality.
//===----------------------------------------------------------------------===//

namespace {
/// We're using raw_ostream to print the tensor (which is probably a bad idea
/// by itself :), which auto formats "char" as a character instead of printing
/// out the byte value.  Use this template to cast to an element type that will
/// cooperate when printing.
template <typename EltType>
struct ElementPrintType {
  using Type = EltType;
};

template <>
struct ElementPrintType<int8_t> {
  using Type = int32_t;
};

template <>
struct ElementPrintType<uint8_t> {
  using Type = uint32_t;
};

static constexpr auto kStartTensorMarker = '[';
static constexpr auto kEndTensorMarker = ']';
static constexpr auto kTensorFiller = " ..., ";
static constexpr size_t kCompactMaxElemsToPrint = 7;
static constexpr size_t kCompactElemPerSide = kCompactMaxElemsToPrint / 2;

} // namespace

template <typename uintT>
static void printElement(raw_ostream &os, const uintT &elem, const DType &dtype,
                         [[maybe_unused]] llvm::FloatStyle floatStyle,
                         [[maybe_unused]] size_t precision) {
  constexpr int numBits = sizeof(elem) * CHAR_BIT;
  static_assert(std::is_unsigned_v<uintT> && std::is_integral_v<uintT>,
                "Use uintXX_t for storage type.");
  static_assert(numBits <= 64,
                "Only support printing up to 64 bit types for now.");

  APInt asAPInt = llvm::APInt(numBits, elem);
  if (dtype.isUInt()) {
    os << asAPInt.getZExtValue();
    return;
  }

  if (dtype.isSInt()) {
    os << asAPInt.getSExtValue();
    return;
  }

  if (dtype.isFloat()) {
    auto writeFloat = [&](const llvm::fltSemantics &semantic) {
      double value = APFloat(semantic, asAPInt).convertToDouble();
      llvm::write_double(os, value, floatStyle, precision);
    };
    dtype.dispatch<void>()
        .template when<DType::f16>([&]() { writeFloat(APFloat::IEEEhalf()); })
        .template when<DType::bf16>([&]() { writeFloat(APFloat::BFloat()); })
        .template when<DType::f32>([&]() { writeFloat(APFloat::IEEEsingle()); })
        .template when<DType::f64>([&]() { writeFloat(APFloat::IEEEdouble()); })
        .otherwise([&]() { os << dtype.getAsString(); });
    return;
  }

  if (dtype.isBool()) {
    os << (asAPInt.getZExtValue() ? "True" : "False");
    return;
  }

  os << dtype.getAsString();
}

template <typename T, typename ElementFunctor>
static ErrorOrSuccess printElementsComplete(const T *ptr, size_t numElements,
                                            ElementFunctor printFn,
                                            raw_ostream &os) {
  auto elts = ArrayRef(ptr, numElements);
  llvm::interleaveComma(elts, os, printFn);
  return success();
}

template <typename T, typename ElementFunctor>
static ErrorOrSuccess printElementsCompact(const T *ptr, size_t numElements,
                                           ElementFunctor printFn,
                                           raw_ostream &os) {
  os << kStartTensorMarker;
  if (numElements < kCompactMaxElemsToPrint) {
    ErrorOrSuccess wasSuccess =
        printElementsComplete(ptr, numElements, printFn, os);
    os << kEndTensorMarker;
    return wasSuccess;
  }

  auto elts = ArrayRef(ptr, numElements);
  llvm::interleaveComma(elts.take_front(kCompactElemPerSide), os, printFn);

  /// Add ... to fill the intermediate values.
  os << ",";
  os << kTensorFiller;

  llvm::interleaveComma(elts.take_back(kCompactElemPerSide), os, printFn);
  os << kEndTensorMarker;
  return success();
}

template <typename FormattingFunc>
static ErrorOrSuccess printElementsWithFormatting(
    raw_ostream &os, const void *srcPtr, const DType &dtype,
    const size_t &numElements, const llvm::FloatStyle &floatStyle,
    const size_t &precision, FormattingFunc &&printWithFormat) {
  auto printWithStorageWidth = [&](const auto *a) -> ErrorOrSuccess {
    return printWithFormat(
        a, numElements,
        [&](auto elem) {
          printElement(os, elem, dtype, floatStyle, precision);
        },
        os);
  };

  switch (dtype.getWidthInBits()) {
  case 8:
    return printWithStorageWidth(static_cast<const uint8_t *>(srcPtr));
  case 16:
    return printWithStorageWidth(static_cast<const uint16_t *>(srcPtr));
  case 32:
    return printWithStorageWidth(static_cast<const uint32_t *>(srcPtr));
  case 64:
    return printWithStorageWidth(static_cast<const uint64_t *>(srcPtr));
  }

  return Error("Unhandled bit-width " + Twine(dtype.getWidthInBits()));
}

static ErrorOrSuccess printTensorComplete(raw_ostream &os, const void *srcPtr,
                                          const TensorSpec &spec,
                                          llvm::FloatStyle floatStyle,
                                          size_t precision) {
  os << "tensor<";
  for (auto elt : spec)
    os << elt << "x";

  auto eltType = spec.getEltType();
  os << eltType.getAsString() << "> [";

  size_t numElements = spec.getNumElements();

  ErrorOrSuccess didPrint = printElementsWithFormatting(
      os, srcPtr, eltType, numElements, floatStyle, precision,
      [&](auto ptr, auto num_elements, auto printFn, auto &os) {
        return printElementsComplete(ptr, num_elements, printFn, os);
      });

  os << "]\n";
  os.flush();

  if (didPrint.isError()) {
    std::string errorMsg;
    llvm::raw_string_ostream errStream(errorMsg);
    errStream << "Printing-Error";
    errStream << "Could not print tensor with dtype " << eltType.getAsString()
              << " with error: \n"
              << didPrint.getError() << "\n";
    return Error(errorMsg);
  }

  return success();
}

static ErrorOrSuccess printTensorCompact(raw_ostream &os, const void *srcPtr,
                                         const TensorSpec &spec,
                                         llvm::FloatStyle floatStyle,
                                         size_t precision) {

  /// What we are doing is printing a series of 2D arrays if the dimension
  /// is greater than 2. We limit print depth to 7 in height, width and depth
  /// Assume the dimension is [1, 9, 100, 100]. We would print pick first 3
  /// [1, 100, 100] matrices, and last 3 [1, 100, 100] matrices and print
  /// them. In each of those we only print first, and last 3 rows and
  /// first and last 3 columns. The intermediaries are filled with '...'
  /// to indicate something is here but we are not displaying it.

  auto dims = spec.getDimsCopy();
  size_t columnElemCount = dims.empty() ? 1 : dims.pop_back_val();
  // If the tensor is a rank-1 vector, then the number of rows is 1.
  size_t rowElemCount = dims.empty() ? 1 : dims.pop_back_val();
  size_t matrixElemCount = columnElemCount * rowElemCount;
  std::string errorMsg;
  llvm::raw_string_ostream errStream(errorMsg);
  os << "tensor(";
  auto eltType = spec.getEltType();

  // Open parens for every other dimension other than rowElemCount &
  // columnElemCount
  for (size_t i = 2, e = spec.getRank(); i < e; ++i)
    os << kStartTensorMarker;

  /// We are basically printing a bunch of 2D tensors in succession
  /// So if a tensor is of numMatrices * rowElemCount * columnElemCount
  /// dimension, we print numMatrices tensors of rowElemCount * columnElemCount
  /// dimension. numMatrices equals the product of all dims other than last two

  /// We have popped out all dims other than those relevant to numMatrices
  /// already.
  size_t numMatrices =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());

  for (size_t matrixIdx = 0; matrixIdx < numMatrices;) {

    /// This is the marker between a rowElemCount*columnElemCount tensor and
    /// next. We don't need marker for the first time
    if (matrixIdx > 0)
      os << ",\n\n";
    os << kStartTensorMarker;

    /// Print row
    for (size_t rowIdx = 0; rowIdx < rowElemCount;) {

      /// We print the next row in a new line
      if (rowIdx > 0)
        os << "\n";

      ErrorOrSuccess didPrint = printElementsWithFormatting(
          os, srcPtr, eltType, columnElemCount, floatStyle, precision,
          [&](auto ptr, auto num_elements, auto printFn, auto &os) {
            return printElementsCompact(ptr + matrixIdx * matrixElemCount +
                                            rowIdx * columnElemCount,
                                        num_elements, printFn, os);
          });

      if (didPrint.isError()) {
        os << "Printing-Error";
        errStream << "Could not print tensor with dtype "
                  << eltType.getAsString() << " with error: \n"
                  << didPrint.getError() << "\n";
      }

      ++rowIdx;

      /// We are skipping printing comma after the last bracket.
      if (rowIdx != rowElemCount)
        os << ",";

      // Intermediate rows are filled with "..." and rowIdx is advanced to third
      // from last
      if (rowElemCount >= kCompactMaxElemsToPrint &&
          rowIdx == kCompactElemPerSide) {
        os << "\n" << kTensorFiller << "\n";
        rowIdx = rowElemCount - kCompactElemPerSide;
      }
    }
    os << kEndTensorMarker;
    ++matrixIdx;

    // Again intermediate matrices are skipped for compactness
    if (numMatrices >= kCompactMaxElemsToPrint &&
        matrixIdx == kCompactElemPerSide) {
      os << "\n" << kTensorFiller << "\n";
      matrixIdx = numMatrices - kCompactElemPerSide;
    }
  }

  // Now every element is printed. We just have to close all open parans.
  for (size_t i = 2, rank = spec.getRank(); i < rank; ++i)
    os << kEndTensorMarker;

  os << ", dtype=";
  os << eltType.getAsString();
  os << ", shape=[";
  llvm::interleave(spec, os, ",");
  os << "])\n";

  if (!errorMsg.empty())
    os << errorMsg << "\n";
  os.flush();
  return success();
}

/// This kernel prints the specified tensor data to stdout.
ErrorOrSuccess M::printTensor(raw_ostream &os, const void *srcPtr,
                              const TensorSpec &spec, ResultOutputStyle style,
                              unsigned precision) {
  // TODO: Add print support for quantization.
  switch (style) {
  case ResultOutputStyle::kFull:
    return printTensorComplete(os, srcPtr, spec, llvm::FloatStyle::Exponent,
                               precision);
  case ResultOutputStyle::kCompact:
    return printTensorCompact(os, srcPtr, spec, llvm::FloatStyle::Fixed, 4);
  case ResultOutputStyle::kBinary:
  case ResultOutputStyle::kNone:

    // No-op.
    return success();
  }
  return Error("unknown ResultOutputStyle style");
}

/// This prints the specified tensor data to stdout.
ErrorOrSuccess Tensor::print(ResultOutputStyle style,
                             unsigned precision) const {
  return print(llvm::outs(), style, precision);
}

/// This prints the specified tensor data to stream.
ErrorOrSuccess Tensor::print(raw_ostream &os, ResultOutputStyle style,
                             unsigned precision) const {
  return printTensor(os, getBuffer(), getSpec(), style, precision);
}
