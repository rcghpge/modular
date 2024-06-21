//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Classes related to returning mlir::InFlightDiagnostic
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_DIAGNOSTIC_H
#define GENERICML_SUPPORT_DIAGNOSTIC_H

#include "Support/ErrorOr.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Diagnostics.h"
#include <variant>

namespace M {

/// Class holding a Value or an mlir::InFlightDiagnostic. Use this class to
/// indicate that an error has already been emitted and a new one does not need
/// to be emitted. Functions receiving this error can modify it if necessary.
/// Use this on functions that either propagate an error or those that can emit
/// an error on an operation. Use `CompileDiagnosticOr` for functions that do
/// not have an operation to report an error on.
template <typename Value>
class [[nodiscard]] CompileDiagnosticOr {
public:
  CompileDiagnosticOr(Value value) : data(std::move(value)) {}
  CompileDiagnosticOr(mlir::InFlightDiagnostic err) : data(std::move(err)) {}
  CompileDiagnosticOr(CompileDiagnosticOr &&other)
      : data(std::move(other.data)) {}
  CompileDiagnosticOr &operator=(CompileDiagnosticOr &&other) {
    if (&other != this) {
      this->~CompileDiagnosticOr();
      new (this) CompileDiagnosticOr(std::move(other));
    }
    return *this;
  }

  /// Move constructor from value.
  template <class OtherT, typename = std::enable_if_t<
                              std::is_convertible<OtherT, Value>::value>>
  CompileDiagnosticOr(OtherT &&val) : data(std::forward<OtherT>(val)) {}

  /// Move constructor from ErrorOr.
  template <class OtherT, typename = std::enable_if_t<
                              std::is_convertible<OtherT, Value>::value>>
  CompileDiagnosticOr(ErrorOr<OtherT> &&other) : data(other) {}

  /// Check if this object contains an error.
  bool isError() const { return data.index() == 0; }
  explicit operator bool() const { return isError(); }

  /// Move the error out of this object. Object is invalid afterwards.
  mlir::InFlightDiagnostic takeError() { return std::get<0>(std::move(data)); }

  /// Move the value out of this object. Object is invalid afterwards.
  Value takeValue() { return std::get<1>(std::move(data)); }

private:
  std::variant<mlir::InFlightDiagnostic, Value> data;
};

class [[nodiscard]] CompileDiagnosticOrSuccess
    : public CompileDiagnosticOr<Detail::Empty> {
public:
  using CompileDiagnosticOr::CompileDiagnosticOr;
  // This allows initialization from success().
  /*implicit*/ CompileDiagnosticOrSuccess(SuccessType success)
      : CompileDiagnosticOr(Detail::Empty()) {}

  // Allow default initialization to success.
  CompileDiagnosticOrSuccess() : CompileDiagnosticOr(Detail::Empty()) {}
};

} // namespace M
#endif // GENERICML_SUPPORT_DIAGNOSTIC_H
