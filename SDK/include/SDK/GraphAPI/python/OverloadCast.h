//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_GRAPHAPI_PYTHON_OVERLOADCAST_H
#define SDK_GRAPHAPI_PYTHON_OVERLOADCAST_H

#include <type_traits>
#include <utility>

//===----------------------------------------------------------------------===//
// overload_cast
//===----------------------------------------------------------------------===//

/// Trait for selecting a specific overload of a function pointer
template <typename... Args>
struct overload_cast_impl {
  template <typename Return>
  constexpr auto
  operator()(Return (*pf)(Args...)) const noexcept -> decltype(pf) {
    return pf;
  }

  template <typename Return, typename Class>
  constexpr auto
  operator()(Return (Class::*pmf)(Args...)) const noexcept -> decltype(pmf) {
    return pmf;
  }

  template <typename Return, typename Class>
  constexpr auto operator()(Return (Class::*pmf)(Args...)
                                const) const noexcept -> decltype(pmf) {
    return pmf;
  }

  template <typename F,
            typename = std::enable_if_t<std::is_invocable_v<F, Args...>>>
  constexpr F operator()(F &&func) const noexcept {
    return std::forward<F>(func);
  }
};

/// Decorator to select a specific overload of a function pointer
template <typename... Args>
static constexpr overload_cast_impl<Args...> overload_cast = {};

#endif // SDK_GRAPHAPI_PYTHON_OVERLOADCAST_H
