# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle


@value
@register_passable("trivial")
struct CString:
    """Represents `const char*` in C. Useful for binding with C APIs."""

    var ptr: DTypePointer[DType.int8]

    fn get_as_string_ref(self) -> StringRef:
        """
        Get the `CString` as `StringRef`. Lifetime is tied to C API.
        For owning version use `__str__()`.
        """
        return StringRef(self.ptr)

    fn __str__(self) -> String:
        """
        Get `CString` as a owning `String`.
        """
        return String(self.ptr)


@always_inline("nodebug")
fn exchange[T: AnyType](inout old_var: T, owned new_value: T) -> T:
    """
    Assign `new_value` to `old_var` and returns the value previously
    contained in `old_var`.
    """
    let old = old_var
    old_var = new_value ^
    return old ^


# ======================================================================#
#                                                                       #
# Utility structs and functions to interact with dylibs.                #
#                                                                       #
# ======================================================================#


@value
@register_passable("trivial")
struct SingleArgCallable[ResultTy: AnyType, ArgTy: AnyType]:
    var func: fn (ArgTy) -> ResultTy

    @always_inline("nodebug")
    fn __call__(self, arg: ArgTy) -> ResultTy:
        return self.func(arg)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyType
](borrowed lib: DLHandle, name: StringRef) -> ResultTy:
    """Call function `name` in dylib with one result and no arguments."""
    return lib.get_function[SingleArgCallable[ResultTy, NoneType]](name)(None)


@always_inline("nodebug")
fn call_dylib_func[
    ArgTy: AnyType
](borrowed lib: DLHandle, name: StringRef, arg: ArgTy) -> None:
    """Call function `name` in dylib with no result and one argument."""
    lib.get_function[SingleArgCallable[NoneType, ArgTy]](name)(arg)


@always_inline("nodebug")
fn call_dylib_func[
    ResultTy: AnyType, ArgTy: AnyType
](borrowed lib: DLHandle, name: StringRef, arg: ArgTy) -> ResultTy:
    """Call function `name` in dylib with one result and one argument."""
    return lib.get_function[SingleArgCallable[ResultTy, ArgTy]](name)(arg)
