//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineAPI/python/InputProcessing.h"
#include "SDK/API/max/c/internal/value.h"
#include "SDK/API/max/c/value.h"
#include "SDK/API/max/cpp/aliases.h"
#include "SDK/EngineAPI/python/MojoValue.h"
#include "Support/CPython/PythonObject.h"

using namespace M;

namespace nb = nanobind;

/// Recursively converts the nanobind object `pyObj` containing a value composed
/// of `array`, `dict`, `list`, `tuple`, `none`, `double`, `long`, `bool`
/// objects to a nested value of generic composed (`Dict`, `List`, Tuple`),
/// tensor (`Tensor`) and primitive types. Tensor data arrays are
/// borrowed and not copied.
M_AsyncValue *core::processValueInput(const nb::object &pyObj,
                                      M_RuntimeContext *context,
                                      M_Status *status) {
  if (nb::dict pyDict; nb::try_cast(pyObj, pyDict)) {
    M_AsyncValue *dictValue = M_createDictAsyncValue(context);
    M_AsyncDict *dict = M_getDictFromValue(dictValue);
    for (auto elem : pyDict) {
      M_AsyncValue *key =
          processValueInput(nb::cast<nb::object>(elem.first), context, status);
      M_AsyncValue *value =
          processValueInput(nb::cast<nb::object>(elem.second), context, status);
      M_insertIntoDict(dict, key, value);
      M_freeValue(key);
      M_freeValue(value);
    }
    M_freeDict(dict);
    return dictValue;
  }

  if (nb::list pyList; nb::try_cast(pyObj, pyList)) {
    M_AsyncValue *listValue = M_createListAsyncValue(context);
    M_AsyncList *list = M_getListFromValue(listValue);
    for (auto elem : pyList) {
      M_AsyncValue *value =
          processValueInput(nb::cast<nb::object>(elem), context, status);
      M_appendToList(list, value);
      M_freeValue(value);
    }
    M_freeList(list);
    return listValue;
  }

  if (nb::tuple pyTuple; nb::try_cast(pyObj, pyTuple)) {
    M_AsyncValue *tupleValue = M_createTupleAsyncValue(context);
    M_AsyncTuple *tuple = M_getTupleFromValue(tupleValue);
    for (auto elem : pyTuple) {
      M_AsyncValue *value =
          processValueInput(nb::cast<nb::object>(elem), context, status);
      M_borrowIntoTuple(tuple, value);
      M_freeValue(value);
    }
    M_freeTuple(tuple);
    return tupleValue;
  }

  if (pyObj.is_none())
    return M_createNoneAsyncValue(context);

  if (nb::str str; nb::try_cast(pyObj, str)) {
    return M_createStringAsyncValue(str.c_str(), context);
  }

  if (int64_t i; nb::try_cast(pyObj, i))
    return M_createLongAsyncValue(i, context);

  if (double d; nb::try_cast(pyObj, d))
    return M_createDoubleAsyncValue(d, context);

  if (bool b; nb::try_cast(pyObj, b))
    return M_createBoolAsyncValue(b, context);

  if (core::MojoValue * mojoVal; nb::try_cast(pyObj, mojoVal))
    return M_createMojoAsyncValueBorrowingFromExisting(mojoVal->value.get(),
                                                       context);

  using namespace ::M::CPython;
  PythonObjectWrapper *wrapper = new PythonObjectWrapper(pyObj.ptr());

  return M_createMojoAsyncValueFromPythonObjectWrapper(
      static_cast<void *>(wrapper), &freePythonObjectWrapper, context);
}
