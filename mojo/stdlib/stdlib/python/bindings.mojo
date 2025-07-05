# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from collections.string.string_slice import get_static_string
from os import abort
from sys.ffi import c_int, _Global
from sys.info import sizeof
from compile.reflection import get_type_name
from memory import stack_allocation

from python import Python, PythonObject
from python._cpython import (
    Py_TPFLAGS_DEFAULT,
    Py_tp_new,
    Py_tp_dealloc,
    Py_tp_init,
    Py_tp_methods,
    Py_tp_repr,
    PyCFunction,
    PyMethodDef,
    PyObject,
    PyObjectPtr,
    PyTypeObject,
    PyTypeObjectPtr,
    PyType_Slot,
    PyType_Spec,
    GILAcquired,
    destructor,
    reprfunc,
    Typed_initproc,
    Typed_newfunc,
)
from python._python_func import PyObjectFunction
from python.python_object import _unsafe_alloc_init
from builtin._startup import _ensure_current_or_global_runtime_init

# ===-----------------------------------------------------------------------===#
# Global `PyTypeObject` Registration
# ===-----------------------------------------------------------------------===#

alias MOJO_PYTHON_TYPE_OBJECTS = _Global[
    "MOJO_PYTHON_TYPE_OBJECTS",
    Dict[StaticString, PythonObject],
    Dict[StaticString, PythonObject].__init__,
]
"""Mapping of Mojo type identifiers to unique `PyTypeObject*` binding
that Mojo type to this CPython interpreter instance."""


fn _register_py_type_object(
    type_id: StaticString, var type_obj: PythonObject
) raises:
    """Register a Python type object for the identified Mojo type.

    The provided Python type object describes how a wrapped Mojo value can
    be used from within Python code.

    Args:
        type_id: The unique type id of a Mojo type.
        type_obj: The Python type object that binds the Mojo type identified
          by `type_id`.

    Raises:
        If a Python type object has already been registered in the current
        session for the provided type id.
    """
    var type_dict = MOJO_PYTHON_TYPE_OBJECTS.get_or_create_ptr()

    if type_id in type_dict[]:
        raise Error(
            (
                "Error building multiple Python type objects bound to"
                " Mojo type with id: "
            ),
            type_id,
        )

    type_dict[][type_id] = type_obj^


fn lookup_py_type_object[T: AnyType]() raises -> PythonObject:
    """Retrieve a reference to the unique Python type describing Python objects
    containing Mojo values of type `T`.

    This function looks up the Python type object that was previously registered
    for the Mojo type `T` using a `PythonTypeBuilder`. The returned type object
    can be used to create Python objects that wrap Mojo values of type `T`.

    Parameters:
        T: The Mojo type to look up.

    Returns:
        A `PythonObject` representing the Python type object that binds the Mojo
        type `T` to the current CPython interpreter instance.

    Raises:
        If no `PythonTypeBuilder` was ever finalized for type `T`, or if no
        Python type object has been registered for the provided type identifier.
    """
    var type_dict = MOJO_PYTHON_TYPE_OBJECTS.get_or_create_ptr()

    # FIXME(MSTDL-1580):
    #   This should use a unique compiler type ID, not the Python name of this
    #   type.

    alias type_name = get_type_name[T]()
    if entry := type_dict[].find(type_name):
        return entry.take()

    raise Error(
        "No Python type object registered for Mojo type with name: ", type_name
    )


# ===-----------------------------------------------------------------------===#
# Mojo Object
# ===-----------------------------------------------------------------------===#

# https://docs.python.org/3/c-api/typeobj.html#slot-type-typedefs


struct PyMojoObject[T: AnyType]:
    """Storage backing a PyObject* wrapping a Mojo value.

    This struct represents the C-level layout of a Python object that contains
    a wrapped Mojo value. It must be ABI-compatible with CPython's PyObject
    structure to enable seamless interoperability between Mojo and Python.

    The struct follows Python's object model where all Python objects begin
    with a PyObject header (ob_base), followed by type-specific data. In this
    case, the type-specific data is a Mojo value of type T.

    See https://docs.python.org/3/c-api/structures.html#c.PyObject for more details.

    Parameters:
        T: The Mojo type being wrapped. Can be any type that satisfies `AnyType`.
    """

    var ob_base: PyObject
    """The standard Python object header containing reference count and type information.

    This must be the first field to maintain ABI compatibility with Python's object layout.
    All Python objects begin with this header structure.
    """

    var mojo_value: T
    """The actual Mojo value being wrapped and exposed to Python.

    This field stores the Mojo data that Python code can interact with through
    the registered type methods and bindings.
    """


fn _tp_dealloc_wrapper[T: AnyType](py_self: PyObjectPtr):
    """Python-compatible wrapper for deallocating a `PyMojoObject`.

    This function serves as the tp_dealloc slot for Python type objects that
    wrap Mojo values. It properly destroys the wrapped Mojo value and frees
    the Python object memory.

    Parameters:
        T: The wrapped Mojo type.

    Args:
        py_self: Pointer to the Python object to be deallocated.
    """
    var self_obj_ptr = py_self.unsized_obj_ptr.bitcast[PyMojoObject[T]]()
    var self_ptr = UnsafePointer[T](to=self_obj_ptr[].mojo_value)

    # TODO(MSTDL-633):
    #   Is this always safe? Wrap in GIL, because this could
    #   evaluate arbitrary code?
    # Destroy this `Person` instance.
    self_ptr.destroy_pointee()

    var cpython = Python().cpython()

    cpython.PyObject_Free(py_self.unsized_obj_ptr.bitcast[NoneType]())


fn _tp_repr_wrapper[T: Representable](py_self: PyObjectPtr) -> PyObjectPtr:
    """Python-compatible wrapper for generating string representation of a
    `PyMojoObject`.

    This function serves as the `tp_repr` slot for Python type objects that
    wrap Mojo values. It calls the Mojo `repr()` function on the wrapped value
    and returns the result as a Python string object.

    Parameters:
        T: The wrapped Mojo type that must be `Representable`.

    Args:
        py_self: Pointer to the Python object to get representation for.

    Returns:
        A new Python string object containing the string representation,
        or null pointer if an error occurs.
    """
    var self_obj_ptr = py_self.unsized_obj_ptr.bitcast[PyMojoObject[T]]()
    var self_ptr = UnsafePointer(to=self_obj_ptr[].mojo_value)

    var repr_str = repr(self_ptr[])

    # NOTE: it is possible that `repr` returns an invalid UTF-8 string, so we
    # let Python decode it (which will return a null pointer and set an error
    # if it is invalid).
    cpython = Python().cpython()
    return cpython.PyUnicode_DecodeUTF8(repr_str)


# ===-----------------------------------------------------------------------===#
# Builders
# ===-----------------------------------------------------------------------===#

alias PyFunction = fn (mut PythonObject, mut PythonObject) -> PythonObject
"""The generic function type for non-raising Python bindings.

The first argument is the self object, and the second argument is a tuple of the
positional arguments. These functions always return a Python object (could be a
`None` object).
"""

alias PyFunctionRaising = fn (
    mut PythonObject, mut PythonObject
) raises -> PythonObject
"""The generic function type for raising Python bindings.

The first argument is the self object, and the second argument is a tuple of the
positional arguments. These functions always return a Python object (could be a
`None` object).
"""


struct PythonModuleBuilder:
    """A builder for creating Python modules with Mojo function and type bindings.

    This builder provides a high-level API for declaring Python bindings for Mojo
    functions and types within a Python module. It manages the registration of
    functions, types, and their associated metadata, then finalizes everything
    into a complete Python module object.

    The builder follows a declarative pattern where you:
    1. Create a builder instance with a module name
    2. Add function bindings using `def_function()`, `def_py_function()`, `def_py_c_function()`
    3. Add type bindings using `add_type[T]()` and configure them
    4. Call `finalize()` to finish building the Python module.

    Example:
        ```mojo
        from python.bindings import PythonModuleBuilder

        var builder = PythonModuleBuilder("my_module")
        builder.def_function[my_func]("my_func", "Documentation for my_func")

        _ = builder.add_type[MyType]("MyType").def_method[my_method]("my_method")

        var module = builder.finalize()
        ```

    Note:
        After calling `finalize()`, the builder's internal state is cleared and
        it should not be reused for creating additional modules.

        TODO: This should be enforced programmatically in the future.
    """

    var module: PythonObject
    """The Python module being built."""

    var functions: List[PyMethodDef]
    """List of function definitions that will be exposed in the module."""

    var type_builders: List[PythonTypeBuilder]
    """List of type builders for types that will be exposed in the module."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self, name: StaticString) raises:
        """Construct a Python module builder with the given module name.

        Args:
            name: The name of the module.

        Raises:
            If the module creation fails.
        """
        self = Self(Python().create_module(name))

    fn __init__(out self, module: PythonObject):
        """Construct a Python module builder with the given module.

        Args:
            module: The module to build.
        """
        self.module = module
        self.functions = []
        self.type_builders = []

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn add_type[
        T: Representable
    ](mut self, type_name: StaticString) -> ref [
        self.type_builders
    ] PythonTypeBuilder:
        """Add a type to the module and return a builder for it.

        Parameters:
            T: The mojo type to bind in the module.

        Args:
            type_name: The name of the type to expose in the module.

        Returns:
            A reference to a type builder registered in the module builder.
        """
        self.type_builders.append(PythonTypeBuilder.bind[T](type_name))
        return self.type_builders[-1]

    fn def_py_c_function(
        mut self,
        func: PyCFunction,
        func_name: StaticString,
        docstring: StaticString = "",
    ):
        """Declare a binding for a function with PyCFunction signature in the
        module.

        Args:
            func: The function to declare a binding for.
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self.functions.append(PyMethodDef.function(func, func_name, docstring))

    fn def_py_function[
        func: PyFunction
    ](mut self, func_name: StaticString, docstring: StaticString = ""):
        """Declare a binding for a function with PyFunction signature in the
        module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self.def_py_c_function(
            _py_c_function_wrapper[func], func_name, docstring
        )

    fn def_py_function[
        func: PyFunctionRaising
    ](mut self, func_name: StaticString, docstring: StaticString = ""):
        """Declare a binding for a function with PyFunctionRaising signature in
        the module.

        Parameters:
            func: The function to declare a binding for.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        self.def_py_c_function(
            _py_c_function_wrapper[func], func_name, docstring
        )

    fn def_function[
        func_type: AnyTrivialRegType, //,
        func: PyObjectFunction[func_type, has_self=False],
    ](mut self, func_name: StaticString, docstring: StaticString = ""):
        """Declare a binding for a function with PythonObject signature in the
        module.

        These signatures can have any number of positional PythonObject
        arguments up to 6, can optionally return a PythonObject, and can raise.

        Example signature types:
        ```mojo
        alias F1 = fn (mut PythonObject) raises -> PythonObject
        alias F2 = fn (mut PythonObject, PythonObject) -> PythonObject
        alias F3 = fn (mut PythonObject, PythonObject, mut PythonObject)
        ```

        Parameters:
            func_type: The type of the function to declare a binding for.
            func: The function to declare a binding for. Users can pass their
                function directly, and it will be implicitly converted to a
                PyObjectFunction if and only if its signature is supported.

        Args:
            func_name: The name with which the function will be exposed in the
                module.
            docstring: The docstring for the function in the module.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut py_args: PythonObject
        ) raises -> PythonObject:
            return func._call_func(py_args)

        self.def_py_function[wrapper](func_name, docstring)

    fn finalize(mut self) raises -> PythonObject:
        """Finalize the module builder, creating the module object.


        All types and functions added to the builder will be built and exposed
        in the module. After calling this method, the builder's internal state
        is cleared and it should not be reused for creating additional modules.

        Returns:
            The finalized Python module containing all registered functions and types.

        Raises:
            If the module creation fails or if we fail to add any of the
            declared functions or types to the module.
        """

        Python.add_functions(self.module, self.functions)
        self.functions.clear()

        for ref builder in self.type_builders:
            builder.finalize(self.module)
        self.type_builders.clear()

        # Check or initialize the global runtime
        _ensure_current_or_global_runtime_init()

        return self.module


struct PythonTypeBuilder(Copyable, Movable):
    """A builder for a Python 'type' binding.

    This is typically used to build a type description of a `PyMojoObject[T]`.

    This builder is used to declare method bindings for a Python type, and then
    create the type binding.

    Finalizing builder created with `PythonTypeObject.bind[T]()` will globally
    register the resulting Python 'type' object as the single canonical type
    object for the Mojo type `T`. Subsequent attempts to register a Python type
    for `T` will raise an exception.

    Registering a Python type object for `T` is necessary to be able to
    construct a `PythonObject` from an instance of `T`, or to downcast an
    existing `PythonObject` to a pointer to the inner `T` value.
    """

    var type_name: StaticString
    """The name the type will be exposed as in the Python module."""

    var _type_id: Optional[StaticString]
    """The unique type identifier for the Mojo type being bound, if any."""

    var basicsize: Int
    """The required allocation size to hold an instance of this type as a Python object."""

    var _slots: Dict[Int, OpaquePointer]
    """Dictionary of Python type slots that define the behavior of the type, mapping slot number to function pointer."""

    var methods: List[PyMethodDef]
    """List of method definitions that will be exposed on the Python type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self, type_name: StaticString, *, basicsize: Int):
        """Construct a new builder for a Python type binding.

        Args:
            type_name: The name the type will be exposed as in the Python module.
            basicsize: The required allocation size to hold an instance of this
              type as a Python object.
        """

        self.type_name = type_name
        self._type_id = None
        self.basicsize = basicsize
        self._slots = {}
        self.methods = []

    @staticmethod
    fn bind[T: Representable](type_name: StaticString) -> PythonTypeBuilder:
        """Construct a new builder for a Python type that binds a Mojo type.

        Parameters:
            T: The mojo type to bind.

        Args:
            type_name: The name the type will be exposed as in the Python module.

        Returns:
            A new type builder instance.
        """
        var b = PythonTypeBuilder(
            type_name,
            basicsize=sizeof[PyMojoObject[T]](),
        )
        b._insert_slot(PyType_Slot.tp_new(_py_new_function_nonregistered))
        b._insert_slot(PyType_Slot.tp_dealloc(_tp_dealloc_wrapper[T]))
        b._insert_slot(PyType_Slot.tp_repr(_tp_repr_wrapper[T]))

        b.methods = List[PyMethodDef]()
        b._type_id = get_type_name[T]()

        return b^

    fn finalize(mut self, module: PythonObject) raises:
        """Finalize the builder and add the created type to a Python module.

        This method completes the type building process by calling the
        parameterless `finalize()` method to create the Python type object, then
        automatically adds the resulting type to the specified Python module
        using the builder's configured type name. After successful completion,
        the builder's method list is cleared to prevent accidental reuse.

        This is a convenience method that combines type finalization and module
        registration in a single operation, which is the most common use case
        when creating Python-accessible Mojo types.

        Args:
            module: The Python module to which the finalized type will be added.
                The type will be accessible from Python code that imports this
                module using the name specified during builder construction.

        Raises:
            If the type object creation fails (see `finalize()` for details) or
            if adding the type to the module fails, typically due to name
            conflicts or module state issues.

        Note:
            After calling this method, the builder's internal state is modified
            (methods list is cleared), so the builder should not be reused for
            creating additional type objects. If you need the type object for
            further operations, use the parameterless `finalize()` method
            instead and manually add it to the module.
        """
        var cpython = Python().cpython()

        if self.methods:
            self.methods.append(PyMethodDef())  # Zeroed item as terminator
            # FIXME: Avoid leaking the methods data pointer in this way.
            self._insert_slot(PyType_Slot.tp_methods(self.methods.steal_data()))

        # Convert _slots dictionary to a list of PyType_Slot structs
        var slots = List[PyType_Slot]()
        for slot_entry in self._slots.items():
            slots.append(PyType_Slot(c_int(slot_entry.key), slot_entry.value))

        # Zeroed item terminator
        slots.append(PyType_Slot.null())

        var type_spec = PyType_Spec(
            # FIXME(MOCO-1306): This should be `T.__name__`.
            self.type_name.unsafe_ptr().bitcast[sys.ffi.c_char](),
            self.basicsize,
            0,
            Py_TPFLAGS_DEFAULT,
            # Note: This pointer is only "read-only" by PyType_FromSpec.
            slots.unsafe_ptr(),
        )

        # Construct a Python 'type' object from our type spec.
        var type_obj_ptr = cpython.PyType_FromSpec(UnsafePointer(to=type_spec))

        if not type_obj_ptr:
            raise cpython.get_error()

        var type_obj = PythonObject(from_owned_ptr=type_obj_ptr)

        # Every Mojo type that is exposed to Python must have EXACTLY ONE
        # `PyTypeObject` instance that represents it. That is important for
        # correctness. This check here ensures that the user is not accidentally
        # creating multiple `PyTypeObject` instances that bind the same Mojo
        # type.
        if type_id := self._type_id:
            _register_py_type_object(type_id[], type_obj)

        Python.add_object(module, self.type_name, type_obj)
        self.methods.clear()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn _insert_slot(mut self, slot: PyType_Slot):
        """Insert a slot into the type builder.
        If the slot is already present, it will be replaced.

        Args:
            slot: The PyType_Slot to insert.
        """
        self._slots[Int(slot.slot)] = slot.pfunc

    fn def_init_defaultable[
        T: Defaultable & Movable,
    ](mut self) raises -> ref [self] Self:
        """Declare a binding for the `__init__` method of the type which
        initializes the type with a default value."""

        @always_inline
        fn default_init_func(
            out self: T, args: PythonObject, kwargs: PythonObject
        ) raises:
            if len(args) > 0 or kwargs.py_object:
                raise "unexpected arguments passed to default initializer function of wrapped Mojo type"
            self = T()

        self._insert_slot(
            PyType_Slot.tp_new(_py_new_function_wrapper[T, default_init_func])
        )
        return self

    fn def_py_init[
        T: Movable, //,
        init_func: fn (out T, args: PythonObject, kwargs: PythonObject),
    ](mut self) raises -> ref [self] Self:
        """Declare a binding for the `__init__` method of the type."""

        @always_inline
        fn raising_wrapper[
            init_func: fn (out t: T, args: PythonObject, kwargs: PythonObject)
        ](out t: T, args: PythonObject, kwargs: PythonObject) raises:
            t = init_func(args, kwargs)

        return self.def_py_init[raising_wrapper[init_func]]()

    fn def_py_init[
        T: Movable, //,
        init_func: fn (out T, args: PythonObject, kwargs: PythonObject) raises,
    ](mut self) raises -> ref [self] Self:
        """Declare a binding for the `__init__` method of the type."""
        self._insert_slot(
            PyType_Slot.tp_new(_py_new_function_wrapper[T, init_func])
        )
        return self

    fn def_py_c_method[
        static_method: Bool = False
    ](
        mut self,
        method: PyCFunction,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PyObjectPtr signature for the
        type.

        Parameters:
            static_method: Whether the method is exposed as a staticmethod.
                Default is False. Note that CPython will pass a nullpointer for
                the first argument for static methods (i.e. instead of passing
                the self object). See [METH_STATIC](https://docs.python.org/3/c-api/structures.html#c.METH_STATIC).

        Args:
            method: The method to declare a binding for.
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """
        self.methods.append(
            PyMethodDef.function[static_method](method, method_name, docstring)
        )
        return self

    fn def_py_method[
        method: PyFunction, static_method: Bool = False
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PyObject signature for the type.

        Parameters:
            method: The method to declare a binding for.
            static_method: Whether the method is exposed as a staticmethod.
                Default is False. Note that CPython will pass a nullpointer for
                the first argument for static methods (i.e. instead of passing
                the self object). See [METH_STATIC](https://docs.python.org/3/c-api/structures.html#c.METH_STATIC).

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        return self.def_py_c_method[static_method](
            _py_c_function_wrapper[method], method_name, docstring
        )

    fn def_py_method[
        method: PyFunctionRaising, static_method: Bool = False
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PyObject signature for the type.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        return self.def_py_c_method[static_method](
            _py_c_function_wrapper[method], method_name, docstring
        )

    fn def_method[
        method_type: AnyTrivialRegType, //,
        method: PyObjectFunction[method_type, has_self=True],
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a method with PythonObject signature for the
        type.

        These signatures can have any number of positional PythonObject
        arguments up to 6 (including self), can optionally return a
        PythonObject, and can raise.

        Example signature types:
        ```mojo
        alias F1 = fn (mut PythonObject) raises -> PythonObject
        alias F2 = fn (mut PythonObject, PythonObject) -> PythonObject
        alias F3 = fn (mut PythonObject, PythonObject, mut PythonObject)
        ```

        Parameters:
            method_type: The type of the method to declare a binding for.
            method: The method to declare a binding for. Users can pass their
                function directly, and it will be implicitly converted to a
                PyObjectFunction if and only if its signature is supported.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut py_args: PythonObject
        ) raises -> PythonObject:
            return method._call_method(py_self, py_args)

        return self.def_py_method[wrapper](method_name, docstring)

    fn def_staticmethod[
        method_type: AnyTrivialRegType, //,
        method: PyObjectFunction[method_type, has_self=False],
    ](
        mut self: Self,
        method_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> ref [self] Self:
        """Declare a binding for a staticmethod with PythonObject signature for
        the type.

        These signatures can have any number of positional PythonObject
        arguments up to 6, can optionally return a PythonObject, and can raise.

        Example signature types:
        ```mojo
        alias F1 = fn (mut PythonObject) raises -> PythonObject
        alias F2 = fn (mut PythonObject, PythonObject) -> PythonObject
        alias F3 = fn (mut PythonObject, PythonObject, mut PythonObject)
        ```

        Parameters:
            method_type: The type of the method to declare a binding for.
            method: The method to declare a binding for. Users can pass their
                function directly, and it will be implicitly converted to a
                PyObjectFunction if and only if its signature is supported.

        Args:
            method_name: The name with which the method will be exposed on the
                type.
            docstring: The docstring for the method of the type.

        Returns:
            The builder with the method binding declared.
        """

        @always_inline
        fn wrapper(
            mut py_self: PythonObject, mut py_args: PythonObject
        ) raises -> PythonObject:
            # CPython will always pass a null pointer for the self argument for
            # static methods.
            return method._call_func(py_args)

        return self.def_py_method[wrapper, static_method=True](
            method_name, docstring
        )


# ===-----------------------------------------------------------------------===#
# PyCFunction Wrappers
# ===-----------------------------------------------------------------------===#


fn _py_new_function_nonregistered(
    subtype: PyTypeObjectPtr, args_ptr: PyObjectPtr, kwargs_ptr: PyObjectPtr
) -> PyObjectPtr:
    var cpython = Python().cpython()
    var error_type = cpython.get_error_global("PyExc_TypeError")
    cpython.PyErr_SetString(
        error_type,
        "No initializer registered for this type. Use def_py_init() or"
        " def_init_defaultable() to register an initializer.".unsafe_cstr_ptr(),
    )
    return {}


fn _py_new_function_wrapper[
    T: Movable,
    user_func: fn (out T, args: PythonObject, kwargs: PythonObject) raises,
](
    subtype: PyTypeObjectPtr, args_ptr: PyObjectPtr, kwargs_ptr: PyObjectPtr
) -> PyObjectPtr:
    """Wrapper function that adapts a Mojo `PyInitFunction` to be callable from
    Python.
    """

    var kwargs = PythonObject(from_owned_ptr=kwargs_ptr)
    var args = PythonObject(from_owned_ptr=args_ptr)

    __disable_del args
    __disable_del kwargs

    var cpython = Python().cpython()

    try:

        fn create_func(out self: T) raises capturing:
            self = user_func(args, kwargs)

        return _unsafe_alloc_init[create_func](subtype).steal_data()

    except e:
        # TODO(MSTDL-933): Add custom 'MojoError' type, and raise it here.
        var error_type = cpython.get_error_global("PyExc_ValueError")
        cpython.PyErr_SetString(error_type, e.unsafe_cstr_ptr())
        return {}


fn _py_c_function_wrapper[
    user_func: PyFunction
](py_self_ptr: PyObjectPtr, args_ptr: PyObjectPtr) -> PyObjectPtr:
    """Wrapper function that adapts a Mojo `PyFunction` to be callable from
    Python.

    This function creates a bridge between Python's C API calling convention
    and Mojo's `PyFunction` signature. It handles the conversion of raw Python
    object pointers to typed Mojo objects, calls the user function, and
    properly manages object lifetimes to prevent reference counting issues.

    The instantiated type of this generic function is a `PyCFunction`,
    suitable for being called from Python's C extension mechanism.

    Parameters:
        user_func: The Mojo function to wrap. Must have the `PyFunction`
            signature.

    Args:
        py_self_ptr: Pointer to the Python object representing 'self' in the
            method call. This is borrowed from the caller.
        args_ptr: Pointer to a Python tuple containing the positional arguments
            passed to the function. This is borrowed from the caller.

    Returns:
        A new Python object pointer containing the result of the user function,
        or a null pointer if an error occurred during execution.

    Note:
        This function carefully manages object ownership according to Python's
        reference counting rules. The input arguments are borrowed references
        that must not be decremented, while the return value is a new reference
        that the caller will own.
    """

    #   > When a C function is called from Python, it borrows references to its
    #   > arguments from the caller. The caller owns a reference to the object,
    #   > so the read-only reference's lifetime is guaranteed until the function
    #   > returns. Only when such a read-only reference must be stored or passed
    #   > on, it must be turned into an owned reference by calling Py_INCREF().
    #   >
    #   >  -- https://docs.python.org/3/extending/extending.html#ownership-rules

    # SAFETY:
    #   Here we illegally (but carefully) construct _owned_ `PythonObject`
    #   values from the read-only object reference arguments. We are careful
    #   down below to prevent the destructor for these objects from running
    #   so that we do not illegally decrement the reference count of these
    #   objects we do not own.
    #
    #   This is valid to do, because these are passed using the `read-only`
    #   argument convention to `user_func`, so logically they are treated
    #   as Python read-only references.
    var py_self = PythonObject(from_owned_ptr=py_self_ptr)
    var args = PythonObject(from_owned_ptr=args_ptr)

    # SAFETY:
    #   Call the user provided function, and take ownership of the
    #   PyObjectPtr of the returned PythonObject.
    var result = user_func(py_self, args).steal_data()

    # Do not destroy the provided PyObjectPtr arguments, since they
    # actually have ownership of the underlying object.
    __disable_del py_self

    # SAFETY:
    #   Prevent `args` from being destroyed, since we don't own it.
    __disable_del args
    return result


# Wrap a `raises` function
fn _py_c_function_wrapper[
    user_func: PyFunctionRaising
](py_self_ptr: PyObjectPtr, py_args_ptr: PyObjectPtr) -> PyObjectPtr:
    """Create a Python C API compatible wrapper for a Mojo function that can
    raise exceptions.

    This function wraps a Mojo function that follows the `PyFunctionRaising`
    signature (can raise exceptions) and makes it compatible with Python's C API
    calling convention.

    Parameters:
        user_func: The Mojo function to wrap. Must have the `PyFunctionRaising`
            signature.

    Args:
        py_self_ptr: Pointer to the Python object representing 'self' (borrowed
            reference).
        py_args_ptr: Pointer to a Python tuple containing the function arguments
            (borrowed reference).

    Returns:
        A new Python object pointer containing the function result, or NULL if
        an exception occurred. The caller takes ownership of the returned
        reference.
    """

    fn wrapper(
        mut py_self: PythonObject, mut args: PythonObject
    ) -> PythonObject:
        var cpython = Python().cpython()

        with GILAcquired(cpython):
            try:
                return user_func(py_self, args)
            except e:
                # TODO(MSTDL-933): Add custom 'MojoError' type, and raise it here.
                var error_type = cpython.get_error_global("PyExc_Exception")

                cpython.PyErr_SetString(
                    error_type,
                    e.unsafe_cstr_ptr(),
                )

                # Return a NULL `PyObject*`.
                return PythonObject(from_owned_ptr=PyObjectPtr())

    # TODO:
    #   Does this lead to multiple levels of indirect function calls for
    #   `raises` functions? Could we fix that by marking `wrapper` here as
    #   `@always_inline`?
    # Call the non-`raises` overload of `_py_c_function_wrapper`.
    return _py_c_function_wrapper[wrapper](py_self_ptr, py_args_ptr)


# ===-----------------------------------------------------------------------===#
# Utilities for building Python bindings
# ===-----------------------------------------------------------------------===#


fn check_arguments_arity(
    arity: Int,
    args: PythonObject,
) raises:
    """Validate that the provided arguments match the expected function arity.

    This function checks if the number of arguments in the provided tuple object
    matches the expected arity for a function call. If the counts don't match,
    it raises a descriptive error message similar to Python's built-in TypeError
    messages.

    Args:
        arity: The expected number of arguments for the function.
        args: A tuple containing the actual arguments passed to the function.

    Raises:
        Error: If the argument count doesn't match the expected arity. The error
               message follows Python's convention for TypeError messages,
               indicating whether too few or too many arguments were provided.
    """
    # TODO: try to extract the current function name from cpython
    return check_arguments_arity(arity, args, "<mojo function>")


fn check_arguments_arity(
    arity: Int,
    args: PythonObject,
    func_name: StringSlice,
) raises:
    """Validate that the provided arguments match the expected function arity.

    This function checks if the number of arguments in the provided tuple object
    matches the expected arity for a function call. If the counts don't match,
    it raises a descriptive error message similar to Python's built-in TypeError
    messages.

    Args:
        arity: The expected number of arguments for the function.
        args: A tuple containing the actual arguments passed to the function.
        func_name: The name of the function being called, used in error messages
                  to provide better debugging information.

    Raises:
        Error: If the argument count doesn't match the expected arity. The error
               message follows Python's convention for TypeError messages,
               indicating whether too few or too many arguments were provided,
               along with the specific function name.
    """

    var arg_count = len(args)

    # The error messages raised below are intended to be similar to the
    # equivalent errors in Python.
    if arg_count != arity:
        if arg_count < arity:
            var missing_arg_count = arity - arg_count

            raise Error(
                String.format(
                    "TypeError: {}() missing {} required positional {}",
                    func_name,
                    missing_arg_count,
                    _pluralize(missing_arg_count, "argument", "arguments"),
                )
            )
        else:
            raise Error(
                String.format(
                    "TypeError: {}() takes {} positional {} but {} were given",
                    func_name,
                    arity,
                    _pluralize(arity, "argument", "arguments"),
                    arg_count,
                )
            )


fn check_and_get_arg[
    T: AnyType
](
    func_name: StaticString, py_args: PythonObject, index: Int
) raises -> UnsafePointer[T]:
    """Get the argument at the given index and downcast it to a given Mojo type.

    Args:
        func_name: The name of the function referenced in the error message if
            the downcast fails.
        py_args: The Python tuple object containing the arguments.
        index: The index of the argument.

    Returns:
        A pointer to the Mojo value contained in the argument.

    Raises:
        If the argument cannot be downcast to the given type.
    """
    return py_args[index].downcast_value_ptr[T](func=func_name)


fn _try_convert_arg[
    T: ConvertibleFromPython
](
    func_name: StringSlice, py_args: PythonObject, argidx: Int, out result: T
) raises:
    try:
        result = T(py_args[argidx])
    except convert_err:
        raise Error(
            String.format(
                (
                    "TypeError: {}() expected argument at position {} to be"
                    " instance of (or convertible to) Mojo '{}'; got '{}'."
                    " (Note: attempted conversion failed due to: {})"
                ),
                func_name,
                argidx,
                get_type_name[T](),
                _get_type_name(py_args[argidx]),
                convert_err,
            )
        )


# NOTE:
#   @always_inline is needed so that the stack_allocation() that appears in
#   the definition below is valid in the _callers_ stack frame, effectively
#   allowing us to "return" a pointer to stack-allocated data from this
#   function.
@always_inline
fn check_and_get_or_convert_arg[
    T: ConvertibleFromPython
](
    func_name: StaticString, py_args: PythonObject, index: Int
) raises -> UnsafePointer[T]:
    """Get the argument at the given index and convert it to a given Mojo type.

    If the argument cannot be directly downcast to the given type, it will be
    converted to it.

    Args:
        func_name: The name of the function referenced in the error message if
            the downcast fails.
        py_args: The Python tuple object containing the arguments.
        index: The index of the argument.

    Returns:
        A pointer to the Mojo value contained in or converted from the argument.

    Raises:
        If the argument cannot be downcast or converted to the given type.
    """

    # Stack space to hold a converted value for this argument, if needed.
    var converted_arg_ptr: UnsafePointer[T] = stack_allocation[1, T]()

    try:
        return check_and_get_arg[T](func_name, py_args, index)
    except e:
        converted_arg_ptr.init_pointee_move(
            _try_convert_arg[T](
                func_name,
                py_args,
                index,
            )
        )
        # Return a pointer to stack data. Only valid because this function is
        # @always_inline.
        return converted_arg_ptr


fn _get_type_name(obj: PythonObject) raises -> String:
    var cpython = Python().cpython()

    var actual_type = cpython.Py_TYPE(obj.unsafe_as_py_object_ptr())
    var actual_type_name = PythonObject(
        from_owned_ptr=cpython.PyType_GetName(actual_type)
    )

    return String(actual_type_name)


fn _pluralize(
    count: Int,
    singular: StaticString,
    plural: StaticString,
) -> StaticString:
    return singular if count == 1 else plural
