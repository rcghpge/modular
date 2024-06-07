# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from max.graph import Graph
from max.engine._utils import call_dylib_func
from sys.ffi import _get_global_or_null
from .anymemory import AnyMemory
from .device import _CDevice
from tensor import TensorSpec


@value
@register_passable("trivial")
struct _CCompiledGraph:
    var ptr: UnsafePointer[NoneType]


@value
@register_passable("trivial")
struct _CExecutableGraph:
    var ptr: UnsafePointer[NoneType]


struct ExecutableGraph:
    """Represents a compiled graph loaded to device and ready for execution."""

    var _impl: _CExecutableGraph
    var _device: Device

    fn __init__(inout self, impl: _CExecutableGraph, device: Device):
        self._impl = impl
        self._device = device

    fn _steal_device_memory_ptr(
        self, inout memory: AnyMemory
    ) raises -> DTypePointer[DType.invalid]:
        var new = AnyMemory()
        var tmp = memory^
        memory = new^
        var ptr = tmp^.device_memory()._steal_impl_ptr()
        return ptr

    fn _add_to_output_list(
        self,
        list: UnsafePointer[NoneType],
        device_memory_ptr: DTypePointer[DType.invalid],
        spec_ptr: UnsafePointer[NoneType],
    ):
        var spec = spec_ptr.bitcast[TensorSpec]()[]
        var device_memory = DeviceMemory(device_memory_ptr, spec^, self._device)
        var typed_list = list.bitcast[List[AnyMemory]]()
        typed_list[].append(device_memory^)

    fn execute(self, owned *inputs: AnyMemory) raises -> List[AnyMemory]:
        """Execute the graph with given inputs.

        Args:
            inputs: Inputs to the graph. Inputs to the graph. Inputs' memory is
                    expected to be on the device for which the graph was
                    compiled.
        Returns:
            Execution output. This will be in the device annotated in graph
            output. If there is no annotation it's considered to be in CPU.
        """
        # Collect the C pointers of inputs to pass to C API.
        var inputs_impl = List[DTypePointer[DType.invalid]]()
        for input in inputs:
            inputs_impl.append(self._steal_device_memory_ptr(input[]))

        alias execute_func_name = "M_executeGraph"

        # We pass a callback function to C API with address to output list
        # The C API will call the callback to fill the list.
        # TODO: We can reserve this in advance after adding inspection APIs
        # to compiled graph.
        var output_list = List[AnyMemory]()
        var output_list_address = UnsafePointer.address_of(output_list)

        var execute_func = self._device.lib.get_handle().get_function[
            fn (
                _CExecutableGraph,
                UnsafePointer[DTypePointer[DType.invalid]],
                Int,
                __type_of(Self._add_to_output_list),
                UnsafePointer[Self],
                UnsafePointer[NoneType],
            ) -> Int
        ](execute_func_name)
        var output_count = execute_func(
            self._impl,
            inputs_impl.unsafe_ptr(),
            len(inputs_impl),
            Self._add_to_output_list,
            UnsafePointer.address_of(self),
            output_list_address.bitcast[NoneType](),
        )

        if len(output_list) != output_count:
            raise "internal error: mismatch on output count during ffi"

        # Make sure inputs are alive
        _ = inputs_impl^
        return output_list

    fn __del__(owned self):
        var lib = self._device.lib
        call_dylib_func(lib.get_handle(), "M_destroyGraph", self._impl)
        _ = self._device^


struct CompiledGraph:
    """Represents graph that is compiled to a device architecture."""

    var _impl: _CCompiledGraph
    var _device: Device

    fn __init__(inout self, impl: _CCompiledGraph, owned device: Device):
        self._impl = impl
        self._device = device^

    fn load(self) -> ExecutableGraph:
        """Load and initialize compiled graph. This will run the setup function
        of graph which includes initializing constants, loading constants into
        device of choice etc.

        Returns:
            Model ready for execution.
        """
        var executable_ptr = call_dylib_func[_CExecutableGraph](
            self._device.lib.get_handle(), "M_loadGraph", self._impl
        )
        return ExecutableGraph(executable_ptr, self._device)

    fn __del__(owned self):
        var lib = self._device.lib
        call_dylib_func(lib.get_handle(), "M_destroyCompiledGraph", self._impl)
        _ = self._device^


fn compile_graph(graph: Graph, device: Device) -> CompiledGraph:
    """Compiles graph to the given device.

    Args:
        graph: Graph to be compiled.
        device: Device the architecture to which graph needs to be compiled.

    Returns:
        Compiled graph ready to be loaded and executed.
    """
    # Graph compiler shares the context with Mojo. This context
    # contains LLCL Runtime, Telemetery etc.
    var max_context = _get_global_or_null["MaxContext"]().address
    var compiled_ptr = call_dylib_func[_CCompiledGraph](
        device.lib.get_handle(),
        "M_compileGraph",
        graph._module().c.ptr,
        max_context,
    )
    return CompiledGraph(compiled_ptr, device)
