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
