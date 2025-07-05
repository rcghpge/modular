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
"""Core graph primitives."""

from __future__ import annotations

import contextlib
import functools
import inspect
import itertools
import traceback
from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, cast

from max import mlir
from max._core import Attribute as _Attribute
from max._core import Block, OpBuilder, Operation
from max._core import Type as _Type
from max._core import Value as _Value
from max._core import graph as _graph
from max._core.dialects import builtin, kgen
from max._core.dialects import mo as _mo

# TODO(GEX-1846): Get rid of this include.
from max.engine import InferenceSession  # type: ignore
from max.mlir.dialects import mo
from max.support.paths import (
    _build_mojo_source_package,
    is_mojo_binary_package_path,
    is_mojo_source_package_path,
)

from .type import (
    BufferType,
    DeviceRef,
    SymbolicDim,
    TensorType,
    Type,
    _ChainType,
)
from .value import BufferValue, TensorValue, Value, _ChainValue
from .weight import Weight

CURRENT_GRAPH: ContextVar[Graph] = ContextVar("CURRENT_GRAPH")
_KERNEL_LIBRARY_PATHS_ATTR_NAME = "_kernel_library_paths"


class KernelLibrary:
    """Manages custom kernel libraries and operations for a graph.

    A kernel library provides access to custom operations and kernels that can
    be loaded from various sources including Mojo binary packages (``.mojopkg``)
    and Mojo source directories. The library handles verification and registration
    of custom operations within the MLIR context.
    """

    _context: mlir.Context
    _analysis: _graph.Analysis

    def __init__(self, context: mlir.Context, paths: list[Path] = []) -> None:
        # TODO(GEX-1846): This is a terrible workaround to initialize M::Context on the Graph API.
        # Get rid of this and properly setup the context instead.
        mock_session = InferenceSession()
        mock_session._impl.register_runtime_context(context)

        self._context = context
        self._analysis = _graph.Analysis(context, paths)

    def library_paths(self) -> list[Path]:
        """Returns the list of kernel library paths.

        Returns:
            A list of :obj:`Path` objects representing the currently loaded
            kernel library paths.
        """
        return self._analysis.library_paths

    def add_path(self, path: Path) -> None:
        """Adds a kernel library path to the analysis.

        Args:
            path: The :obj:`Path` to the kernel library to be added to the
                current analysis.
        """
        self._analysis.add_path(path)

    def load_paths(
        self, context: mlir.Context, custom_extensions: Iterable[Path]
    ) -> None:
        """Loads custom operations from provided library paths.

        Performs additional "smart" library loading logic for custom operation
        libraries in additional formats. The loading logic supports the
        following formats:

        - Compiled Mojo binary packages with ``.mojopkg`` extension
        - Mojo source directory with custom operations

        The loaded libraries are added to the current kernel library.

        Args:
            context: The MLIR context for loading MLIR operations.
            custom_extensions: The file paths to the custom operation libraries.
        """
        with context:
            for ext_path in custom_extensions:
                if is_mojo_binary_package_path(ext_path):
                    self.add_path(ext_path)
                elif is_mojo_source_package_path(ext_path):
                    # Builds the source directory into a .mojopkg file.
                    self.add_path(_build_mojo_source_package(ext_path))
                else:
                    raise ValueError(
                        "Path provided as custom extension to Graph must be a "
                        + f"Mojo source or binary package: {ext_path}"
                    )

    def __getitem__(self, kernel: str):
        if kernel not in self._analysis.symbol_names:
            raise KeyError(kernel)
        return self._analysis.kernel(kernel)

    def __contains__(self, kernel: str) -> bool:
        return kernel in self._analysis.symbol_names

    def __iter__(self):
        yield from sorted(self._analysis.symbol_names)

    def verify_custom_op(self, custom_op: mlir.Operation) -> None:
        """Verifies that a custom operation is valid within the current context.

        Args:
            custom_op: The :obj:`mlir.Operation` to be verified against the
                current kernel library analysis.
        """
        self._analysis.verify_custom_op(custom_op)


# From https://stackoverflow.com/a/76301341
class _classproperty:
    def __init__(self, func) -> None:
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


@dataclass(frozen=True)
class _GraphWeight:
    weight: Weight
    value: TensorValue


def _location(ignore_frames: int = 1):
    """Creates an MLIR Location with the current Python call stack."""
    if not mlir.Context.current:
        raise RuntimeError("Can't create location: No MLIR context active")

    # Extract the stack into summaries
    # - Avoids reference cycles
    # - Doesn't keep references to closures

    # Always remove at least _location
    tb = traceback.extract_stack()[: -(ignore_frames + 1)]
    if not tb:
        return mlir.Location.unknown()

    return _graph.frame_loc(mlir.Context.current, tb)


def _to_mlir(o):
    # Convert args from instances of Python graph-api Value() to mlir.Value
    if hasattr(o, "to_mlir"):
        return o.to_mlir()
    elif isinstance(o, (list, tuple)):
        return type(o)(_to_mlir(ov) for ov in o)
    elif isinstance(o, dict):
        return {k: _to_mlir(v) for k, v in o.items()}
    return o


def _set_output_param_decls(op: Operation, params: dict[str, None]):
    # Interfaces don't yet support isinstance checks, so this is a cheap proxy.
    # - nanobind doesn't allow custom metaclasses, but __instancecheck__
    #   must be defined on a metaclass
    # - Interfaces are protocols even though we know when they are explicitly
    #   implemented so that attrs/types/ops may implement them without declaring
    #   it in the stub files
    # - it's not trivial to define our own `isa`-like check. Such a check needs
    #   to name the template parameter for `mlir::isa<T>` explicitly in C++, but
    #   if `isa` is defined as a staticmethod it interferes with protocol type
    #   checking.
    if not hasattr(op, "output_param_decls"):
        return
    op: Operation & _mo.ParamDeclarationInterface  # type: ignore
    # Add symbolic dims of tensor results to the list of graph params and
    # declared output params of the op
    # Use a dict as an ordered set for new param decls. Maps keys to None.
    result_parameters = dict.fromkeys(
        itertools.chain.from_iterable(
            value.type.parameters
            for result in op.results
            if isinstance(
                value := Value.from_mlir(result), (TensorValue, BufferValue)
            )
        )
    )
    names = [parameter.name for parameter in result_parameters]
    # Track any newly declared parameters.
    if new_params := dict.fromkeys(names - params.keys()):
        params.update(new_params)
        si64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
        # We can't overload the setter yet, so the interface annotation is wrong
        # TODO(MAXPLAT-306): See https://github.com/wjakob/nanobind/discussions/1063
        op.output_param_decls = kgen.ParamDeclArrayAttr(
            [kgen.ParamDeclAttr(name, si64) for name in new_params]
        )


class Graph:
    """Represents a single MAX graph.

    A `Graph` is a callable routine in MAX Engine. Like functions, graphs have a
    name and signature. Unlike a function, which follows an imperative
    programming model, a `Graph` follows a dataflow programming model, using
    lazily-executed, parallel operations instead of sequential instructions.

    When you instantiate a graph, you must specify the input shapes as one or
    more :obj:`TensorType` values. Then, build a sequence of ops and set the
    graph output with :obj:`output()`. For example:

    .. code-block:: python

        from dataclasses import dataclass

        import numpy as np
        from max.dtype import DType
        from max.graph import Graph, TensorType, TensorValue, ops

        @dataclass
        class Linear:
            weight: np.ndarray
            bias: np.ndarray

            def __call__(self, x: TensorValue) -> TensorValue:
                weight_tensor = ops.constant(self.weight, dtype=DType.float32, device=DeviceRef.CPU())
                bias_tensor = ops.constant(self.bias, dtype=DType.float32, device=DeviceRef.CPU())
                return ops.matmul(x, weight_tensor) + bias_tensor

        linear_graph = Graph(
            "linear",
            Linear(np.ones((2, 2)), np.ones((2,))),
            input_types=[TensorType(DType.float32, (2,))]
        )

    You can't call a `Graph` directly from Python. You must compile it and
    execute it with MAX Engine. For more detail, see the tutorial about how to
    [build a graph with MAX
    Graph](/max/tutorials/get-started-with-max-graph-in-python).

    When creating a graph, a global sequence of chains is initialized and stored
    in Graph._current_chain. Every side-effecting op, e.g. buffer_load,
    store_buffer, load_slice_buffer, store_slice_buffer, will use the current
    chain to perform the op and and update Graph._current_chain with a new
    chain. Currently, the input/output chains for mutable ops can be used at
    most once. The goal of this design choice is to prevent data races.
    """

    # Use a dict rather than a set to keep params ordered.
    # This is to make IR generation deterministic for model IR cache hits.
    # Note that insertion order in built-in dict has been guaranteed since
    # Python 3.7.
    _params: dict[str, None]
    _mlir_op: mlir.Operation | mlir.OpView
    _graph_body: mlir.Block
    _module: mlir.Module
    _context_state: list
    _weights: dict[str, _GraphWeight]
    # A global sequence of chains that is updated by side-effecting ops.
    _current_chain: _ChainValue
    _current_block: mlir.Block
    _should_verify_ops: bool
    _has_chain_input: bool

    _kernel_library: KernelLibrary

    _subgraphs: dict[str, Graph] = {}

    def __init__(
        self,
        name: str,
        forward: Optional[Callable] = None,
        input_types: Iterable[Type] = (),
        path: Optional[Path] = None,
        *args,
        custom_extensions: list[Path] = [],
        context: Optional[mlir.Context] = None,
        kernel_library: Optional[KernelLibrary] = None,
        module: Optional[mlir.Module] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            name: A name for the graph.
            forward: The sequence of graph ops for the forward pass (inference).
            input_types: The data type(s) for the input tensor(s).
            path: The path to a saved graph (internal use only).
            custom_extensions: The extensions to load for the model.
              Supports paths to `.mojopkg` or `.mojo` sources with custom ops.
        """
        self.name = name
        if path is not None:
            self._load_mlir(path)
            return

        self._params = dict.fromkeys(
            dim.name
            for t in input_types
            if isinstance(t, (TensorType, BufferType))
            for dim in t.shape
            if isinstance(dim, SymbolicDim)
        )
        self._context_state = []
        context = context or mlir.Context()
        self._should_verify_ops = True

        with context, _location() as loc:
            # Create the top level module op.
            self._module = module or mlir.Module.create()
            _module: builtin.ModuleOp = Operation._from_cmlir(  # type: ignore
                self._module.operation
            )
            builder = OpBuilder(_module.body.end)

            op = builder.create(_mo.GraphOp, loc)(
                name=name,
                input_types=[t.to_mlir() for t in input_types],
                result_types=[],
            )

            si64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
            # TODO(MAXPLAT-306): Type annotations are wrong here
            op.input_parameters = kgen.ParamDeclArrayAttr(  # type: ignore
                [kgen.ParamDeclAttr(p, si64) for p in self._params]
            )

            self._mlir_op = mlir.Operation._CAPICreate(op._CAPIPtr)  # type: ignore
            self._current_block = self._mlir_op.regions[0].blocks[0]
            self._graph_body = self._current_block

        self._weights = {}
        self._has_chain_input = False

        if self._graph_body.arguments:
            mlir_maybe_chain_value = _Value._from_cmlir(
                self._graph_body.arguments[-1]
            )
            if isinstance(mlir_maybe_chain_value.type, _mo.ChainType):
                self._has_chain_input = True
                self._current_chain = _ChainValue.from_mlir(
                    cast(_Value[_mo.ChainType], mlir_maybe_chain_value)
                )

        if not self._has_chain_input:
            self._current_chain = cast(
                _ChainValue, self._add_op(mo.chain_create, [])[0]
            )

        assert isinstance(self._current_chain, _ChainValue)

        # Initialize the kernel library and load custom extensions paths.
        self._kernel_library = kernel_library or KernelLibrary(context)
        self._import_kernels(custom_extensions)

        self._subgraphs = {}

        if forward is not None:
            # If the forward method was passed stage the graph directly in the
            # constructor.
            with self:
                result = forward(*self.inputs, *args, **kwargs)
                # Account for forward methods that return None, a single
                # output, or multiple outputs.
                outputs = (
                    ()
                    if result is None
                    else (result,)
                    if not isinstance(result, Iterable)
                    else result
                )
                self.output(*outputs)

    @functools.cached_property
    def inputs(self) -> Sequence[Value]:
        """The input values of the graph."""
        body_args = self._graph_body.arguments
        if body_args and self._has_chain_input:
            body_args = body_args[:-1]

        return tuple(
            Value.from_mlir(_Value._from_cmlir(arg))
            for arg in body_args  # type: ignore
        )

    @property
    def _context(self) -> mlir.Context:
        return self._mlir_op.context

    def add_subgraph(
        self,
        name: str,
        forward: Optional[Callable] = None,
        input_types: Iterable[Type] = (),
        path: Optional[Path] = None,
        custom_extensions: list[Path] = [],
    ) -> Graph:
        """Creates and adds a subgraph to the current graph.

        Creates a new :obj:`Graph` instance configured as a subgraph of the current
        graph. The subgraph inherits the parent graph's MLIR context, module, and
        symbolic parameters. A chain type is automatically appended to the input
        types to enable proper operation sequencing within the subgraph.

        The created subgraph is marked with special MLIR attributes to identify it
        as a subgraph and is registered in the parent graph's subgraph registry.

        Args:
            name: The name identifier for the subgraph.
            forward: The optional callable that defines the sequence of operations
                for the subgraph's forward pass. If provided, the subgraph will be
                built immediately using this callable.
            input_types: The data types for the subgraph's input tensors. A chain
                type will be automatically added to these input types.
            path: The optional path to a saved subgraph definition to load from
                disk instead of creating a new one.
            custom_extensions: The list of paths to custom operation libraries
                to load for the subgraph. Supports ``.mojopkg`` files and Mojo
                source directories.
        """
        subgraph = Graph(
            name=name,
            forward=forward,
            input_types=[*input_types, _ChainType()],
            path=path,
            # *args,
            custom_extensions=custom_extensions,
            context=self._context,
            module=self._module,
            # **kwargs,
        )
        subgraph._mlir_op.attributes["isSubgraph"] = mlir.UnitAttr.get()
        subgraph._mlir_op.attributes["inputParameters"] = (
            self._mlir_op.attributes["inputParameters"]
        )
        subgraph._params = dict.fromkeys(self._params)
        self._subgraphs[name] = subgraph
        return subgraph

    def _update_chain(self, new_chain: _ChainValue) -> None:
        self._current_chain = new_chain

    def __enter__(self) -> Graph:
        self._context_state.append(state := self._enter())
        return state.__enter__()

    def __exit__(self, *exc):
        self._context_state.pop().__exit__(*exc)

    @contextlib.contextmanager
    def _enter(self):
        token = CURRENT_GRAPH.set(self)
        try:
            with self._context:
                yield self
        finally:
            CURRENT_GRAPH.reset(token)

    @contextlib.contextmanager
    def local_weights_and_chain(self):
        """Creates a local scope for weights and chain state modifications.

        Provides a context manager that creates an isolated scope where the
        graph's weights dictionary and current chain state can be modified
        without affecting the parent scope. Upon entering the context, the
        current weights and chain state are saved. Any modifications made
        within the context are automatically reverted when exiting the context,
        restoring the original state.

        This is particularly useful for operations that need to temporarily
        modify graph state, such as building subgraphs or executing operations
        within isolated blocks where state changes should not persist.
        """
        weights = self._weights.copy()
        current_chain = self._current_chain
        try:
            yield
        finally:
            self._weights = weights
            self._current_chain = current_chain

    @contextlib.contextmanager
    def _block(self, block: mlir.Block):
        with self.local_weights_and_chain():
            current_block, self._current_block = self._current_block, block
            try:
                yield self._current_block
            finally:
                self._current_block = current_block

    @contextlib.contextmanager
    def _pause_verification(self):
        """Temporarily disable verification."""
        old_value = self._should_verify_ops
        try:
            self._should_verify_ops = False
            yield
        finally:
            self._should_verify_ops = old_value

    def _verify_op(self, op: mlir.Operation | mlir.OpView) -> None:
        if self._should_verify_ops:
            with self._capturing_mlir_diagnostics():
                op.verify()

    @contextlib.contextmanager
    def _capturing_mlir_diagnostics(self):
        diagnostics = []

        def handler(d) -> bool:
            diagnostics.append(str(d))
            return True

        # Temporarily hookup a handler to record diagnostics from mlir.
        # These are used to generate a better error message on failure.
        handle = self._context.attach_diagnostic_handler(handler)
        try:
            yield None
        except (mlir.MLIRError, ValueError) as e:  # type: ignore
            # MLIRError is raised from the MLIR Python bindings on MLIR
            # errors, however so is ValueError when operation create fails.
            # So catch both exception types and report diagnostics here.
            diags = "\n  ".join(diagnostics)
            raise ValueError(f"Diagnostics:\n    {diags}\n{e}") from None
        finally:
            handle.detach()

    @_classproperty
    def current(cls) -> Graph:
        """Gets the currently active graph from the execution context.

        Retrieves the :obj:`Graph` instance that is currently active within
        the execution context. The current graph is automatically set when
        entering a graph's context using a ``with`` statement or when the
        graph is being built. This provides access to the active graph from
        within operation definitions and other graph construction code.
        """
        try:
            current = CURRENT_GRAPH.get()
        except LookupError as exc:
            raise LookupError("No graph found") from exc
        assert current
        return current

    @property
    def _body(self) -> mlir.Block:
        return self._current_block

    def _add_op_generated(
        self, op_type: type[Operation], *args, **kwargs
    ) -> list[Value]:
        """Wrapper for clients that only require the op results."""
        with self._context, _location() as location:
            builder = OpBuilder(Block._from_cmlir(self._current_block).end)
            op = builder.create(op_type, location)(
                *_to_mlir(args), **_to_mlir(kwargs)
            )
            assert op.verify()
        _set_output_param_decls(op, self._params)
        return [Value.from_mlir(result) for result in op.results]

    def _add_op(self, op, *args, **kwargs) -> list[Value]:
        """Wrapper for clients that only require the op results."""
        results, _ = self._add_op_get_op_with_results(op, *args, **kwargs)
        return results

    def _add_op_get_op_with_results(
        self, op, *args, _ip: Optional[mlir.InsertionPoint] = None, **kwargs
    ) -> tuple[list[Value], mlir.OpView]:
        # Convert args from instances of Python graph-api Value() to mlir.Value
        def unwrap(arg):
            if isinstance(arg, Value):
                return mlir.Value._CAPICreate(arg._mlir_value._CAPIPtr)  # type: ignore
            elif isinstance(arg, Type):
                return mlir.Type._CAPICreate(arg.to_mlir()._CAPIPtr)
            elif isinstance(arg, (list, tuple)):
                return [unwrap(elem) for elem in arg]
            elif isinstance(arg, _Attribute):
                return mlir.Attribute._CAPICreate(arg._CAPIPtr)  # type: ignore
            elif isinstance(arg, _Type):
                return mlir.Type._CAPICreate(arg._CAPIPtr)  # type: ignore
            elif isinstance(arg, _Value):
                return mlir.Value._CAPICreate(arg._CAPIPtr)  # type: ignore
            else:
                return arg

        unwrapped_args = tuple(unwrap(arg) for arg in args)
        unwrapped_kwargs = {k: unwrap(arg) for k, arg in kwargs.items()}

        # Construct and insert an op in the body of the graph
        # Insertion point is where the op is to be created in the IR structure
        # location contains info about the source of the op (e.g. file, line)
        with _ip or mlir.InsertionPoint(self._body), _location():
            try:
                with self._capturing_mlir_diagnostics():
                    results = op(*unwrapped_args, **unwrapped_kwargs)

                    if _ip is None or _ip.ref_operation is None:
                        # Get the op we just staged, which is the last op in the body block.
                        ops = self._body.operations

                        staged_op = self._body.operations[len(ops) - 1]
                    else:
                        cur_block = _ip.block
                        ops = cur_block.operations
                        for idx, op in enumerate(ops):
                            if op == _ip.ref_operation:
                                staged_op = ops[idx - 1]
                                break
                        else:
                            assert False, (
                                "Could not find constructed operation in current block"
                            )
                    self._verify_op(staged_op)

            except (mlir.MLIRError, ValueError) as e:  # type: ignore
                # MLIRError is raised from the MLIR Python bindings on MLIR
                # errors, however so is ValueError when operation creation fails.
                # So catch both exception types here.
                mapped_args: dict[str, Any]
                try:
                    mapped_args = (
                        inspect.signature(op).bind(*args, **kwargs).arguments
                    )
                except TypeError:
                    mapped_args = {"args": list(args), **kwargs}
                raise ValueError(
                    f"Failed to create op '{op.__qualname__}':\nInputs:\n"
                    + "".join(
                        f"    {k} = {v!r}\n" for k, v in mapped_args.items()
                    )
                    + f"\n{e}"
                    # Intentionally suppress extra stack traces from max._mlir.
                ) from None

        _set_output_param_decls(Operation._from_cmlir(staged_op), self._params)
        if isinstance(results, (mlir.Operation, mlir.OpView)):
            return [], staged_op

        # Convert op results from  mlir.Value to instances of Value graph-api
        if isinstance(results, mlir.Value):
            results = [Value.from_mlir(_Value._from_cmlir(results))]
        else:
            results = [
                Value.from_mlir(_Value._from_cmlir(result))
                for result in results
            ]

        return results, staged_op

    def _build_block(
        self,
        block: mlir.Block,
        block_fn: Callable[[], Iterable[TensorValue] | TensorValue | None],
        block_terminator_op: mlir.Operation | mlir.OpView,
        block_name: str,
        expected_output_types: list[Type] | None,
    ) -> None:
        """Builds and verifies a block within the graph.

        Args:
            block: The MLIR block to build into
            block_fn: Callable that generates the block's operations and returns results
            block_terminator_op: Operation to terminate the block (e.g. mo.YieldOp)
            block_name: Name of the block for error reporting
            expected_output_types: List of expected output types for the block
            add_chain: Whether to append the current chain to block results

        Raises:
            ValueError: If the number of results doesn't match expected outputs
            ValueError: If any result type doesn't match the expected type

        Note:
            Manages the chain state automatically, restoring the parent chain after
            block construction. The chain is used to track operation ordering.

            It is the caller's responsibility to update the graph chain after
            the block is built.
        """
        with self._block(block), _location():
            expected_output_types = expected_output_types or []

            results = block_fn() or []

            results = (
                list(results) if isinstance(results, Iterable) else [results]
            )
            result_types = [result.type for result in results]
            if result_types != expected_output_types:
                raise TypeError(
                    f"Results don't match expected types: \n{result_types=}, \n{expected_output_types=}"
                )

            _ = self._add_op(
                block_terminator_op, results + [self._current_chain]
            )

    def output(self, *outputs: Value) -> None:
        """Sets the output nodes of the :obj:`Graph`."""
        # mo.output doesn't support infer_type
        graph_body_args = self._graph_body.arguments
        mlir_values = [o._mlir_value for o in outputs]
        if self._has_chain_input:
            mlir_values.append(self._current_chain._mlir_value)

        # We have a type mismatch now, these are MLIR types
        output_types = [value.type for value in mlir_values]

        # Need to set some more stuff.
        function_type = mlir.FunctionType.get(
            [_Value._from_cmlir(arg).type for arg in graph_body_args],  # type: ignore
            output_types,
        )
        signature = mlir.Type.parse(f"!kgen.generator<{function_type}>")
        self._mlir_op.attributes["signature"] = mlir.TypeAttr.get(signature)
        self._mlir_op.attributes["functionType"] = mlir.TypeAttr.get(
            function_type
        )

        self._add_op(mo.output, mlir_values)

        # Set the result_names metadata on the staged op, which is needed by
        # the engine for execution.
        # Note that result_names here needs to match kMgpModelResultNames.
        input_names = [f'"input{i}"' for i in range(len(self.inputs))]
        self._mlir_op.attributes["argument_names"] = mlir.Attribute.parse(
            f"[{', '.join(input_names)}]"
        )
        output_names = [f'"output{i}"' for i in range(len(output_types))]
        self._mlir_op.attributes["result_names"] = mlir.Attribute.parse(
            f"[{', '.join(output_names)}]"
        )

        self._subgraphs = {}
        # Outputting means the graph is complete. Verify the entire graph.
        try:
            with self._capturing_mlir_diagnostics():
                assert self._mlir_op.verify()
        except Exception as e:
            print(self)
            raise ValueError(
                "Graph failed to verify. Please file an issue. This should be"
                " impossible." + f"\n{e}"
            ) from None

    def _erase_output_if_present(self) -> None:
        terminator = self._body.operations[-1]
        if isinstance(terminator, mo.OutputOp):
            terminator.erase()

    @property
    def output_types(self) -> list[Type]:
        """View of the types of the graph output terminator."""
        terminator = self._body.operations[-1]
        terminator_operands = terminator.operands
        if terminator_operands and self._has_chain_input:
            terminator_operands = terminator_operands[:-1]

        if not isinstance(terminator, mo.OutputOp):
            raise TypeError("Graph not yet terminated by a call to output")
        return [
            Value.from_mlir(_Value._from_cmlir(v)).type
            for v in terminator_operands  # type: ignore
        ]

    def _load_mlir(self, path: Path) -> None:
        self._context_state = []
        with open(path) as f:
            with mlir.Context() as ctx, _location() as loc:
                # Create the top level module op.
                self._module = mlir.Module.create()
                with mlir.InsertionPoint(self._module.body):
                    self._module = self._module.parse(f.read(), ctx)
                    # Set the mo.graph op, which is the first operation in the
                    # module body block.
                    self._mlir_op = self._module.body.operations[0]

        # Initialize the Kernel Library
        kernels_paths = []
        if _KERNEL_LIBRARY_PATHS_ATTR_NAME in self._mlir_op.attributes:
            paths_attr = self._mlir_op.attributes[
                _KERNEL_LIBRARY_PATHS_ATTR_NAME
            ]
            if isinstance(paths_attr, mlir.ArrayAttr):
                kernels_paths = [Path(str(x)) for x in paths_attr]
        self._kernel_library = KernelLibrary(self._context, kernels_paths)

    def add_weight(
        # TODO(GEX-2121): Remove `force_initial_weight_on_host`
        self,
        weight: Weight,
        force_initial_weight_on_host: bool = True,
    ) -> TensorValue:
        """Adds a weight to the graph.

        If the weight is in the graph already, return the existing value.

        Args:
            weight: The weight to add to the graph.
            force_initial_weight_on_host: If true, then forces weights
                to initially be allocated on host before being moved to
                the indicated device. This is needed as a stop gap
                until we have a more fleshed out ownership model of
                external constants.

        Returns:
            A :obj:`TensorValue` that contains this weight.

        Raises:
            ValueError: If a weight with the same name already exists in the graph.
        """
        if graph_weight := self._weights.get(weight.name):
            if graph_weight.weight is weight:
                if force_initial_weight_on_host:
                    transferred_value = graph_weight.value.to(weight.device)
                    if transferred_value is not graph_weight.value:
                        assert isinstance(
                            transferred_value._mlir_value.owner, Operation
                        )
                        assert isinstance(
                            graph_weight.value._mlir_value.owner, Operation
                        )
                        transferred_value._mlir_value.owner.move_after(
                            graph_weight.value._mlir_value.owner
                        )
                    return transferred_value
                else:
                    return graph_weight.value
            else:
                raise ValueError(
                    f"Weight '{weight.name}' already exists in Graph {self}"
                )

        initial_device = (
            DeviceRef.CPU() if force_initial_weight_on_host else weight.device
        )

        tensor_type = TensorType(
            weight.dtype, weight.shape, device=initial_device
        ).to_mlir()
        weight_tensor = Graph.current._add_op(
            mo.constant_external,
            result=tensor_type,
            name=weight.name,
            align=(
                # Default to dtype alignment unless otherwise specified, for
                # example by checkpoint metadata.
                weight.align if weight.align is not None else weight.dtype.align
            ),
            device=initial_device.to_mlir(),
            is_placeholder=weight._placeholder,
            has_alias=weight._has_alias,
            _ip=mlir.InsertionPoint.at_block_begin(self._graph_body),
        )[0]

        const_external_op = weight_tensor._mlir_value.owner
        self._weights[weight.name] = _GraphWeight(weight, weight_tensor)
        if initial_device != weight.device:
            weight_tensor = weight_tensor.to(weight.device)
            assert isinstance(const_external_op, Operation)
            assert isinstance(weight_tensor._mlir_value.owner, Operation)
            weight_tensor._mlir_value.owner.move_after(const_external_op)
        return weight_tensor

    def __repr__(self) -> str:
        return str(self._mlir_op)

    def _import_kernels(self, paths: Iterable[Path]) -> None:
        with self._context:
            self._kernel_library.load_paths(self._context, paths)

            # Update the graph attribute for the library paths.
            self._mlir_op.attributes[_KERNEL_LIBRARY_PATHS_ATTR_NAME] = (
                mlir.ArrayAttr.get(
                    [
                        mlir.StringAttr.get(str(path), self._context)
                        for path in self._kernel_library.library_paths()
                    ]
                )
            )

    @property
    def kernel_libraries_paths(self) -> list[Path]:
        """Returns the list of extra kernel libraries paths for the custom ops."""

        return self._kernel_library.library_paths()
