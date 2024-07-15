# MAX SDK Python interop

This documents Python interoperability in the MAX API layers.

## Graph API

### Bindings

In the MAX graph Python API, the general design is that max/graph/core.cpp will
replace the role of _c.mojo.
Furthermore, the Python API uses upstream MLIR Python's type casting so that
methods defined on max.graph.core can take MLIR C API types directly.

Take the following example of creating an MLIR DialectRegistry type from the
MLIR Python API, then passing that DialectRegistry to the MAX graph C API.

First, from Python, call:

```python
import max.graph.core as _c
from max.graph import mlir

registry = mlir.ir.DialectRegistry()
_c.load_modular_dialects(registry._CAPIPtr)
```

From max/graph/core.cpp register a method that takes the C API type
`MlirDialectRegistry`:

```c++
// SDK/include/SDK/GraphAPI/python/Bindings.h
void load_modular_dialects(MlirDialectRegistry registry) {
  MAXG_loadModularDialects(registry);
}

// SDK/lib/GraphAPI/python/bindings/max/graph/core.cpp
PYBIND11_MODULE(core, m) {
  /// Dialect registration.
  m.def("load_modular_dialects", &Python::load_modular_dialects);
}
```

### Package architecture

The relevant target definitions are:

- `//SDK:GraphPython`
  Contains the C++ bindings logic defined in C++
  files under SDK/lib/GraphAPI/python/lib.
  Currently these are just calls into the MAX graph C API.
  The target for the corresponding C++ sources is
  `//SDK/lib/GraphAPI/python:cpp_srcs`.

- `//SDK/lib/GraphAPI/python/max/graph:core`
  The pybind module shared library.
  This defines the `PyInit` function for `max.graph.core` and all the
  methods on the module via `PYBIND11_MODULE`.
  This is Python code defined via pybind11 in
  SDK/lib/GraphAPI/python/bindings/max/graph/core.cpp.
  The corresponding sources are
  `//SDK/lib/GraphAPI/python:cpp_bindings_srcs`.

- `//SDK/lib/GraphAPI/python:max_graph_editable`
  This defines the local editable installation target for the MAX graph
  Python API.
  See ModularFramework/docs/README.md for usage.
