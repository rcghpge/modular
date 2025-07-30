"""Public API accessors to reduce the number of load statements needed in BUILD.bazel files."""

load("@aspect_rules_py//py:defs.bzl", "py_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", _py_grpc_library = "py_grpc_library")
load("@rules_pkg//pkg:mappings.bzl", _strip_prefix = "strip_prefix")
load("@rules_proto//proto:defs.bzl", _proto_library = "proto_library")
load("//bazel/internal:binary_test.bzl", "binary_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:lit.bzl", _lit_tests = "lit_tests")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_binary.bzl", _modular_py_binary = "modular_py_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_binary.bzl", _mojo_binary = "mojo_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_filecheck_test.bzl", _mojo_filecheck_test = "mojo_filecheck_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_library.bzl", _mojo_library = "mojo_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test.bzl", _mojo_test = "mojo_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test_environment.bzl", _mojo_test_environment = "mojo_test_environment")  # buildifier: disable=bzl-visibility
load("//bazel/pip:pip_requirement.bzl", _requirement = "pip_requirement")

mojo_filecheck_test = _mojo_filecheck_test
mojo_test = _mojo_test
mojo_test_environment = _mojo_test_environment
proto_library = _proto_library
py_grpc_library = _py_grpc_library
requirement = _requirement
strip_prefix = _strip_prefix

# Deps that aren't open source so we need to remap to pull from the wheel instead.
_DEPS_FROM_WHEEL = [
    "//max",
    "//max/driver",
    "//max/dtype",
    "//max/engine",
    "//max/graph",
    "//max/interfaces",
    "//max/mojo",
    "//max/profiler",
    "//max/support",
    "//max:_core",
    "//max/_core_mojo",
]

def _is_internal_reference(dep):
    """Check if a dependency is an internal reference."""
    return dep.startswith(("//GenericML", "//KGEN/", "//Kernels/", "//SDK/integration-test/pipelines/python", "//SDK/lib/API/python/max/mlir", "//SDK:max"))

def _has_internal_reference(deps):
    return any([_is_internal_reference(dep) for dep in deps])

def _remove_internal_data(data):
    # TODO: This is a hack that potentially breaks things at runtime
    if type(data) != type([]):
        return []
    return [d for d in data if not _is_internal_reference(d)]

def _rewrite_deps(deps):
    """Rewrite dependencies to use the open-source package names, or to come from the wheel."""
    new_deps = []
    for dep in deps:
        if dep.startswith("//SDK/lib/API/python/"):
            replaced_dep = dep.replace("//SDK/lib/API/python/", "//")
            if replaced_dep in _DEPS_FROM_WHEEL:
                replaced_dep = "@modular_wheel//:wheel"
            if replaced_dep not in new_deps:
                new_deps.append(replaced_dep)
        else:
            new_deps.append(dep)
    return new_deps

def modular_py_library(
        data = [],
        deps = [],
        visibility = ["//visibility:public"],
        **kwargs):
    if _has_internal_reference(deps):
        return

    py_library(
        data = _remove_internal_data(data),
        deps = _rewrite_deps(deps),
        visibility = visibility,
        **kwargs
    )

# buildifier: disable=function-docstring
def modular_py_binary(
        name,
        deps = [],
        data = [],
        env = {},
        **kwargs):
    if name == "pipelines":
        # TODO: Fix this hack, there is a layering issue with what is open source right now
        deps.append("//max/entrypoints:mojo")
        data = []
        env = {}

    # TODO: There is some data we can fix by pulling from the wheel
    if _has_internal_reference(deps) or _has_internal_reference(data):
        return

    _modular_py_binary(
        name = name,
        data = data,
        env = env,
        deps = _rewrite_deps(deps),
        **kwargs
    )

def mojo_library(deps = [], **kwargs):
    if _has_internal_reference(deps):
        return

    _mojo_library(
        deps = deps,
        **kwargs
    )

def mojo_binary(
        data = [],
        deps = [],
        **kwargs):
    if _has_internal_reference(deps) or _has_internal_reference(data):
        return
    _mojo_binary(
        data = data,
        deps = deps,
        **kwargs
    )

# buildifier: disable=function-docstring
def modular_run_binary_test(name, external_noop = False, **kwargs):
    if external_noop:
        return
    if name.endswith(".example-test"):
        return  # TODO: Fix custom_ops python examples
    binary_test(
        name = name,
        **kwargs
    )

def lit_tests(tools = [], data = [], **kwargs):
    if _has_internal_reference(data) or _has_internal_reference(tools):
        return

    _lit_tests(
        data = data,
        tools = tools,
        **kwargs
    )

def _noop(**_kwargs):
    pass

modular_py_test = _noop
mojo_kgen_lib = _noop
pkg_attributes = _noop
pkg_filegroup = _noop
pkg_files = _noop
