"""`modular_eager_py_test`: `modular_py_test` + build-warmed eager-GC cache wiring.
"""

load("//bazel:api.bzl", "modular_py_test")

_WARM_PKG = "//max/tests/integration/xarch_warm"

# One warm per lane: the GPU lanes' warm holds the CPU and accelerator slots, and
# every other lane's only the CPU slot.
_WARM_DATA = select({
    "//:b200_gpu": [_WARM_PKG + ":warm_gpu"],
    "//:mi355_gpu": [_WARM_PKG + ":warm_gpu"],
    "//conditions:default": [_WARM_PKG + ":warm_cpu"],
})

# `XARCH_WARM_RLOCATIONS` must name the same target `_WARM_DATA` provides; the
# plugin resolves it into `MAX_EAGER_WARM_STORE`.
_GPU_RLOC = {"XARCH_WARM_RLOCATIONS": "$(rlocationpaths {}:warm_gpu)".format(_WARM_PKG)}
_CPU_RLOC = {"XARCH_WARM_RLOCATIONS": "$(rlocationpaths {}:warm_cpu)".format(_WARM_PKG)}
_WARM_RLOCATION_ENV = select({
    "//:b200_gpu": _GPU_RLOC,
    "//:mi355_gpu": _GPU_RLOC,
    "//conditions:default": _CPU_RLOC,
})

_WARM_ENV = {
    # Load the GC matrix from the warm once at import, not lazily per dispatch.
    "MAX_EAGER_OP_PRECOMPILE": "1",
}

_HIDE_ACCELERATORS_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
}

# Warm-adoption pytest plugin.
_PLUGIN_LABEL = _WARM_PKG + ":eager_warm_plugin"
_PLUGIN_MODULE = "eager_warm_plugin"

def modular_eager_py_test(
        name,
        srcs,
        cpu_only = False,
        data = [],
        env = {},
        deps = [],
        ignore_extra_deps = [],
        args = [],
        **kwargs):
    """A `modular_py_test` that loads the build-time eager-GC warm store.

    Every eager op family loads its compiled sweep from the warm, and a warm
    that lacks a slot this test's devices need fails the test rather than
    compiling.

    Args:
        name: Test target name.
        srcs: Test sources.
        cpu_only: For tests that do not require a GPU. More of a hack to force
          the test to run with `CUDA_VISIBLE_DEVICES=""` and
          `HIP_VISIBLE_DEVICES=""` which allows the test to use cached ops.
        data: Extra runtime deps; merged with the warm store dirs.
        env: Extra env vars; merged over the warm env (caller wins on conflict).
        deps: py deps; merged with the warm-adopt plugin lib.
        ignore_extra_deps: Forwarded; the plugin lib is appended (it is loaded
            via `-p`, never statically imported, so pydeps must not flag it).
        args: Extra pytest args; the `-p` plugin registration is prepended.
        **kwargs: Forwarded verbatim to `modular_py_test` (`gpu_constraints`,
            `size`, `tags`, `per_test_shard_count`, ...).
    """
    if cpu_only:
        warm_data = [_WARM_PKG + ":warm_cpu"]
        warm_env = _CPU_RLOC | _HIDE_ACCELERATORS_ENV
    else:
        warm_data = _WARM_DATA
        warm_env = _WARM_RLOCATION_ENV
    modular_py_test(
        name = name,
        srcs = srcs,
        data = data + warm_data,
        env = _WARM_ENV | warm_env | env,
        deps = deps + [_PLUGIN_LABEL],
        ignore_extra_deps = ignore_extra_deps + [_PLUGIN_LABEL],
        args = ["-p", _PLUGIN_MODULE] + args,
        **kwargs
    )
