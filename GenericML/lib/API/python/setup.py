# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Project configuration in addition to pyproject.toml"""
import os
import sys
from glob import glob

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


# Override bdist_wheel to force creation of a platform-specific wheel.
# The project contains C extensions that are built via CMake and added to the
# wheel after the fact.
# Therefore the wheel is not pure Python, yet setuptools thinks it is.
# Hence the need for this workaround.
class bdist_wheel(_bdist_wheel):
    def run(self):
        # Print the wheel basename for CMake to pick up during configure.
        impl_tag, abi_tag, plat_tag = self.get_tag()
        print(f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}")

        super().run()

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


# Copy shared libraries and executables into the wheel.
PY_MODULE_EXTENSION = ".pyd" if sys.platform == "win32" else ".so"
SHARED_LIBS = [
    os.path.abspath(p)
    for p in (
        glob(os.path.join("modular", "lib", "*"))
        + glob(os.path.join("modular", "engine", f"*{PY_MODULE_EXTENSION}"))
    )
]
EXECUTABLES = [os.path.abspath(os.path.join("modular", "bin", "*"))]
# Copy everything under `.mojo_cache` to the wheel.
MOJO_CACHE = [
    os.path.abspath(os.path.join("modular", ".mojo_cache", "**", "*"))
]
# Copy Modular configuration file to the wheel.
MODULAR_CFG = [os.path.abspath(os.path.join("modular", "modular.cfg"))]

setup(
    cmdclass={"bdist_wheel": bdist_wheel},
    name="modular-engine",
    package_data={
        "modular.engine": EXECUTABLES + SHARED_LIBS + MOJO_CACHE + MODULAR_CFG
    },
    packages=["modular.engine"],
    include_package_data=True,
)
