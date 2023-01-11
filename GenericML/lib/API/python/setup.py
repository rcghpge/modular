# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Project configuration in addition to pyproject.toml"""
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


setup(cmdclass={"bdist_wheel": bdist_wheel})
