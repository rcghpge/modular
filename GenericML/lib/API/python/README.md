# Python Packaging

## Build and install

Here's how to build and install the Python package:

1. First be sure you have completed the `modular` repo setup from the
   [README](https://github.com/modularml/modular#getting-set-up), up until (and
   including) the `build-llvm` command.

1. Build TensorFlow (from the `modular` root):

   ```shell
   build-external-tensorflow
   ```

1. Configure the build:

   ```shell
   # Specify your venv root (if using), as this is something you need
   # for the Python API doc build (assuming you'll also want that):
   cmake-modular-release \
     -DMODULAR_PYTHON_API_BUILD_BINDINGS=ON
   ```

1. Build the Python wheel:

   ```shell
   build modular-pybind-ext
   ```

1. Install the Python wheel:

   ```shell
   # Finish this filename as appropriate:
   python3 -m pip install .derived/build/GenericML/lib/API/python/modular_engine-...whl
   ```

Now you should be all set.

BTW, you're also all set to build the Python API docs with this command:

```shell
build inference_engine_docs_python
```

Also see the [Python packaging overview on Notion](https://www.notion.so/modularai/Python-Packaging-99d63148865b435993b9551c1b6f6ff4#0f96f88b6a7844749f56f4ad7f928aea).
