# Clip-VIT demo

## Instructions to run notebook (tensorflow and pytorch)

- Build modular with setup instructions given in PyTorch/docs/Readme.md

- [Optional] Create virtual environment

- Install modular AI Engine

```shell
pip install .derived/build-release/GenericML/lib/API/python/modular_engine-<version-info>.whl
```

- Install dependencies

```shell
pip install -r requirements.txt
```

- Unset `LD_LIBRARY_PATH`. This is to prevent import errors from tensorflow.

```shell
unset LD_LIBRARY_PATH
```

- Run jupyterlab

```shell
jupyter lab
```

- Browse and run notebooks. First run getting-started.ipynb to download models.

## Instructions to run just tensorflow model

- Install modular engine from prebuilt wheel or build wheel on tree. For
getting URL of pre-built nightly wheel look at the workflow logs of Build AI
Engine wheel workflow

- Install dependencies

```shell
pip install -r requirements.txt
```

- Install jupyter lab

```shell
pip install juyterlab
```

- Run jupyterlab

```shell
jupyter lab
```

- Browse and run notebooks. First run getting-started.ipynb to download model.
