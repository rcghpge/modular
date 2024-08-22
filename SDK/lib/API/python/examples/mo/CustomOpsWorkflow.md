# Custom Ops workflow

Consider the `mo` representation of a model given below.

```mlir
module {
  mo.graph @serving_default(%arg0: !mo.tensor<[8], f32>) -> !mo.tensor<[8], f32> attributes {argument_names = ["args_0"], result_names = ["output_0"]} {
    %0 = mo.sqrt(%arg0) : !mo.tensor<[8], f32>
    mo.output %0 : !mo.tensor<[8], f32>
  }
}
```

- This is created from the following `tensorflow` model

    ```python
    import tensorflow as tf

    DEFAULT_BATCH_SIZE = 8

    class TestModel(tf.Module, ABC):
        """
        The main API the SavedModel generation relies on.
        """

        _input_shape: Sequence[int] = []

        @abstractmethod
        def __call__(self, *args: Any) -> Any:
            """
            Run the model on the given inputs. To be implemented by subclasses.
            """
            raise NotImplementedError()

        @property
        def input_shape(self) -> Tuple[int]:
            """
            Returns the input shape. Subclasses should prefer setting the
            `_input_shape` field, but can also override this.
            """
            return (DEFAULT_BATCH_SIZE,) + tuple(self._input_shape)

        @tf.function
        def serving_default(self, *args: Any) -> Any:
            """
            The function used to generate the exported signature in the SavedModel.
            The name of this function (i.e. `serving_default`) is used when invoking
            the conversion tool (e.g. `tf-translate`), so DO NOT change this name.
            """
            return self(*args)

    class SqrtOp(TestModel):
        @tf.function
        def __call__(self, x: tf.Tensor) -> tf.Tensor:
            return tf.math.sqrt(x)
    ```

This model takes a tensor, calculates its square root and returns the
result. The sqrt kernel has an implementation in mojo and is shipped with
AI engine. Now lets say user has their own version of sqrt.

```mojo
@export
fn my_sqrt[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return 42
```

For using this kernel in the model first step is to register the kernel as
implementation of a op, In this case `mo.sqrt`. For this user will import
the relevant decorators from `register` (This will be a public facing package)

```mojo
from register import *
```

Next step is mark the existing mojo function. That is given as follows

```mojo
@mogg_register_override("mo.sqrt", 1)
@mogg_elementwise
@export
fn my_sqrt[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    ...
```

`@mogg_register_override` decorator tells which `mo` operation to override
and what’s the priority of this implementation in the event there are
multiple. `@mogg_elementwise` marks the kernel as an elementwise kernel, i.e
a kernel which is implemented in terms of simd and is passed as a lambda to
the elementwise generator.

Last step is to create a mojo package with the given kernel. A package can
have multiple custom ops.

```bash
user_op
├── __init__.mojo
├── other_user_ops.mojo
└── user_sqrt.mojo

$ mojo package user_op -o user_ops.mojopkg
```

Now the mojo ops are ready to be used with the inference engine. User can
register this op at model compilation time with the API and proceed as normal.

```c
// Setup
M_useKernelsFrom(compileConfig, "user_ops.mojopkg", status);
// Compile, and execute model
```

```python
model = session.load(model_path, custom_ops_path=custom_ops_path)
```

## Usage with `mt`

```bash
mt --execute mo-model.mlir --kernel-library-paths=user_ops.mojopkg

``
