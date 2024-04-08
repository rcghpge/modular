# `sqrt1`

Generated with the following code:

```python
import tensorflow as tf


class Sqrt(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.float32)])
    def sqrt(self, x):
        return tf.sqrt(x)


model = Sqrt()
tf.saved_model.save(model, "sqrt1")
```
