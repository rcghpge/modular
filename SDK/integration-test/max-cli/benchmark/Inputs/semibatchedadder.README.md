# `semibatchedadder`

Generated with the following code:

```python
import tensorflow as tf

class Adder(tf.Module):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 5], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 5], dtype=tf.float32),
    ])
    def add(self, x, y):
        return x + y


model = Adder()
tf.saved_model.save(model, "semibatchedadder")
```
