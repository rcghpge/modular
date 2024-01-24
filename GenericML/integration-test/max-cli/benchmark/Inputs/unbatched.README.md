# `unbatched`

Generated with the following code:

```python
import tensorflow as tf

class Reluer(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[50, 50], dtype=tf.float32)])
    def relu(self, x):
        return tf.nn.relu(x)


model = Reluer()
tf.saved_model.save(model, "unbatched")
```
