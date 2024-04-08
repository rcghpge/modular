# BenchmarkCLI Unit Test Inputs

## `empty.dat`

Empty file.

## `basic.npy`

Generated as follows:

```python
import numpy as np
np.save("basic.npy", np.arange(6).reshape(1, 2, 3))
```

## `zero-rank.npy`

Generated as follows:

```python
import numpy as np
np.save("zero-rank.npy", np.arange(1).reshape(()))
```

## `multable.npy`

Generated as follows:

```python
import numpy as np
np.save("multable.npy", np.arange(3).reshape(1, 3) * np.arange(3).reshape(3, 1))
```
