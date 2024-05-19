# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from pathlib import Path

from max.tensor import Tensor


trait Checkpoint(Movable):
    def __init__(inout self, path: Path):
        """Initializes the weights file from a path.

        Args:
            path: Filepath to the model's weights file.
        """
        ...

    def get[type: DType](self, key: String) -> Tensor[type]:
        """Returns a tensor for `key` at layer `layer_idx`, possibly seeking the file.

        `self` is `inout` here due to implementations that seek a file pointer.

        Args:
            key: Used to look up the tensor in the weights file.

        Returns:
            A tensor corresponding to `key` and `layer_idx` and containing a
            copy of data from the weights file.

        Raises:
            An error for invalid key arguments.
        """
        ...


struct ReplitCheckpoint(Checkpoint):
    """Reads from a directory containing serialized Mojo Tensors."""

    var model_path: Path

    def __init__(inout self, path: Path):
        """Initializes the weights from a path.

        Args:
            path: Path to model weights directory.
        """
        self.model_path = path

    fn __moveinit__(inout self, owned existing: Self):
        self.model_path = existing.model_path

    def get[type: DType](self, key: String) -> Tensor[type]:
        """Returns a tensor for `key` at layer `layer_idx`, possibly seeking the file.

        `self` is `inout` here due to implementations that seek a file pointer.

        Args:
            key: Used to look up the tensor in the weights file.

        Returns:
            A tensor corresponding to `key` and `layer_idx` and containing a
            copy of data from the weights file.

        Raises:
            An error for invalid key arguments.
        """
        path = self.model_path / key
        if not path.exists():
            raise "Could not load checkpoint tensor value. " + str(
                path
            ) + " does not exist."
        return Tensor[type].load(path)
