# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Interfaces and result/status classes for pipeline operations in the MAX API.

This module defines the core status and result structures used by pipeline components
to communicate operation outcomes, including success and cancellation states.
"""

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

from .request import RequestID


class PipelineInputs:
    """
    Base class representing inputs to a pipeline operation.

    This class serves as a marker interface for all pipeline input types.
    Concrete implementations should inherit from this class and define
    the specific input data structures required for their pipeline operations.

    .. code-block:: python

        class MyPipelineInputs(PipelineInputs):
            def __init__(self, data: str, config: dict):
                self.data = data
                self.config = config
    """

    ...


# TODO: We should change this to a abstract base class, but this blocks
# msgspec.Struct implementations of this class.
@runtime_checkable
class PipelineOutput(Protocol):
    """
    Protocol representing the output of a pipeline operation.

    Subclasses must implement the `is_done` property to indicate whether
    the pipeline operation has completed.
    """

    @property
    def is_done(self) -> bool:
        """
        Indicates whether the pipeline operation has completed.

        Returns:
            bool: True if the operation is done, False otherwise.
        """
        ...


PipelineOutputType = TypeVar("PipelineOutputType", bound=PipelineOutput)
"""
Type variable for pipeline output types.

This TypeVar is bound to PipelineOutput, ensuring that any type used with this
variable must implement the PipelineOutput protocol. Used for generic typing
in pipeline operations to maintain type safety while allowing flexibility
in output types.

Bounds:
    PipelineOutput: All types must implement the PipelineOutput protocol
"""

PipelineOutputsDict: TypeAlias = dict[RequestID, PipelineOutputType]
"""
Type alias for a dictionary mapping string keys to PipelineOutput instances.

This is used to represent a collection of pipeline outputs, where each key
identifies a specific output.
"""


PipelineInputsType = TypeVar("PipelineInputsType", bound=PipelineInputs)
"""
Type variable for pipeline input types.

This TypeVar is bound to PipelineInputs, ensuring that any type used with this
variable must inherit from the PipelineInputs base class. Used for generic typing
in pipeline operations to maintain type safety while allowing flexibility
in input types.

Bounds:
    PipelineInputs: All types must inherit from PipelineInputs base class
"""


class Pipeline(Generic[PipelineInputsType, PipelineOutputType], ABC):
    """
    Abstract base class for pipeline operations.

    This generic abstract class defines the interface for pipeline operations that
    transform inputs of type PipelineInputsType into outputs of type PipelineOutputsDict[PipelineOutputType].
    All concrete pipeline implementations must inherit from this class and implement
    the execute method.

    Type Parameters:
        PipelineInputsType: The type of inputs this pipeline accepts, must inherit from PipelineInputs
        PipelineOutputType: The type of outputs this pipeline produces, must be a subclass of PipelineOutput

    .. code-block:: python

        class MyPipeline(Pipeline[MyInputs, MyOutput]):
            def execute(self, inputs: MyInputs) -> dict[RequestID, MyOutput]:
                # Implementation here
                pass
    """

    @abstractmethod
    def execute(
        self, inputs: PipelineInputsType
    ) -> PipelineOutputsDict[PipelineOutputType]:
        """
        Execute the pipeline operation with the given inputs.

        This method must be implemented by all concrete pipeline classes.
        It takes inputs of the specified type and returns outputs according
        to the pipeline's processing logic.

        Args:
            inputs: The input data for the pipeline operation, must be of type PipelineInputsType

        Returns:
            The results of the pipeline operation, as a dictionary mapping RequestID to PipelineOutputType

        Raises:
            NotImplementedError: If not implemented by a concrete subclass
        """
        ...

    @abstractmethod
    def release(self, request_id: RequestID) -> None:
        """
        Release any resources or state associated with a specific request.

        This method should be implemented by concrete pipeline classes to perform
        cleanup or resource deallocation for the given request ID. It is typically
        called when a request has completed processing and its associated resources
        (such as memory, cache, or temporary files) are no longer needed.

        Args:
            request_id (RequestID): The unique identifier of the request to release resources for.

        Returns:
            None

        Raises:
            NotImplementedError: If not implemented by a concrete subclass.
        """
        ...
