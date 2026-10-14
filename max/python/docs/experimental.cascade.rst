:title: max.experimental.cascade
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.experimental.cascade
========================

.. automodule:: max.experimental.cascade
   :no-members:

.. currentmodule:: max.experimental.cascade

Pipelines
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   CascadePipeline
   GenAIPipeline
   GenAIInterface

Runtimes
--------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   LocalRuntime
   Runtime
   Worker
   ~max.experimental.cascade.core.Proxy
   ~max.experimental.cascade.grpc_runtime.GrpcRuntimeClient
   ~max.experimental.cascade.grpc_runtime.SubprocGrpcRuntimeClient
   ~max.experimental.cascade.http_runtime.HttpRuntimeProxy
   ~max.experimental.cascade.http_runtime.SubprocHttpRuntime

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   pipeline_method
   worker_method

Serving
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ~max.experimental.cascade.serve.CascadeFastAPI

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ~max.experimental.cascade.serve.build_router

Requests
--------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ChatMessage
   GenAIRequest
   Modality

Chunks
------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   GenAIChunk
   GenAIImageChunk
   GenAIReasoningChunk
   GenAITextChunk
   GenAITool
   GenAIToolCall

Generation options
------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ImageGenOptions
   TextGenOptions

Results
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   MaybeAsync
   Result
   ResultIter

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   WorkerType
