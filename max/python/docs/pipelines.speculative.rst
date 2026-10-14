:title: max.pipelines.speculative
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.speculative
=========================

.. automodule:: max.pipelines.speculative
   :no-members:

.. currentmodule:: max.pipelines.speculative

Submodules
----------

.. autosummary::
   :nosignatures:

   max.pipelines.speculative.adaptive_width
   max.pipelines.speculative.block_driver
   max.pipelines.speculative.depth_schedule
   max.pipelines.speculative.driver
   max.pipelines.speculative.spec_target

.. toctree::
   :maxdepth: 1
   :hidden:

   pipelines.speculative.adaptive_width
   pipelines.speculative.block_driver
   pipelines.speculative.depth_schedule
   pipelines.speculative.driver
   pipelines.speculative.spec_target

Configuration
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   SpeculativeConfig
   SpeculativeMethod
   VerifyWidthRange

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   MAGIC_DRAFT_TOKEN_ID

Draft weights
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DraftAliases

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   NO_ALIASES

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   validate_draft_state_dict

Spec-decode inputs
------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   SpecDecodeGraphInputs
   SpecDecodeInputTypeSpec
   SpecDecodeTailValues
   ~max.pipelines.speculative.spec_input_types.SpecDecodeGraphSignature

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   build_spec_decode_input_types
   decode_spec_decode_input_values
   decode_spec_decode_tail
   ~max.pipelines.speculative.unified_graph_ops.accept_and_pick_next_tokens
   ~max.pipelines.speculative.unified_graph_ops.apply_overlap_bitmask
   ~max.pipelines.speculative.unified_graph_ops.broadcast_per_device
   ~max.pipelines.speculative.unified_graph_ops.gather_accepted_hidden_states
   ~max.pipelines.speculative.unified_graph_ops.gather_rows_per_device
   ~max.pipelines.speculative.unified_graph_ops.merge_tokens_and_host_offsets
   ~max.pipelines.speculative.unified_graph_ops.shift_corrected_tokens

Width policies
--------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ~max.pipelines.speculative.spec_width_policy.SpecWidthPolicy

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ~max.pipelines.speculative.spec_width_policy.declares_skippable_draft

Token merging
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   RaggedTokenMerger

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   ragged_token_merger
   ~max.pipelines.speculative.ragged_token_merger.compute_host_merged_offsets
