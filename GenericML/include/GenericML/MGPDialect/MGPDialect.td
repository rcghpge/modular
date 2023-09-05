//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPDIALECT_TD
#define GENERICML_MGPDIALECT_MGPDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MGPDialect
//===----------------------------------------------------------------------===//

def MGPDialect : Dialect {
  let name = "mgp";

  let summary = "Buffers, contexts and other framework/kernel independent "
                "glue.";
  let description = [{
    The MGP (pronounced "em-goop") dialect is intended to provide the 'glue'
    for binding framework-specific (eg `mtfl.add`), third-party
    (eg `mlasp.conv2d`) and KGENned kernels/ops together into a single
    executable model.

    MGP itself does not provide any op implementations. MGP is directly
    translatable to MEF (see MLIRToMEF.cpp) without any additional passes,
    however MGP supports source-to-source optimizations to hoist allocations,
    avoid unnecessary conversions, avoid unnecessary shape calculations,
    and avoid unnecessary use of tensor wrappers. Framework-specific dialects
    may depend on MGP, but not vice versa. MGP supports ops both in the
    'value semantic' style (take and return values) and the 'buffer semantic'
    style (take buffers, some of which are pre-allocated to hold outputs).

    Currently MGP is focused exclusively on CPU with a view to allowing both
    tfl and tf models to mix their native ops with a small number of MLAS and
    KGEN ops.

    MGP has five aspects:
    1. Provides a generic `mgp.model` op holding the compiled model’s ‘init’ and
       ‘execute’ invokables. Models supply a framework name attribute to
       indicate to the Modular runtime which framework-specific API to
       instantiate around those two entry points. Generally this includes a
       framework-specific subclass of `M::Model`.
    2. Provides a generic `!mgp.context type`, backed by `M::MGP::Context`,
       which provides an arbitrary number of `M::MGP::StateSlot` 'state slots'.
       Contexts can be returned by model 'init' blocks and then be reused by
       each call to a model 'execute' block. The compiler can use those slots
       to give operations a place to construct and retrieve operation-specific
       cached values. (This mimics the tfl 'Node' struct and the tf OpKernel
       base class). A well-known slot can also be used by the compiler to hold
       a framework-specific context object, thus giving all operations access
       to all global context state.
    3. Provides a generic `!mgp.cpu.buffer type`, backed by
       `M::LLCL::BufferRef`, which represents a (possibly owned) array of
       bytes. The bytes are 'uninterpreted', ie we do not attempt to capture
       rank, dimensions, element type or endianness. In this way we support
       future third-party types beyond tensors, support library-specific
       'packed' representations, and act as a least-common denominator for
       conveying tensor values between ops which may use different tensor
       wrapper types. Buffers may also be cached in ‘buffer slots’ within
       state slots, thus giving every operation the ability to re-use already
       allocated buffers for temporary and output results if desired.
    4. Provides a generic `!mgp.cpu.tensor` type, backed by `M::Tensor`,
       which provides an uncontroversial CPU-only runtime representation for
       a statically-ranked tensor with a shape, dtype and underlying buffer.
       Ops (including the implementations of MO ops) and conversions (see
       below) are free to use this type when convenient so as to be framework
       independent. However it is not a one-size-fits-all universal tensor
       type (and it is very unlikely if it should be generalized to support
       non-CPU memory and layouts). It is possible to create `!mgp.cpu.tensors`
       from a pair of `!mgp.cpu.tensor_spec` (backed by `M::TensorSpec`) and
       `!mgp.cpu.buffer`. It is also possible to create a `!mgp.cpu.tensor_spec`
       from its dynamic dimensions (as the MLIR builtin index type, backed
       by a size_t). In this way MGP provides baseline support for dynamic
       shapes.
    5. [Pending for D] Provides a `mgp.kgen.invoke` primitive for invoking
       KGENed ops whose signatures, due to fusion, may not be known until model
       compile time. The implementations are drawn from the mojo_cache and
       stored in object form in the MEF file. Shim code is also generated at
       model compile time to mediate between the MEF Frame and the ops native
       calling convention (as per an ABI), and similarly stored in object form
       in the MEF file. These shims and ops are linked and registered at model
       setup time and can then be invoked as for any MEF primitive.

    Though not in MGP directly, MGP requires co-operation from the host
    framework in the form of:
    - Conversion primitives between `!mgp.tensor` and any framework native
      types (eg `TfLiteTensor`). For example `tflp.to_mtensor` and
      `tflp.from_mtensor` (see `Conversions.cpp`). These conversions alias the
      underlying buffers when memory safety and memory layout permit. However
      in general these conversions may need to copy or transform buffers.
    - A subclass of `M::Model`, such as `M::TFLModel`, to ensure model
      arguments and results match the native framework representation.

    And, obviously, MGP needs actual op implementations to glue together, such
    as provided by the MLASP dialect and `MLASPPrimitives.cpp`.

    An example of mixing this all together for TFLite can be seen in the
    `MTFLToPrimitives.cpp` pass. (This pass is currently written in 'big step'
    form, and will be refactored as MO and more graph compiler passes come
    online.)

    MGP does not include:
    - Primitives/kernals, such as `MGP.add`. These must be supplied by a
      supporting dialect and runtime implementation.
    - GPU or other non-CPU buffer or tensor representations. These will be
      added by supporting dialects.
    - Transformations into and out of the `!mgp.cpu.tensor` representation.
      These must be supplied by the host framework primitive dialect.

    Commented examples of MGP reside in `TFLite/test/tfl/mgp-examples.mlir`.
  }];

  let cppNamespace = "::M::MGP";
  let dependentDialects = [
    "::M::MEF::MEFDialect",
    "::M::MIP::MIPDialect",
    "::M::MDialect"
  ];

  let extraClassDeclaration = [{
   private:
    /// Register all dialect types.
    void registerTypes();
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect ops.
    void registerOperations();
  }];

  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
  // TODO: Remove this once dialects have been updated to use properties.
  let usePropertiesForAttributes = 0;
}

#endif // GENERICML_MGPDIALECT_MGPDIALECT_TD
