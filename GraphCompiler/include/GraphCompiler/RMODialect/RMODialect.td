//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This declares the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMODIALECT_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMODIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// RMODialect
//===----------------------------------------------------------------------===//

def RMODialect : Dialect {
  let name = "rmo";

  let summary = "Relaxed Modular Operations dialect.";
  let description = [{
    RMO (Relaxed Modular Operations dialect) is a dialect representing a graph
    of Modular Operations (MO nodes) with relaxed constraints on the argument
    types. This is in contrast to MO dialect, which RMO is eventually lowered
    to, where types of operands around shape are strictly enforced.

    MO (Modular Operations dialect) is designed to be a strongly typed dialect
    where each operation does a single thing. For example, `mo.add` does not
    have implicit broadcasting, which would violate the principle of doing a
    single thing. As a purely elementwise operator, the construction of IR
    requires the user to prove all operands and result types are the same to
    enforce the principle of being strongly typed.

    These choices enable MO to be easily optimized, however they impose annoying
    requirements on users which aim to write programs directly in MO. The
    strongly typed nature of MO means frontend dialects generally need some
    level of shape inference to be able to type-check the strong verifiers. This
    is also not always possible in programs with high amounts of data-dependent
    shape behavior. Furthermore, most dialects will have fairly dynamic
    operations which dispatch to different code paths based on some aspect of
    shape or runtime value. For example, `add` in nearly all ML frameworks will
    broadcast, and `matmul` will correspond to a traditional `matmul` or
    `batch_matmul` depending on the ranks of the input. To lower these
    operations in MO requires stringing together multiple MO operators together,
    each with strict shape-related verifiers. It is altogether not an enjoyable
    experience.

    RMO deals with strict typing of MO, by inserting rebinds and dynamic runtime
    checks. Furthermore, as a non-optimizing layer, RMO can have more dynamic
    semantics. For example, while MO will never have a built-in broadcasting
    add, RMO will be allowed to have one. Furthermore, while a `mo.add` will
    require strong types which match, RMO operators will accept nearly any shape
    the user provides generally.

    A core part of the design is having an RMO operator analogous to every MO
    operator with the same semantics just with a relaxed verifier around IR.
    For example we have `mo.add` and `rmo.mo.add`, both of which don't broadcast
    but `rmo.mo.add` will accept nearly any shape.

    Users can then build operators as compositions of existing RMO operators to
    support whatever dynamic behavior they need, i.e. construct an rmo.bcast_add
    out of the RMO operators analogous to MO.
  }];

  let cppNamespace = "::M::RMO";
  let dependentDialects = [
    "MO::MODialect"
  ];

  let extraClassDeclaration = [{
  private:
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect ops.
    void registerOperations();
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD
