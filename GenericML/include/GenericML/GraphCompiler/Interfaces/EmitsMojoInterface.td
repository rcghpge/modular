//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_EMITSMOJOINTERFACE_TD
#define GENERICML_GRAPHCOMPILER_INTERFACES_EMITSMOJOINTERFACE_TD

include "mlir/IR/OpBase.td"

def EmitsMojoInterface : OpInterface<"EmitsMojoInterface"> {
  let description = [{
    An op that implements this interface will ultimately lower to Mojo text.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Return the body of the kernel, which should be the code we will compile",
      "mlir::LogicalResult", "emitMojo", (ins "M::MOGG::EmitMojoState&":$state)
    >,
    InterfaceMethod<
      "Return a set of operand indices that should be unpacked when lowering to a generic compiled region.",
      "mlir::BitVector", "operandsToUnpack", (ins), [{}], [{ return mlir::BitVector($_op->getNumOperands(), true); }]
    >,
    InterfaceMethod<
      "Return a set of result indices that should be packed when lowering to a generic compiled region.",
      "mlir::BitVector", "resultsToPack", (ins), [{}], [{ return mlir::BitVector($_op->getNumResults(), true); }]
    >,
    InterfaceMethod<
      "Return a boolean to indicate whether the emitted code raises exceptions.",
      "bool", "isRaising", (ins), [{}], [{ return false; }]
    >,
  ];
}

def KernelTerminatorProvider : OpInterface<"KernelTerminatorProvider"> {
  let description = [{
    An op that inherits from this interface must have at least one output op as
    user. This is to enforce the fact that outputs actually don't exist and
    instead live in an implicit void** array in kernels.

    This is a temporary interface that will go away when the type system
    properly models what happens in kernels. For the time being, all ops that
    feed into an output op will implement this interface and use the following
    function for both validating that they feed an output op but also to get the
    output index that they should be using when emitting Mojo.
  }];

  let methods = [
    InterfaceMethod<
      [{
        As we emit Mojo, this functions checks that the op feeds into an output
        op and returns the operand index.
      }],
      "std::optional<uint64_t>", "getOutputIndex",
      (ins),
      [{}],
      [{
        if (llvm::none_of($_op->getResult(0).getUsers(),
                   [](auto user) { return user->template hasTrait<mlir::OpTrait::IsTerminator>(); })) {
          $_op->emitError() <<
            "Expected at least one user to be an output op. Canonicalizer pass"
            "could be missing before Emit Mojo";
          return std::nullopt;
        }

        uint64_t resultIndex;
        for (auto &&use : $_op->getResult(0).getUses()) {
          // Can't use MOGG::OutputOp because this interface can't depend on the
          // MOGG dialect.
          if (use.getOwner()->template hasTrait<mlir::OpTrait::IsTerminator>()) {
            resultIndex = use.getOperandNumber();
            break;
          }
        }

        return resultIndex;
      }]
    >,
  ];

  let cppNamespace = "::M::MOGG";
}

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_EMITSMOJOINTERFACE_TD
