//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTYPES_TD
#define GENERICML_MGPDIALECT_MGPTYPES_TD

include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPDialect.td"
include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPTypeInterfaces.td"
include "MLRT/GraphRT/MEFDialect/MEFTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "Support/MDialect/MAttrs.td"

// Base class for types in this dialect.
class MGP_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MGPDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ContextType
//===----------------------------------------------------------------------===//

def MGP_ContextType : MGP_Type<"Context", "context"> {
  let summary = "Type of execution contexts holding a number of state slots.";
  let description = [{
    Contexts may be created by the 'init' entry point of a model, then reused
    by each call to the model's 'execute' entry point.

    Contexts may contain any number of 'state slots'. A state slot holds
    arbitrary data needed by operations or entire framework runtimes, such as
    attributes, scratch space, pre-computed data or cached buffers. Each op may
    declare its own state slot structure, and each call to that op in the graph
    (an 'operation') may have its own instance of that structure.

    In general a computational operator, such as 'lib.foo', will be represented
    by two primitives, such as 'libp.foo.setup' and 'libp.foo'. The former can
    be invoked by the model's 'init' body to allocate and initialize a state
    slot. The latter can access that state slot when the operation is executing
    in the model's 'execute' body.

    Contexts are represented by `M::MGP::Context` in the Modular runtime.

    Example:
    ```mlir
    !mgp.context
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

def MGP_DTypeType : MGP_Type<"DType", "dtype"> {
  let summary = "Runtime representation of the element type of a CPU tensor.";
  let description = [{
    This type represents a compact runtime encoding of an element type. It is
    implemented by M::DType in the Modular runtime, and is only intended to
    represent common int and float types common across modern CPUs.

    Examples:
    ```mlir
    !mgp.dtype   // A uint8_t encoding of an element type at runtime.
    ```
  }];

  let extraClassDeclaration = [{
    /// Returns the DType representing the given MLIR IntegerType, FloatType
    /// or NoneType. The NoneType is represented as unknown. Returns an error
    /// if the MLIR type has no representation as a DType.
    static ErrorOr<std::optional<DType>> fromMlirType(Type mlirType);

    /// Returns the MLIR IntegerType, FloatType or NoneType representing the
    /// given optDType. The unknown value is represented as the NoneType MLIR
    /// type. Returns an error if the DType has no representation as an MLIR
    /// type.
    static ErrorOr<Type> toMLIRType(MLIRContext *context, std::optional<DType> optDType);

    /// Returns error if the given optDType cannot be represented as an
    /// MLIR IntegerType, FloatType or NoneType. Otherwise returns success.
    static ErrorOrSuccess hasMLIRType(std::optional<DType> optDType);

    /// Returns the 'raw form' of optDType. This will be 0 if DType is unknown,
    /// otherwise the DType's value.
    static uint8_t asRawDType(std::optional<DType> optDType);

    /// Return the 'optional form' of rawDType. This will be none if the
    /// rawDType is 0, otherwise the DType with rawDType as value.
    static std::optional<DType> fromRawDType(uint8_t rawDType);
  }];
}

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

def MGP_List : MGP_Type<"List", "list",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Type of immutable list container.";
  let description = [{
    This type represents an immutable list of elements.
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

//===----------------------------------------------------------------------===//
// Opaque
//===----------------------------------------------------------------------===//

def MGP_OpaqueType : MGP_Type<"Opaque", "opaque",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Opaque MGP type";
  let description = [{
    This is an opaque type that can represent anything.
    This is backed by an AnyAsyncValue type.

    Example:
    ```mlir
    !mgp.opaque
    !mgp.opaque<{foo = "bar"}>
    ```
  }];

  let parameters = (ins "StringAttr": $symbol,
                        OptionalParameter<"std::optional<DictionaryAttr>">:$metadata);
  let assemblyFormat = "`<` $symbol (`,` $metadata^)? `>`";
}

//===----------------------------------------------------------------------===//
// DeviceContextType
//===----------------------------------------------------------------------===//

def MGP_DeviceContextType : MGP_Type<"DeviceContext", "device_context", []> {
  let summary = "MGP representation of the driver DeviceContext";
  let description = [{
    This is essentially opaque to MGP but it is still useful to represent
    directly as it is simpler to introspect.

    Examples:
    ```mlir
    // A device context for CUDA gpu 3.
    !mgp.device_context<<"gpu", 3>>
    ```
  }];

  let parameters = (ins M_DeviceRefAttr:$device);
  let assemblyFormat = "`<` $device `>`";
}

include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPDeviceTypes.td"

#endif  // GENERICML_MGPDIALECT_MGPTYPES_TD
