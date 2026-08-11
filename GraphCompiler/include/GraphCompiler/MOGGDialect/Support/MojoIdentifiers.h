//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Central catalog of every Mojo-side identifier the GraphCompiler depends on
// and the matching C++ wrappers for high-traffic Mojo types.
//
// Anything the GC encodes about a Mojo struct, free function, method, field,
// parameter name, decorator, package, or MLIR attribute key lives here. The
// goal is that a rename or file-move on the Mojo side becomes a one-line edit
// here, with the rest of the GC unchanged.
//
// Conventions:
//   - All constants use `kCamelCase`.
//   - Prefixes communicate role:
//       kStruct*    Mojo struct names emitted into source.
//       kFn*        Mojo free-function names emitted into source.
//       kField*     Mojo field/method names accessed on values (`.foo`).
//       kEnum*      Mojo enum values (qualified or unqualified).
//       kTrait*     Mojo trait names used in conformance declarations.
//       kComptime*  Pre-rendered comptime type expressions.
//       kAlias*     Mojo alias names.
//       kDecorator* Mojo decorator strings (`@always_inline`, ...).
//       kParameter* Mojo parameter names (kwargs in emitted calls, generic
//                   parameter names, kernel parameter inference keys).
//       kPackage*   Mojo package names.
//       kAttr*      MLIR attribute keys carried on KGEN/MOGG ops.
//       kMogg*      MOGG-internal label strings (also MLIR attribute keys,
//                   but with the `mogg.` namespace prefix baked in).
//
// All constants are at namespace `M::MOGG`; code that already lives in that
// namespace can reference them unqualified.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOIDENTIFIERS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOIDENTIFIERS_H

#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <utility>

namespace M::MOGG {

//===----------------------------------------------------------------------===//
// MLIR attribute keys (carried on MOGG ops)
//===----------------------------------------------------------------------===//

/// Module-level `DictionaryAttr` mapping simple struct names (e.g.
/// "MyStruct") to their fully-qualified Mojo names (e.g.
/// "foo.bar.MyStruct"). Populated by `mogg-import-packages` while walking
/// the loaded packages' `StructDeclOp` trees, and consumed by
/// `mogg-rewrite-opaque-types` to rewrite `!mo.opaque<"...">` references
/// in the module. The mapping is the canonical persistence form: it
/// travels with the module (e.g. when constant folding copies decls into
/// a sub-module) so downstream callers don't need to re-walk packages.
constexpr llvm::StringLiteral kOpaqueTypeMappingAttrName =
    "mogg.opaque_type_mapping";

//===----------------------------------------------------------------------===//
// MOGG kernel function labels
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kMoggExecuteFunctionLabel = "mogg.execute";
constexpr llvm::StringLiteral kMoggShapeFunctionLabel = "mogg.shape";
constexpr llvm::StringLiteral kMoggUpdateViewFunctionLabel = "mogg.update_view";
constexpr llvm::StringLiteral kMoggElementwiseFunctionLabel =
    "mogg.elementwise";
constexpr llvm::StringLiteral kMoggTileElementwiseFunctionLabel =
    "tile_elementwise";
constexpr llvm::StringLiteral kMoggBuildGraphFunctionLabel = "mogg.build_graph";

/// Mojo source method names of extensibility kernels (`@register`-decorated
/// structs) and the decorator-spelling constants. `kFnRegister`,
/// `kFnRegisterInternal`, and `kMoggExecuteFuncName` are owned by KGEN's
/// `LITUtils.h` (the Mojo parser uses them to drive value-witness generation
/// and decorator recognition) and re-exported here.
using KGEN::kFnRegister;
using KGEN::kFnRegisterInternal;
using KGEN::kMoggExecuteFuncName;
constexpr llvm::StringLiteral kMoggShapeFuncName = "shape";
constexpr llvm::StringLiteral kMoggElementwiseFuncName = "elementwise";
constexpr llvm::StringLiteral kMoggUpdateViewFuncName = "update_input_view";
constexpr llvm::StringLiteral kMoggBuildGraphFuncName = "build_graph";

/// Argument indices into the vector returned by `getDecoratorLambdaArgument`
/// for the `@extensibility.register` decorator (name, type, api, arch, model).
constexpr size_t kRegisterDecoratorArgName = 0;
constexpr size_t kRegisterDecoratorArgType = 1;
constexpr size_t kRegisterDecoratorArgApi = 2;
constexpr size_t kRegisterDecoratorArgArch = 3;
constexpr size_t kRegisterDecoratorArgModel = 4;

constexpr llvm::StringLiteral kMoggTargetParameterName = "target";

/// The name of the output_rank parameter which we support lowering specially.
constexpr llvm::StringLiteral kMoggOutputRankParameterName = "output_rank";

/// The number of DPS (Destination Passing Style) output operands there are.
constexpr llvm::StringLiteral kMoggNumDpsOutputs = "mogg.num_dps_outputs";

/// Discardable attributes carrying the trait-lane per-argument tensor-trait
/// contracts (`#mogg.tensor_arg<...>` or `unit`) on a `mogg.kernel` op. The
/// input array is indexed in exec-body-argument (operand) space, the output
/// array in result space. Set only when at least one argument is a trait arg;
/// absent for pure legacy `ManagedTensorSlice` kernels (so their lowering and
/// printing are unchanged). Read in `buildMojoIOTensorObject` to route trait
/// args to `mogg.tensor_arg.__init__`.
constexpr llvm::StringLiteral kInputTensorArgContractsAttrName =
    "mogg.input_tensor_arg_contracts";
constexpr llvm::StringLiteral kOutputTensorArgContractsAttrName =
    "mogg.output_tensor_arg_contracts";

/// Per-stub-argument buffer-capture comptime metadata stamped on the MOGG-ABI
/// `mogg.kernel.stub` for trait tensor operands. Entry `i` is a
/// `DictionaryAttr` `{dtype, layout_type}` (Mojo type expressions as strings)
/// when stub argument `i` is a `_MoggTensorBufferCapture`, or an empty string
/// otherwise. Emit types the stub parameter as
/// `_MoggTensorBufferCapture[dtype, layout_type]`; execute-facing trait args
/// are assembled inside the stub body from the capture + contract.
constexpr llvm::StringLiteral kTraitStubBufferCaptureAttrName =
    "mogg.trait_stub_buffer_capture";

/// Legacy D9 generic trait bounds (superseded by buffer capture for trait
/// tensor operands). Retained for shape-function stubs that still use generic
/// `DenseTensor` parameters.
constexpr llvm::StringLiteral kTraitStubArgBoundsAttrName =
    "mogg.trait_stub_arg_bounds";

/// `has_epilogue_fusion` parameter on accumulation-style fused kernels.
constexpr llvm::StringLiteral kMoggHasEpilogueFusionParameterName =
    "has_epilogue_fusion";

/// `has_prologue_fusion` parameter: true iff a producer is fused into the
/// kernel's input
constexpr llvm::StringLiteral kMoggHasPrologueFusionParameterName =
    "has_prologue_fusion";

constexpr llvm::StringLiteral kMoggTraceNameParameterName = "_trace_name";

/// Roots of Mojo packages that hold the MAX compiler's internal-only
/// decorators (used to recognise `@register_internal` and friends without a
/// symbol-table lookup into dependency packages). `kPackageStd` and
/// `kPackageExtensibility` are owned by KGEN's `LITUtils.h` (the Mojo parser
/// tests against them too) and re-exported here.
constexpr llvm::StringLiteral kPackageMax = "max";
using KGEN::kPackageExtensibility;
using KGEN::kPackageStd;
constexpr llvm::StringLiteral kPackageManagedTensorSlice =
    "managed_tensor_slice";

/// Symbol-path leaf names used in MLIR symbol-tree matching (these are the
/// last segments of a `SymbolRefAttr` path; they don't appear dotted in
/// emitted source the way the `kStruct*` constants do).
/// `kLeafManagedTensorSlice` and `kLeafDeviceContext` are owned by KGEN's
/// `LITUtils.h` (the Mojo parser tests against them too) and re-exported
/// here.
using KGEN::kLeafDeviceContext;
using KGEN::kLeafManagedTensorSlice;
constexpr llvm::StringLiteral kLeafVariadicTensors = "VariadicTensors";
constexpr llvm::StringLiteral kLeafFusedInputVariadicTensors =
    "_FusedInputVariadicTensors";
constexpr llvm::StringLiteral kLeafFusedOutputVariadicTensors =
    "_FusedOutputVariadicTensors";
constexpr llvm::StringLiteral kLeafSIMD = "SIMD";
constexpr llvm::StringLiteral kLeafIndexList = "IndexList";
constexpr llvm::StringLiteral kLeafTileTensor = "TileTensor";
constexpr llvm::StringLiteral kLeafDeviceContextArray = "DeviceContextArray";
constexpr llvm::StringLiteral kLeafDeviceGraphBuilder = "DeviceGraphBuilder";

/// Mojo module names that the MojoRegistry imports unconditionally.
constexpr llvm::StringLiteral kMoggKernelsModuleName = "builtin_kernels";
constexpr llvm::StringLiteral kMoggPrimitivesModuleName = "builtin_primitives";

/// `@register_internal` name of the helper that prefixes an async error with
/// its region name. Called from `AsyncErrorOp::emitMojo` alongside the op's
/// own primitive (`mogg.async.error`), so this one can't be derived from
/// `getOperationName()`.
constexpr llvm::StringLiteral kMoggPrimitiveFormatRegionError =
    "mogg.format_region_error";

//===----------------------------------------------------------------------===//
// Mojo type symbol names (as stored in MLIR by the Mojo compiler)
//===----------------------------------------------------------------------===//

/// The stored Mojo type symbol name of Tensor type in extensibility kernels.
constexpr llvm::StringLiteral kMojoDpsTensorTypeName =
    "extensibility::managed_tensor_slice::ManagedTensorSlice";

constexpr llvm::StringLiteral kMojoInternalDpsSimdTypeName = "std::SIMD";
constexpr llvm::StringLiteral kMojoInternalDpsIntTypeName = "std::Int";
constexpr llvm::StringLiteral kMojoInternalDpsUintTypeName = "std::UInt";
constexpr llvm::StringLiteral kMojoInternalDpsBoolTypeName = "std::Bool";
constexpr llvm::StringLiteral kMojoInternalDpsIndexListTypeName =
    "std::IndexList";

/// Fully-qualified form of IndexList's type name, matching the dotted format
/// produced by KernelDeclAdaptor's `buildOpaqueSymbol` (every segment joined
/// by '.', usable directly as a Mojo qualified name). The KernelDeclAdaptor-
/// based code path stores opaque type names in this form, so
/// `isOpaqueWithTypeName` checks against it directly.
constexpr llvm::StringLiteral kMojoInternalDpsIndexListTypeQualifiedName =
    "std.utils.index.IndexList";

/// VariadicTensors is the unfused variant; the fused variants carry per-
/// element fusion structs via a heterogeneous Tuple.
constexpr llvm::StringLiteral kMojoVariadicTensorsName =
    "extensibility::managed_tensor_slice::VariadicTensors";
constexpr llvm::StringLiteral kMojoFusedInputVariadicTensorsName =
    "extensibility::managed_tensor_slice::_FusedInputVariadicTensors";
constexpr llvm::StringLiteral kMojoFusedOutputVariadicTensorsName =
    "extensibility::managed_tensor_slice::_FusedOutputVariadicTensors";

/// The stored Mojo type symbol name of device contexts in extensibility
/// kernels.
constexpr llvm::StringLiteral kMojoExtensibilityApiDeviceContextTypeName =
    "max::DeviceContext";

/// The stored Mojo type symbol name of device-context arrays in
/// extensibility kernels (used by multi-device custom ops).
constexpr llvm::StringLiteral kMojoExtensibilityApiDeviceContextArrayTypeName =
    "max::DeviceContextArray";

/// These type names can be confusing. Inside the Graph Compiler code, the
/// types show up like this:
///   - StaticString -> std::StringSpan
///   - String -> std::String
///   - StringSpan -> std::Origin
constexpr llvm::StringLiteral kMojoInternalStaticStringTypeName =
    "std::StringSpan";
constexpr llvm::StringLiteral kMojoStaticStringTypeName = "std::StaticString";

/// In the emit-mojo path, DType parameters appear as LIT struct types with
/// type name "std::DType" rather than KGEN::DTypeType.
constexpr llvm::StringLiteral kMojoDtypeTypeName = "std::DType";

constexpr llvm::StringLiteral kMojoStringTypeName = "std::String";
constexpr llvm::StringLiteral kMojoInternalStringSliceTypeName = "std::Origin";

//===----------------------------------------------------------------------===//
// Mojo struct names (dotted, as emitted into Mojo source)
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kStructIO = "extensibility.IO";
constexpr llvm::StringLiteral kStructIndexList = "std.utils.IndexList";
constexpr llvm::StringLiteral kStructPointer = "Pointer";
/// Generic (parametrizable) origin struct, e.g. `UntrackedOrigin[mut=...]`.
/// Distinct from the FIXED `kComptimeMutUntrackedOrigin`/
/// `kComptimeMutAnyOrigin` aliases below, which bake in a specific
/// mutability and so can't be used where `mut` must stay a generic
/// parameter.
constexpr llvm::StringLiteral kStructUntrackedOrigin = "UntrackedOrigin";
constexpr llvm::StringLiteral kStructOptional = "Optional";
constexpr llvm::StringLiteral kStructString = "String";
constexpr llvm::StringLiteral kStructDeviceContext =
    "max.gpu.host.device_context.DeviceContext";
constexpr llvm::StringLiteral kStructDeviceContextArray =
    "max.gpu.host.device_context.DeviceContextArray";
constexpr llvm::StringLiteral kStructDeviceGraph =
    "max.gpu.host.device_graph.DeviceGraph";
constexpr llvm::StringLiteral kStructDeviceGraphBuilder =
    "max.gpu.host.device_graph.DeviceGraphBuilder";
constexpr llvm::StringLiteral kStructDeviceGraphNode =
    "max.gpu.host.device_graph.DeviceGraphNode";
constexpr llvm::StringLiteral kStructDeviceGraphCache =
    "max.gpu.host.device_graph.DeviceGraphCache";
// Device-graph method and keyword-argument names live on `DeviceGraphRef` and
// `DeviceGraphBuilderRef` below, beside the accessors that emit them.
constexpr llvm::StringLiteral kStructManagedTensorSlice =
    "extensibility.ManagedTensorSlice";
constexpr llvm::StringLiteral kStructInt = "Int";
constexpr llvm::StringLiteral kStructSIMDLength = "SIMDLength";
constexpr llvm::StringLiteral kStructUInt = "UInt";
constexpr llvm::StringLiteral kStructBool = "Bool";
constexpr llvm::StringLiteral kStructAddressSpace = "std.memory.AddressSpace";
constexpr llvm::StringLiteral kStructDType = "DType";
constexpr llvm::StringLiteral kStructSIMD = "SIMD";
// Qualified so the emitted tile `compute` signature/body is import-resolvable:
// the generated Mojo module imports packages qualified (`import layout`, etc.),
// so bare `TileTensor`/`TensorLayout`/`TileCopier` don't bind when the tile
// path is actually compiled. `TileTensor`/`TensorLayout` are re-exported from
// `layout`; `TileCopier` lives in `layout.tile_io` (imported below).
constexpr llvm::StringLiteral kStructTileTensor = "layout.TileTensor";
constexpr llvm::StringLiteral kStructTensorLayout = "layout.TensorLayout";
constexpr llvm::StringLiteral kStructTileCopier = "layout.tile_io.TileCopier";
constexpr llvm::StringLiteral kStructTileLayout = "layout.tile_layout.Layout";
/// Trait-lane wrapper carrying the graph-visible `StoreAccess` marker over an
/// output-store fusion functor (`OutputStoreFusion` conformance).
constexpr llvm::StringLiteral kStructMoggOutputStoreFusionTensor =
    "_MoggOutputStoreFusionTensor";
constexpr llvm::StringLiteral kStructStateContext =
    "builtin_primitives.primitives.StateContext";
constexpr llvm::StringLiteral kStructStaticTuple = "std.utils.StaticTuple";
constexpr llvm::StringLiteral kStructArray = "std.collections.Array";
constexpr llvm::StringLiteral kStructMoggAsyncPackHelper =
    "builtin_primitives.primitives.MoggAsyncPackHelper";
constexpr llvm::StringLiteral kStructScalar = "Scalar";
constexpr llvm::StringLiteral kStructVariadic = "Variadic";
constexpr llvm::StringLiteral kStructParameterList = "ParameterList";
constexpr llvm::StringLiteral kStructStaticTensorSpec =
    "managed_tensor_slice.StaticTensorSpec";
constexpr llvm::StringLiteral kStructStaticTensorSpecList =
    "extensibility.StaticTensorSpecList";
constexpr llvm::StringLiteral kStructVariadicTensors =
    "extensibility.VariadicTensors";
constexpr llvm::StringLiteral kStructStaticString = "StaticString";
constexpr llvm::StringLiteral kStructMsFusionPack =
    "managed_tensor_slice._FusionPack";
constexpr llvm::StringLiteral kStructMsFusedInputVariadicTensors =
    "managed_tensor_slice._FusedInputVariadicTensors";
constexpr llvm::StringLiteral kStructMsFusedOutputVariadicTensors =
    "managed_tensor_slice._FusedOutputVariadicTensors";
constexpr llvm::StringLiteral kStructTrace = "max.runtime.tracing.Trace";
constexpr llvm::StringLiteral kStructIntTuple = "layout.IntTuple";
constexpr llvm::StringLiteral kStructCoord = "layout.Coord";
constexpr llvm::StringLiteral kAliasIdx = "layout.Idx";
constexpr llvm::StringLiteral kFnRowMajor = "layout.row_major";

constexpr llvm::StringLiteral kAliasIndexListToTileLayout =
    "managed_tensor_slice._IndexListToTileLayout";

//===----------------------------------------------------------------------===//
// Mojo enum values
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kEnumAddressSpaceGeneric = "GENERIC";
constexpr llvm::StringLiteral kEnumIoInput = "extensibility.IO.Input";
constexpr llvm::StringLiteral kEnumTraceLevelOp =
    "max.runtime.tracing.TraceLevel.OP";
/// `DType.int8` member used to spell out a one-byte dtype.
constexpr llvm::StringLiteral kEnumDTypeInt8 = "int8";
/// `DType.int` member: the native index-sized integer dtype used for the
/// dynamic (runtime) elements of layout `Coord`s.
constexpr llvm::StringLiteral kEnumDTypeInt = "int";

//===----------------------------------------------------------------------===//
// Mojo free-function names (emitted into Mojo source)
//===----------------------------------------------------------------------===//

// Qualified (std.memory is imported qualified) so the tile `compute` body's
// staging allocation binds when the tile path is compiled.
constexpr llvm::StringLiteral kFnStackAllocation =
    "std.memory.stack_allocation";
/// `std.utils.numerics.nan[dtype]()` / `inf[dtype]()`: dtype-parameterized
/// constructors for the special float values that have no Mojo literal
/// spelling.
constexpr llvm::StringLiteral kFnNumericsNan = "std.utils.numerics.nan";
constexpr llvm::StringLiteral kFnNumericsInf = "std.utils.numerics.inf";
constexpr llvm::StringLiteral kFnToManagedTensorSlice =
    "builtin_primitives.primitives.to_managed_tensor_slice";
constexpr llvm::StringLiteral kFnSimdLoadFromManagedTensorSlice =
    "managed_tensor_slice.simd_load_from_managed_tensor_slice";
constexpr llvm::StringLiteral kFnSimdLoadFromTensorPointer =
    "managed_tensor_slice.simd_load_from_tensor_pointer";
constexpr llvm::StringLiteral kFnSimdStoreIntoManagedTensorSlice =
    "managed_tensor_slice.simd_store_into_managed_tensor_slice";
constexpr llvm::StringLiteral kFnSimdStoreIntoTensorPointer =
    "managed_tensor_slice.simd_store_into_tensor_pointer";
constexpr llvm::StringLiteral kFnUnpackTensorSpec =
    "builtin_primitives.primitives.unpack_tensor_spec";
constexpr llvm::StringLiteral kFnGetSimdWidthForDtypes =
    "builtin_primitives.primitives.get_simd_width_for_dtypes";
constexpr llvm::StringLiteral kFnUnpackBufferRef =
    "builtin_primitives.primitives.unpack_buffer_ref";
constexpr llvm::StringLiteral kFnBufferConstant =
    "builtin_primitives.primitives.mgp_buffer_constant";
constexpr llvm::StringLiteral kFnUnpackDeviceCtx =
    "builtin_primitives.primitives.unpack_device_ctx";
constexpr llvm::StringLiteral kFnUnpackStateCtx =
    "builtin_primitives.primitives.unpack_state_ctx";
constexpr llvm::StringLiteral kFnUnpackTensor =
    "builtin_primitives.primitives.unpack_tensor";
constexpr llvm::StringLiteral kFnAllZeros =
    "builtin_primitives.primitives.all_zeros";
constexpr llvm::StringLiteral kFnMoggAsyncPackOwnedTensor =
    "builtin_primitives.primitives.mogg_async_pack_owned_tensor";
constexpr llvm::StringLiteral kFnMsSimdLoadFromTensorPointer =
    "managed_tensor_slice.simd_load_from_tensor_pointer";
constexpr llvm::StringLiteral kFnMsSimdStoreIntoTensorPointer =
    "managed_tensor_slice.simd_store_into_tensor_pointer";
/// Comptime hook returning the `(rows, cols)` tile shape a tile-programming
/// kernel operates on, matching the driver's `foreach_fusion_tile` default.
constexpr llvm::StringLiteral kFnGetKernelTileShape =
    "managed_tensor_slice.get_kernel_tile_shape";
constexpr llvm::StringLiteral kFnToIndexList = "layout.to_index_list";
/// Converts a `layout.Coord` into an `IndexList[rank]`; `std.utils.coord` is
/// not re-exported from `std.utils`, so this needs its own qualified name.
constexpr llvm::StringLiteral kFnCoordToIndexList =
    "std.utils.coord.coord_to_index_list";
/// The async task primitives are package-private: `std.runtime` does not
/// re-export them, so these need the qualified `_asyncrt` module name.
constexpr llvm::StringLiteral kFnCreateTask =
    "std.runtime._asyncrt.create_task";

/// Affinity-aware variant: `_create_task(handle, *, desired_worker_id=<id>)`.
/// Only emitted for GPU-device launches; maps to the package-private function
/// added alongside `task_id_for_device` in `max.runtime.asyncrt`.
constexpr llvm::StringLiteral kFnCreateTaskWithAffinity =
    "std.runtime._asyncrt._create_task";
constexpr llvm::StringLiteral kFnTaskIdForDevice =
    "max.runtime.asyncrt.task_id_for_device";
constexpr llvm::StringLiteral kFnGetSafeTaskId =
    "max.runtime.tracing.get_safe_task_id";

/// Mojo functions emitted directly.
constexpr llvm::StringLiteral kFnMax = "std.math.max";
constexpr llvm::StringLiteral kFnPrint = "print";

// NOTE: Mojo field / method names accessed on values via `.foo` are no longer
// free constants. Each one is a `static constexpr llvm::StringLiteral` member
// of the wrapper class for the type that owns it (see "Wrapper classes for
// high-traffic Mojo types" below).

//===----------------------------------------------------------------------===//
// Trait names (used in struct conformance declarations)
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kTraitInputFusion =
    "managed_tensor_slice.InputFusion";
constexpr llvm::StringLiteral kTraitOutputFusion =
    "managed_tensor_slice.OutputFusion";
constexpr llvm::StringLiteral kTraitComputeOutputFusion =
    "managed_tensor_slice.ComputeOutputFusion";
constexpr llvm::StringLiteral kTraitComputeOutputFusionTile =
    "managed_tensor_slice.ComputeOutputFusionTile";
constexpr llvm::StringLiteral kTraitOutputFusionTile =
    "managed_tensor_slice.OutputFusionTile";
constexpr llvm::StringLiteral kTraitElementwiseFusion =
    "managed_tensor_slice.ElementwiseFusion";
constexpr llvm::StringLiteral kTraitElementwiseFusionTile =
    "managed_tensor_slice.ElementwiseFusionTile";

//===----------------------------------------------------------------------===//
// Decorator strings
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kDecoratorAlwaysInline = "@always_inline";

//===----------------------------------------------------------------------===//
// Pre-rendered comptime type expressions
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kComptimeOpaquePointer =
    "OpaquePointer[MutAnyOrigin]";
constexpr llvm::StringLiteral kComptimeInputTensor =
    "extensibility.InputTensor";
constexpr llvm::StringLiteral kComptimeOutputTensor =
    "extensibility.OutputTensor";
constexpr llvm::StringLiteral kComptimeMutableInputTensor =
    "managed_tensor_slice._MutableInputTensor";
constexpr llvm::StringLiteral kComptimeUInt8 = "UInt8";
constexpr llvm::StringLiteral kComptimeMutAnyOrigin = "MutAnyOrigin";
constexpr llvm::StringLiteral kComptimeMutUntrackedOrigin =
    "MutUntrackedOrigin";
constexpr llvm::StringLiteral kComptimeNone = "None";
constexpr llvm::StringLiteral kComptimeOptionalErrorEmpty = "Optional[Error]()";
constexpr llvm::StringLiteral kComptimeMutByteBuffer =
    "builtin_primitives.primitives.MutByteBuffer";
constexpr llvm::StringLiteral kStructOwnedByteBuffer =
    "builtin_primitives.primitives.OwnedByteBuffer";
constexpr llvm::StringLiteral kStructOwnedTensor =
    "builtin_primitives.primitives.OwnedTensor";
/// `async_pack()` method on `OwnedByteBuffer` / `OwnedTensor`: builds a fresh
/// `AsyncValue` over the composite's backing memory, used to register a
/// device-graph output.
constexpr llvm::StringLiteral kFnAsyncPack = "async_pack";
constexpr llvm::StringLiteral kComptimeDTypeNoneType = "NoneType";

//===----------------------------------------------------------------------===//
// Parameter names (kernel param inference keys, kwargs in emitted calls)
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kParameterMut = "mut";
constexpr llvm::StringLiteral kParameterInput = "input";
constexpr llvm::StringLiteral kParameterDType = "dtype";
constexpr llvm::StringLiteral kParameterRank = "rank";
/// SIMD width parameter name in a fusion functor's `__call__`/`load`/`store`
/// method signature (paired with `dtype`/`rank`/`element_alignment`).
constexpr llvm::StringLiteral kParameterWidth = "width";
constexpr llvm::StringLiteral kParameterIOSpec = "io_spec";
constexpr llvm::StringLiteral kParameterStaticSpec = "static_spec";
constexpr llvm::StringLiteral kParameterSize = "size";
/// `DeviceContextArray[length=N]`'s parameter keyword.
constexpr llvm::StringLiteral kParameterLength = "length";
constexpr llvm::StringLiteral kParameterStaticSpecs = "static_specs";
constexpr llvm::StringLiteral kParameterSimdWidth = "simd_width";
constexpr llvm::StringLiteral kParameterElementAlignment = "element_alignment";
constexpr llvm::StringLiteral kParameterElementSize = "element_size";
constexpr llvm::StringLiteral kParameterLayoutType = "LayoutType";
constexpr llvm::StringLiteral kParameterOrigin = "origin";
constexpr llvm::StringLiteral kParameterAddressSpace = "address_space";
constexpr llvm::StringLiteral kParameterLinearIdxType = "linear_idx_type";
constexpr llvm::StringLiteral kParameterStorage = "Storage";
constexpr llvm::StringLiteral kParameterTaskId = "task_id";
constexpr llvm::StringLiteral kParameterOutput = "output";
constexpr llvm::StringLiteral kParameterElementType = "element_type";
/// `Pointer.__init__(to=<value>)` kwarg.
constexpr llvm::StringLiteral kParameterTo = "to";
/// Runtime kwargs of `foreach_fusion` (`tensor`/`elem`/`ctx`). Keep in sync
/// with `max/kernels/src/graph_compiler/builtin_primitives/primitives.mojo`.
constexpr llvm::StringLiteral kParameterForeachTensor = "tensor";
constexpr llvm::StringLiteral kParameterForeachElem = "elem";
constexpr llvm::StringLiteral kParameterForeachCtx = "ctx";
constexpr llvm::StringLiteral kMoggForeachTileRegistration =
    "mogg.call.foreach_tile";
/// Argument names used in trait-conforming fusion struct methods
/// (`load`/`store`/`compute`).
constexpr llvm::StringLiteral kParameterSelf = "self";
constexpr llvm::StringLiteral kParameterIdx = "idx";
constexpr llvm::StringLiteral kParameterVal = "val";
constexpr llvm::StringLiteral kParameterDst = "dst";
constexpr llvm::StringLiteral kParameterTileCoords = "tile_coords";
constexpr llvm::StringLiteral kParameterCopier = "copier";
/// The comptime copier-*type* parameter in the tile `compute` signature. It
/// must differ from the runtime arg name (`kParameterCopier`), or the emitted
/// method redefines `copier`. Mirrors the trait's `Copier`/`copier` split.
constexpr llvm::StringLiteral kParameterCopierType = "Copier";
constexpr llvm::StringLiteral kParameterStoreCopier = "store_copier";
constexpr llvm::StringLiteral kParameterStoreCopierType = "StoreCopier";

//===----------------------------------------------------------------------===//
// Stdlib package names
//===----------------------------------------------------------------------===//

/// Packages that the emit-Mojo path imports unconditionally into every
/// generated Mojo module. TODO(GEX-2488): Do this programmatically by
/// checking the Modular config rather than hardcoding.
constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral>
    kPackagesToImport[] = {
        {"builtin_primitives.primitives", "builtin_primitives"},
        {"builtin_kernels", "builtin_kernels"},
        {"extensibility", ""},
        {"extensibility.managed_tensor_slice", "managed_tensor_slice"},
        {"max.gpu.host.device_context", ""},
        {"max.gpu.host.device_graph", ""},
        {"std.utils", ""},
        {"std.utils.numerics", ""},
        {"std.math", ""},
        {"std.runtime", ""},
        {"std.runtime._asyncrt", ""},
        {"std.memory", ""},
        {"std.collections", ""},
        {"max.runtime.tracing", ""},
        {"layout", ""},
        // Makes `layout.tile_io.TileCopier` resolvable in the tile `compute`
        // signature when the tile path is compiled.
        {"layout.tile_io", ""},
};

/// TODO(GEX-2488): Find another way to detect that a type is an `IOSpec`
/// without relying on package / type name.
inline bool isMojoIOSpecTypeName(llvm::ArrayRef<mlir::FlatSymbolRefAttr> refs) {
  // After the extensibility fuse, `IOSpec` lives in
  // `extensibility/managed_tensor_slice.mojo` and the Mojo compiler emits
  // it under the dotted path `managed_tensor_slice.IOSpec`.
  constexpr llvm::StringLiteral mojoTypeName = "IOSpec";
  if (refs.empty())
    return false;
  return refs.back().getValue() == mojoTypeName &&
         refs.front().getValue() == kPackageExtensibility;
}

//===----------------------------------------------------------------------===//
// IOSpec encoding constants and Mojo enum names
//===----------------------------------------------------------------------===//

constexpr bool kIOSpecImmutable = false;
constexpr bool kIOSpecMutable = true;

constexpr int64_t kIOSpecIOOutput = 0;
constexpr int64_t kIOSpecIOInput = 1;
constexpr int64_t kIOSpecIOFusedInput = 2;
constexpr int64_t kIOSpecIOFusedOutput = 3;
constexpr int64_t kIOSpecIOFusedComputeOutput = 31;
constexpr int64_t kIOSpecIOFusedComputeOutputTile = 32;
constexpr int64_t kIOSpecIOFusedOutputTile = 33;
constexpr int64_t kIOSpecIOUnknown = -1;

constexpr llvm::StringLiteral kInputTensor = "InputTensor";
constexpr llvm::StringLiteral kOutputTensor = "OutputTensor";
constexpr llvm::StringLiteral kMutableInputTensor = "MutableInputTensor";
constexpr llvm::StringLiteral kFusedInputTensor = "FusedInputTensor";
constexpr llvm::StringLiteral kFusedOutputTensor = "FusedOutputTensor";
constexpr llvm::StringLiteral kFusedComputeOutputTensor =
    "FusedComputeOutputTensor";
constexpr llvm::StringLiteral kFusedComputeOutputTileTensor =
    "FusedComputeOutputTileTensor";
constexpr llvm::StringLiteral kFusedOutputTileTensor = "FusedOutputTileTensor";

/// Mojo `IO` enum member-alias short names (the names that appear inside
/// `KGEN::SugarKind::MemberAlias` attributes when an `IOSpec` parameter is
/// stored in elaborated bytecode).
constexpr llvm::StringLiteral kIoInput = "Input";
constexpr llvm::StringLiteral kIoOutput = "Output";
constexpr llvm::StringLiteral kIoFusedInput = "FusedInput";
constexpr llvm::StringLiteral kIoFusedOutput = "FusedOutput";
constexpr llvm::StringLiteral kIoFusedComputeOutput = "FusedComputeOutput";
constexpr llvm::StringLiteral kIoFusedComputeOutputTile =
    "_FusedComputeOutputTile";
constexpr llvm::StringLiteral kIoFusedOutputTile = "_FusedOutputTile";

//===----------------------------------------------------------------------===//
// Wrapper classes for high-traffic Mojo types
//===----------------------------------------------------------------------===//
//
// Each wrapper holds a `MojoValue` and exposes typed accessors that use the
// centralized field/method-name constants above. They make the Mojo ABI the
// GraphCompiler relies on visible at the C++ call site: instead of
// `value.get("unsafe_ptr")()` you write `mts.unsafePtr()`. A rename in Mojo
// becomes a one-line edit in the wrapper rather than a grep across the GC.
//
// The wrappers are intentionally thin (no validation, no caching). They are
// pass-by-value, hold a `MojoValue`, and inline trivially.
//
//===----------------------------------------------------------------------===//

/// Wrapper around a Mojo `ManagedTensorSlice` value. Exposes the fields and
/// fusion-bind methods the GC accesses during Mojo emission.
class ManagedTensorSliceRef {
public:
  explicit ManagedTensorSliceRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo field / method symbol names on `ManagedTensorSlice`.
  static constexpr llvm::StringLiteral kUnsafePtr = "unsafe_ptr";
  static constexpr llvm::StringLiteral kShape = "shape";
  static constexpr llvm::StringLiteral kStrides = "strides";
  static constexpr llvm::StringLiteral kToTileTensor = "to_tile_tensor";
  static constexpr llvm::StringLiteral kStaticSpec = "static_spec";
  static constexpr llvm::StringLiteral kStaticShapeTuple =
      "_static_shape_tuple";
  static constexpr llvm::StringLiteral kStaticStridesTuple =
      "_static_strides_tuple";
  static constexpr llvm::StringLiteral kAlignment = "alignment";
  /// `_ptr` is the underlying-storage pointer field used by VariadicTensors
  /// elements.
  static constexpr llvm::StringLiteral kPtr = "_ptr";
  static constexpr llvm::StringLiteral kIoSpec = "io_spec";
  /// Bind methods for fusion lambdas.
  static constexpr llvm::StringLiteral kBindToFusedInput =
      "_bind_to_fused_input";
  static constexpr llvm::StringLiteral kBindToFusedOutput =
      "_bind_to_fused_output";
  static constexpr llvm::StringLiteral kBindToFusedComputeOutput =
      "_bind_to_fused_compute_output";
  static constexpr llvm::StringLiteral kBindToFusedComputeOutputTile =
      "_bind_to_fused_compute_output_tile";
  static constexpr llvm::StringLiteral kBindToFusedOutputTile =
      "_bind_to_fused_output_tile";

  /// Field accessors (return the raw field value, not a call).
  MojoValue staticSpec() const { return value.get(kStaticSpec); }
  MojoValue staticShapeTuple() const { return value.get(kStaticShapeTuple); }
  MojoValue staticStridesTuple() const {
    return value.get(kStaticStridesTuple);
  }
  MojoValue alignment() const { return value.get(kAlignment); }
  MojoValue ioSpec() const { return value.get(kIoSpec); }
  MojoValue rawPtr() const { return value.get(kPtr); }

  /// Method calls (parens included).
  MojoValue unsafePtr() const { return value.get(kUnsafePtr)(); }
  MojoValue shape() const { return value.get(kShape)(); }
  MojoValue strides() const { return value.get(kStrides)(); }
  MojoValue toTileTensor() const { return value.get(kToTileTensor)(); }

  /// Fusion bind method invocations. Returns the method handle (uncalled);
  /// the caller appends `({}, args)` to invoke.
  MojoValue bindToFusedInput() const { return value.get(kBindToFusedInput); }
  MojoValue bindToFusedOutput() const { return value.get(kBindToFusedOutput); }
  MojoValue bindToFusedComputeOutput() const {
    return value.get(kBindToFusedComputeOutput);
  }

  MojoValue value;
};

/// Wrapper around a Mojo `StaticTensorSpec` value.
class StaticTensorSpecRef {
public:
  explicit StaticTensorSpecRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo field / method symbol names on `StaticTensorSpec`.
  static constexpr llvm::StringLiteral kStaticLayout = "static_layout";
  static constexpr llvm::StringLiteral kAlignment = "alignment";
  static constexpr llvm::StringLiteral kRank = "rank";
  static constexpr llvm::StringLiteral kGetInternals = "get_internals";
  /// `shape_tuple` / `strides_tuple` are comptime members of `StaticTensorSpec`
  /// itself (not of the `get_internals()` result, which only carries the
  /// runtime `alignment` / `address_space`).
  static constexpr llvm::StringLiteral kShapeTuple = "shape_tuple";
  static constexpr llvm::StringLiteral kStridesTuple = "strides_tuple";

  MojoValue staticLayout() const { return value.get(kStaticLayout); }
  MojoValue alignment() const { return value.get(kAlignment); }
  MojoValue rank() const { return value.get(kRank); }
  MojoValue shapeTuple() const { return value.get(kShapeTuple); }
  MojoValue stridesTuple() const { return value.get(kStridesTuple); }

  /// `get_internals()` returns a `StaticTensorSpecInternal` carrying the
  /// runtime `alignment` / `address_space`.
  MojoValue getInternals() const { return value.get(kGetInternals)(); }

  MojoValue value;
};

/// Wrapper around a Mojo `OwnedByteBuffer` value: the storage-carrying
/// composite every `!mgp.buffer` value lowers to (a non-owning `MutByteBuffer`
/// view plus an `AnyAsyncValueRef` storage handle, mirroring the C++
/// `TensorBufferRef`). Consumers unwrap `.view` before handing the value to a
/// kernel stub or a register-passable container.
class OwnedByteBufferRef {
public:
  explicit OwnedByteBufferRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo field / method symbol names on `OwnedByteBuffer`.
  static constexpr llvm::StringLiteral kView = "view";
  static constexpr llvm::StringLiteral kUnsafePtr = "unsafe_ptr";

  /// `val.view` — the non-owning `MutByteBuffer` view over the memory.
  MojoValue view() const { return value.get(kView); }

  /// `val.unsafe_ptr()` — the view's raw device data pointer
  /// (`Pointer[Int8, MutAnyOrigin]`).
  MojoValue unsafePtr() const { return value.get(kUnsafePtr)(); }

  MojoValue value;
};

/// Wrapper around a Mojo `OwnedTensor` value: the storage-carrying composite
/// every `!mgp.tensor` value lowers to (a non-owning `DynamicTensor` view plus
/// an `AnyAsyncValueRef` storage handle, the tensor-typed analogue of
/// `OwnedByteBuffer`). Consumers unwrap `.tensor` before handing the value to a
/// kernel stub.
class OwnedTensorRef {
public:
  explicit OwnedTensorRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo field symbol name on `OwnedTensor`.
  static constexpr llvm::StringLiteral kTensor = "tensor";

  /// `val.tensor` — the non-owning `DynamicTensor` view over the memory.
  MojoValue tensor() const { return value.get(kTensor); }

  MojoValue value;
};

/// Wrapper around a Mojo `DeviceGraphBuilder` value: the recorder handed to the
/// build closure of `mgp.device_graph.create`, which every node in the graph is
/// recorded into.
class DeviceGraphBuilderRef {
public:
  explicit DeviceGraphBuilderRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo method / keyword-argument symbol names on `DeviceGraphBuilder`.
  static constexpr llvm::StringLiteral kAddInput = "add_input";
  static constexpr llvm::StringLiteral kAddOutput = "add_output";
  static constexpr llvm::StringLiteral kRegion = "region";
  static constexpr llvm::StringLiteral kContext = "context";
  static constexpr llvm::StringLiteral kRecordingContext = "recording_context";
  static constexpr llvm::StringLiteral kDependencies = "dependencies";

  /// `builder.add_input(input)` — the graph-owned stable location the body
  /// records against in place of the caller's buffer, so a replay does not
  /// depend on the caller's address surviving.
  MojoValue addInput(MojoValue input) const {
    return value.get(kAddInput)(/*parameters=*/{}, {std::move(input)});
  }

  /// `builder.add_output` — the method handle, uncalled; the caller appends
  /// `({}, args)` or hands it to `MojoBuilder::call`.
  MojoValue addOutput() const { return value.get(kAddOutput); }

  /// `builder.region(work, dependencies = deps)` — joins every node `work`
  /// records into a single node usable as one downstream dependency.
  MojoValue region(MojoValue work, MojoValue dependencies) const {
    return value.get(kRegion)(
        /*parameters=*/{},
        {std::move(work),
         MojoValue(kDependencies).equal(std::move(dependencies))});
  }

  /// `builder.context()` — the device context backing the graph.
  MojoValue context() const { return value.get(kContext)(); }

  /// `builder.recording_context()` — a context facade that records enqueued
  /// work into this builder rather than executing it.
  MojoValue recordingContext() const { return value.get(kRecordingContext)(); }

  MojoValue value;
};

/// Wrapper around a Mojo `DeviceGraph` value: an instantiated graph that can be
/// replayed. Construction goes through the static `create` factory below rather
/// than through a value.
class DeviceGraphRef {
public:
  explicit DeviceGraphRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo static-method / keyword-argument symbol names on `DeviceGraph`.
  static constexpr llvm::StringLiteral kCreate = "create";
  static constexpr llvm::StringLiteral kCache = "cache";

  /// `DeviceGraph.create(args...)` — the static factory that creates a builder,
  /// runs the build closure and instantiates the recorded graph. `args` is the
  /// device context, the build closure, then the cache key inputs.
  static MojoValue create(llvm::ArrayRef<MojoValue> args) {
    return MojoValue(kStructDeviceGraph).get(kCreate)(/*parameters=*/{}, args);
  }

  /// `cache = <value>` — the keyword argument selecting the caching overload of
  /// `create`, which reuses a graph an earlier call built.
  static MojoValue cacheArg(MojoValue cache) {
    return MojoValue(kCache).equal(std::move(cache));
  }

  MojoValue value;
};

/// Wrapper around a Mojo `StateContext` value: the per-model state the execute
/// block threads through, holding the cached buffers and the device graph
/// cache.
class StateContextRef {
public:
  explicit StateContextRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo method symbol name on `StateContext`.
  static constexpr llvm::StringLiteral kGetDeviceGraphCache =
      "get_device_graph_cache";

  /// `ctx.get_device_graph_cache()` — a reference to the model's device graph
  /// cache, created on first use. Not copyable, so callers take a pointer to
  /// the result rather than binding it to a variable.
  MojoValue getDeviceGraphCache() const {
    return value.get(kGetDeviceGraphCache)();
  }

  MojoValue value;
};

/// Wrapper around a Mojo `OpaquePointer[MutAnyOrigin]` value.
class OpaquePointerRef {
public:
  explicit OpaquePointerRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo method symbol names on `Pointer` / `OpaquePointer`. The
  /// `unsafe_`-prefixed spellings are the current API; the unprefixed ones
  /// are deprecated aliases kept for source compatibility.
  static constexpr llvm::StringLiteral kBitcast = "unsafe_bitcast";
  static constexpr llvm::StringLiteral kTakePointee = "unsafe_take_pointee";
  /// Type-level constructor (`OpaquePointer.unsafe_dangling()`); invoked on the
  /// type rather than an instance, so there's no member accessor for it.
  static constexpr llvm::StringLiteral kUnsafeDangling = "unsafe_dangling";

  /// `unsafe_bitcast[Type]` — caller does `.bitcast(type)()` to invoke after
  /// supplying the type parameter.
  MojoValue bitcast(const MojoValue &targetType) const {
    return value.get(kBitcast)[targetType];
  }

  /// `unsafe_take_pointee()` reads through the pointer.
  MojoValue takePointee() const { return value.get(kTakePointee)(); }

  MojoValue value;
};

/// Wrapper around a Mojo `Task` value returned by `create_task`.
class MojoTaskRef {
public:
  explicit MojoTaskRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo method symbol name on `Task`.
  static constexpr llvm::StringLiteral kWait = "wait";

  MojoValue wait() const { return value.get(kWait)(); }

  MojoValue value;
};

/// Wrapper around a Mojo `StaticString` / `String` value, exposing the
/// `unsafe_ptr` + `byte_length` accessor pair.
class MojoStringRef {
public:
  explicit MojoStringRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo field / method symbol names on `String` / `StaticString`.
  static constexpr llvm::StringLiteral kUnsafePtr = "unsafe_ptr";
  static constexpr llvm::StringLiteral kByteLength = "byte_length";

  MojoValue unsafePtr() const { return value.get(kUnsafePtr)(); }
  MojoValue byteLength() const { return value.get(kByteLength)(); }

  MojoValue value;
};

/// Wrapper around a Mojo `TileTensor` value. Encapsulates construction and
/// field-access patterns used during tile-based epilogue fusion emission.
class TileTensorRef {
public:
  explicit TileTensorRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo field / method symbol names on `TileTensor` (and the
  /// `Pointer` it is built from).
  static constexpr llvm::StringLiteral kLayout = "layout";
  static constexpr llvm::StringLiteral kUnsafePtr = "unsafe_ptr";
  static constexpr llvm::StringLiteral kUnsafeOriginCast = "unsafe_origin_cast";
  static constexpr llvm::StringLiteral kDstAddressSpace = "dst_address_space";
  static constexpr llvm::StringLiteral kSrcAddressSpace = "src_address_space";
  static constexpr llvm::StringLiteral kAddressSpace = "address_space";
  static constexpr llvm::StringLiteral kAddressSpaceCast =
      "unsafe_address_space_cast";
  static constexpr llvm::StringLiteral kStaticProduct = "static_product";

  /// `val.layout` — the runtime tile layout.
  MojoValue layout() const { return value.get(kLayout); }

  /// `value.unsafe_address_space_cast[Copier.src_address_space]()` — casts a
  /// tile into the copier's source address space so it matches the abstract
  /// `Copier.copy` signature, whose `src` parameter requires
  /// `address_space = Copier.src_address_space`. The staging destination tile
  /// is already allocated in `Copier.dst_address_space` (see `makeStorage`), so
  /// only the source (global-memory) tile needs this cast.
  MojoValue castToSrcAddressSpace() const {
    MojoValue space = MojoValue(kParameterCopierType).get(kSrcAddressSpace);
    return value.get(kAddressSpaceCast)[space]();
  }

  /// Constructs a GENERIC-address-space `TileTensor` over stack-allocated
  /// staging storage (which lives in `Copier.dst_address_space`). The copier
  /// stages a tile into this storage in `Copier.dst_address_space`, but the
  /// tile `*.elementwise(TileTensor[dtype, LayoutType, MutAnyOrigin])` overload
  /// reads it through a GENERIC / `MutAnyOrigin` view. The pointer is cast to
  /// GENERIC *before* wrapping so the resulting tile's `linear_idx_type` is
  /// recomputed for the GENERIC address space (a whole-tile
  /// `unsafe_address_space_cast` would instead preserve the dst-space
  /// `linear_idx_type` and fail to match):
  ///   `TileTensor(storage.unsafe_address_space_cast[AddressSpace.GENERIC]()
  ///               .unsafe_origin_cast[MutAnyOrigin](), layout)`
  static MojoValue fromGenericStorage(const MojoValue &storage,
                                      const MojoValue &layout) {
    MojoValue genericSpace =
        MojoValue(kStructAddressSpace).get(kEnumAddressSpaceGeneric);
    MojoValue ptr = storage.get(kAddressSpaceCast)[genericSpace]().get(
        kUnsafeOriginCast)[kComptimeMutAnyOrigin]();
    return MojoValue(kStructTileTensor)({}, {ptr, layout});
  }

  /// `Int(val.dim[d]())` — compile-time dimension query. Wrapped in `Int`
  /// because `TileTensor.dim()` returns a layout index-type `Scalar`, but the
  /// tile-offset arithmetic mixes it with `Int` (`tile_coords`, strides).
  MojoValue dim(unsigned d) const {
    return MojoValue(kStructInt)({}, {value.get("dim")[d]()});
  }

  /// Emits the backing stack-allocated storage pointer for a local tile, in
  /// the address space the copier stages into:
  ///   `stack_allocation[LayoutType.static_product, dtype,
  ///   address_space = copier.dst_address_space]()`
  static MojoValue makeStorage() {
    MojoValue layoutProduct =
        MojoValue(kParameterLayoutType).get(kStaticProduct);
    // `dst_address_space` is a comptime field of `TileCopier`, so it must be
    // read from the comptime type parameter (`Copier`), not the runtime arg
    // (`copier`).
    MojoValue addressSpace =
        MojoValue(kAddressSpace)
            .equal(MojoValue(kParameterCopierType).get(kDstAddressSpace));
    return MojoValue(kFnStackAllocation)[{
        layoutProduct, MojoValue(kParameterDType), addressSpace}]();
  }

  /// Constructs a mutable `TileTensor` backed by stack-allocated storage with
  /// the given layout:
  ///   `TileTensor(storage.unsafe_origin_cast[MutAnyOrigin](), layout)`
  static MojoValue fromMutStorage(const MojoValue &storage,
                                  const MojoValue &layout) {
    MojoValue ptr = storage.get(kUnsafeOriginCast)[kComptimeMutAnyOrigin]();
    return MojoValue(kStructTileTensor)({}, {ptr, layout});
  }

  /// Constructs a `TileTensor` from a raw pointer into global (HBM) memory:
  ///   `TileTensor((ptr).unsafe_origin_cast[MutUntrackedOrigin](), layout)`
  /// The pointer expression is parenthesised to handle compound expressions
  /// (e.g. `self.ptr_N + offset`) before the method dispatch.
  static MojoValue fromExternalPtr(const MojoValue &ptr,
                                   const MojoValue &layout) {
    MojoValue parenPtr(llvm::formatv("({})", ptr.text).str());
    MojoValue castPtr =
        parenPtr.get(kUnsafeOriginCast)[kComptimeMutUntrackedOrigin]();
    return MojoValue(kStructTileTensor)({}, {castPtr, layout});
  }

  /// The `TileTensor` field type that captures a tensor in a tile-based fusion
  /// struct: `dtype` plus a `TileLayout` rebuilt from the capture's comptime
  /// `StaticTensorSpec` parameter `specParam` (e.g. `Self.spec_0`). Matches the
  /// type of `ManagedTensorSlice.to_tile_tensor()`, so the field accepts it.
  static MojoValue captureFieldType(llvm::StringRef dtypeText,
                                    llvm::StringRef specParam) {
    return MojoValue(
        llvm::formatv("{0}[{1}, {2}[shape_types = "
                      "{3}.static_layout._shape_types, stride_types = "
                      "{3}.static_layout._stride_types], {4}]",
                      kStructTileTensor, dtypeText, kStructTileLayout,
                      specParam, kComptimeMutUntrackedOrigin)
            .str());
  }

  /// Constructs a `TileTensor` from a raw pointer plus fully explicit
  /// dtype/LayoutType/mut, for callers where none of those are statically
  /// known (e.g. rebuilding a captured tensor's own tile from its own
  /// per-capture struct parameters):
  ///   `TileTensor[dtype=<dtype>, LayoutType=<layoutType>,
  ///   origin=UntrackedOrigin[mut=<mut>]](ptr=<ptr>, layout=<layout>)`
  /// Unlike `fromExternalPtr`, mutability is a caller-supplied value rather
  /// than the fixed `MutUntrackedOrigin` alias.
  static MojoValue fromExternalPtrExplicit(const MojoValue &ptr,
                                           const MojoValue &layout,
                                           const MojoValue &dtype,
                                           const MojoValue &layoutType,
                                           const MojoValue &mut) {
    MojoValue origin = MojoValue(
        kStructUntrackedOrigin)[{MojoValue(kParameterMut).equal(mut)}];
    MojoValue tileType = MojoValue(
        kStructTileTensor)[{MojoValue(kParameterDType).equal(dtype),
                            MojoValue(kParameterLayoutType).equal(layoutType),
                            MojoValue(kParameterOrigin).equal(origin)}];
    return tileType(
        {}, {MojoValue("ptr").equal(ptr), MojoValue(kLayout).equal(layout)});
  }

  MojoValue value;
};

/// Type-level helper for `ParameterList`. Wraps the `of` comptime factory used
/// to pack a heterogeneous list of comptime values into a single
/// `ParameterList`. Unlike the other wrappers this holds no instance — `of` is
/// invoked on the `ParameterList` type itself.
class ParameterListRef {
public:
  /// Mojo method symbol name on `ParameterList`.
  static constexpr llvm::StringLiteral kOf = "of";

  /// `ParameterList.of[elements]()`.
  static MojoValue of(llvm::ArrayRef<MojoValue> elements) {
    return MojoValue(kStructParameterList).get(kOf)[elements]();
  }
};

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOIDENTIFIERS_H
