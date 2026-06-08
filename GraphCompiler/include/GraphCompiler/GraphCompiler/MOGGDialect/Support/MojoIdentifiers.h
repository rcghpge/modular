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

#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
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

constexpr llvm::StringLiteral kMoggTargetParameterName = "target";

/// The name of the output_rank parameter which we support lowering specially.
constexpr llvm::StringLiteral kMoggOutputRankParameterName = "output_rank";

/// The number of DPS (Destination Passing Style) output operands there are.
constexpr llvm::StringLiteral kMoggNumDpsOutputs = "mogg.num_dps_outputs";

/// `lambdas_have_fusion` parameter on accumulation-style fused kernels.
constexpr llvm::StringLiteral kMoggLambdasHaveFusionParameterName =
    "lambdas_have_fusion";

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
constexpr llvm::StringLiteral kLeafDeviceContextList = "DeviceContextList";
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
    "std::DeviceContext";

/// The stored Mojo type symbol name of device-context lists in extensibility
/// kernels (used by multi-device custom ops).
constexpr llvm::StringLiteral kMojoExtensibilityApiDeviceContextListTypeName =
    "std::DeviceContextList";

/// These type names can be confusing. Inside the Graph Compiler code, the
/// types show up like this:
///   - StaticString -> std::StringSlice
///   - String -> std::String
///   - StringSlice -> std::Origin
constexpr llvm::StringLiteral kMojoInternalStaticStringTypeName =
    "std::StringSlice";
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
constexpr llvm::StringLiteral kStructUnsafePointer = "UnsafePointer";
constexpr llvm::StringLiteral kStructOptional = "Optional";
constexpr llvm::StringLiteral kStructString = "String";
constexpr llvm::StringLiteral kStructDeviceContext =
    "std.gpu.host.device_context.DeviceContext";
constexpr llvm::StringLiteral kStructDeviceContextList =
    "std.gpu.host.device_context.DeviceContextList";
constexpr llvm::StringLiteral kStructManagedTensorSlice =
    "extensibility.ManagedTensorSlice";
constexpr llvm::StringLiteral kStructInt = "Int";
constexpr llvm::StringLiteral kStructSIMDSize = "SIMDSize";
constexpr llvm::StringLiteral kStructUInt = "UInt";
constexpr llvm::StringLiteral kStructBool = "Bool";
constexpr llvm::StringLiteral kStructAddressSpace = "std.memory.AddressSpace";
constexpr llvm::StringLiteral kStructDType = "DType";
constexpr llvm::StringLiteral kStructSIMD = "SIMD";
constexpr llvm::StringLiteral kStructTileTensor = "TileTensor";
constexpr llvm::StringLiteral kStructTensorLayout = "TensorLayout";
constexpr llvm::StringLiteral kStructTileCopier = "TileCopier";
constexpr llvm::StringLiteral kStructStateContext =
    "builtin_primitives.primitives.StateContext";
constexpr llvm::StringLiteral kStructStaticTuple = "std.utils.StaticTuple";
constexpr llvm::StringLiteral kStructInlineArray =
    "std.collections.InlineArray";
constexpr llvm::StringLiteral kStructMoggAsyncPackHelper =
    "builtin_primitives.primitives.MoggAsyncPackHelper";
constexpr llvm::StringLiteral kStructScalar = "Scalar";
constexpr llvm::StringLiteral kStructVariadic = "Variadic";
constexpr llvm::StringLiteral kStructParameterList = "ParameterList";
constexpr llvm::StringLiteral kStructStaticTensorSpec =
    "extensibility.managed_tensor_slice.StaticTensorSpec";
constexpr llvm::StringLiteral kStructStaticTensorSpecList =
    "extensibility.StaticTensorSpecList";
constexpr llvm::StringLiteral kStructVariadicTensors =
    "extensibility.VariadicTensors";
constexpr llvm::StringLiteral kStructStaticString = "StaticString";
constexpr llvm::StringLiteral kStructMsFusionPack =
    "extensibility.managed_tensor_slice._FusionPack";
constexpr llvm::StringLiteral kStructMsFusedInputVariadicTensors =
    "extensibility.managed_tensor_slice._FusedInputVariadicTensors";
constexpr llvm::StringLiteral kStructMsFusedOutputVariadicTensors =
    "extensibility.managed_tensor_slice._FusedOutputVariadicTensors";
constexpr llvm::StringLiteral kStructTrace = "std.runtime.tracing.Trace";
constexpr llvm::StringLiteral kStructIntTuple = "layout.IntTuple";
constexpr llvm::StringLiteral kStructCoord = "layout.Coord";
constexpr llvm::StringLiteral kAliasIdx = "layout.Idx";
constexpr llvm::StringLiteral kFnRowMajor = "layout.row_major";

constexpr llvm::StringLiteral kAliasIndexListToTileLayout =
    "extensibility.managed_tensor_slice._IndexListToTileLayout";

//===----------------------------------------------------------------------===//
// Mojo enum values
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kEnumAddressSpaceGeneric = "GENERIC";
constexpr llvm::StringLiteral kEnumIoInput = "extensibility.IO.Input";
constexpr llvm::StringLiteral kEnumTraceLevelOp =
    "std.runtime.tracing.TraceLevel.OP";
/// `DType.int8` member used to spell out a one-byte dtype.
constexpr llvm::StringLiteral kEnumDTypeInt8 = "int8";
/// `DType.int` member: the native index-sized integer dtype used for the
/// dynamic (runtime) elements of layout `Coord`s.
constexpr llvm::StringLiteral kEnumDTypeInt = "int";

//===----------------------------------------------------------------------===//
// Mojo free-function names (emitted into Mojo source)
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kFnStackAllocation = "stack_allocation";
constexpr llvm::StringLiteral kFnToManagedTensorSlice =
    "builtin_primitives.primitives.to_managed_tensor_slice";
constexpr llvm::StringLiteral kFnSimdLoadFromManagedTensorSlice =
    "extensibility.managed_tensor_slice.simd_load_from_managed_tensor_slice";
constexpr llvm::StringLiteral kFnSimdLoadFromTensorPointer =
    "extensibility.managed_tensor_slice.simd_load_from_tensor_pointer";
constexpr llvm::StringLiteral kFnSimdStoreIntoManagedTensorSlice =
    "extensibility.managed_tensor_slice.simd_store_into_managed_tensor_slice";
constexpr llvm::StringLiteral kFnSimdStoreIntoTensorPointer =
    "extensibility.managed_tensor_slice.simd_store_into_tensor_pointer";
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
constexpr llvm::StringLiteral kFnUnpackTensor =
    "builtin_primitives.primitives.unpack_tensor";
constexpr llvm::StringLiteral kFnAllZeros =
    "builtin_primitives.primitives.all_zeros";
constexpr llvm::StringLiteral kFnMoggAsyncPackBorrow =
    "builtin_primitives.primitives.mogg_async_pack_borrow";
constexpr llvm::StringLiteral kFnGetBufferMemStorageHandle =
    "builtin_primitives.primitives.get_buffer_mem_storage_handle";
/// register_passable value-arg `foreach` overload, selected via the `func`
/// kwarg. The legacy `foreach_fusion` overload is reached through
/// `CallForeachOp::getKernelSymName()` instead.
constexpr llvm::StringLiteral kFnMoggForeach =
    "builtin_primitives.primitives.foreach";
constexpr llvm::StringLiteral kFnMsSimdLoadFromTensorPointer =
    "extensibility.managed_tensor_slice.simd_load_from_tensor_pointer";
constexpr llvm::StringLiteral kFnMsSimdStoreIntoTensorPointer =
    "extensibility.managed_tensor_slice.simd_store_into_tensor_pointer";
constexpr llvm::StringLiteral kFnToIndexList = "layout.to_index_list";
constexpr llvm::StringLiteral kFnCreateTask = "std.runtime.asyncrt.create_task";

/// Affinity-aware variant: `_create_task(handle, *, desired_worker_id=<id>)`.
/// Only emitted for GPU-device launches; maps to the package-private function
/// added alongside `task_id_for_device` in `std.runtime.asyncrt`.
constexpr llvm::StringLiteral kFnCreateTaskWithAffinity =
    "std.runtime.asyncrt._create_task";
constexpr llvm::StringLiteral kFnTaskIdForDevice =
    "std.runtime.asyncrt.task_id_for_device";
constexpr llvm::StringLiteral kFnGetSafeTaskId =
    "std.runtime.tracing.get_safe_task_id";

/// Mojo builtins emitted directly.
constexpr llvm::StringLiteral kFnMax = "max";
constexpr llvm::StringLiteral kFnPrint = "print";

// NOTE: Mojo field / method names accessed on values via `.foo` are no longer
// free constants. Each one is a `static constexpr llvm::StringLiteral` member
// of the wrapper class for the type that owns it (see "Wrapper classes for
// high-traffic Mojo types" below).

//===----------------------------------------------------------------------===//
// Trait names (used in struct conformance declarations)
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kTraitInputFusion =
    "extensibility.managed_tensor_slice.InputFusion";
constexpr llvm::StringLiteral kTraitOutputFusion =
    "extensibility.managed_tensor_slice.OutputFusion";
constexpr llvm::StringLiteral kTraitComputeOutputFusion =
    "extensibility.managed_tensor_slice.ComputeOutputFusion";
constexpr llvm::StringLiteral kTraitComputeOutputFusionTile =
    "extensibility.managed_tensor_slice.ComputeOutputFusionTile";
constexpr llvm::StringLiteral kTraitElementwiseFusion =
    "extensibility.managed_tensor_slice.ElementwiseFusion";

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
    "extensibility.managed_tensor_slice._MutableInputTensor";
constexpr llvm::StringLiteral kComptimeUInt8 = "UInt8";
constexpr llvm::StringLiteral kComptimeMutAnyOrigin = "MutAnyOrigin";
constexpr llvm::StringLiteral kComptimeMutExternalOrigin = "MutExternalOrigin";
constexpr llvm::StringLiteral kComptimeNone = "None";
constexpr llvm::StringLiteral kComptimeOptionalErrorEmpty = "Optional[Error]()";
constexpr llvm::StringLiteral kComptimeMutByteBuffer =
    "builtin_primitives.primitives.MutByteBuffer";
constexpr llvm::StringLiteral kComptimeDTypeNoneType = "NoneType";

//===----------------------------------------------------------------------===//
// Parameter names (kernel param inference keys, kwargs in emitted calls)
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kParameterMut = "mut";
constexpr llvm::StringLiteral kParameterInput = "input";
constexpr llvm::StringLiteral kParameterDType = "dtype";
constexpr llvm::StringLiteral kParameterRank = "rank";
constexpr llvm::StringLiteral kParameterIOSpec = "io_spec";
constexpr llvm::StringLiteral kParameterStaticSpec = "static_spec";
constexpr llvm::StringLiteral kParameterSize = "size";
constexpr llvm::StringLiteral kParameterStaticSpecs = "static_specs";
constexpr llvm::StringLiteral kParameterSimdWidth = "simd_width";
constexpr llvm::StringLiteral kParameterElementAlignment = "element_alignment";
constexpr llvm::StringLiteral kParameterElementSize = "element_size";
constexpr llvm::StringLiteral kParameterLayoutType = "LayoutType";
constexpr llvm::StringLiteral kParameterOrigin = "origin";
constexpr llvm::StringLiteral kParameterAddressSpace = "address_space";
constexpr llvm::StringLiteral kParameterLinearIdxType = "linear_idx_type";
constexpr llvm::StringLiteral kParameterTaskId = "task_id";
constexpr llvm::StringLiteral kParameterOutput = "output";
constexpr llvm::StringLiteral kParameterElementType = "element_type";
/// `UnsafePointer.__init__(to=<value>)` kwarg.
constexpr llvm::StringLiteral kParameterTo = "to";
/// Runtime kwargs of `foreach` / `foreach_fusion`: `func`/`tensor`/`ctx`
/// selects `foreach`, `tensor`/`elem`/`ctx` selects `foreach_fusion`. Keep in
/// sync with
/// `max/kernels/src/graph_compiler/builtin_primitives/primitives.mojo`.
constexpr llvm::StringLiteral kParameterForeachFunc = "func";
constexpr llvm::StringLiteral kParameterForeachTensor = "tensor";
constexpr llvm::StringLiteral kParameterForeachElem = "elem";
constexpr llvm::StringLiteral kParameterForeachCtx = "ctx";
/// Argument names used in trait-conforming fusion struct methods
/// (`load`/`store`/`compute`).
constexpr llvm::StringLiteral kParameterSelf = "self";
constexpr llvm::StringLiteral kParameterIdx = "idx";
constexpr llvm::StringLiteral kParameterVal = "val";
constexpr llvm::StringLiteral kParameterTileCoords = "tile_coords";
constexpr llvm::StringLiteral kParameterCopier = "copier";

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
        {"std.utils", ""},
        {"std.runtime", ""},
        {"std.runtime.tracing", ""},
        {"std.memory", ""},
        {"std.collections", ""},
        {"layout", ""},
};

/// TODO(GEX-2488): Find another way to detect that a type is an `IOSpec`
/// without relying on package / type name.
inline bool isMojoIOSpecTypeName(llvm::ArrayRef<mlir::FlatSymbolRefAttr> refs) {
  // After the extensibility fuse, `IOSpec` lives in
  // `extensibility/managed_tensor_slice.mojo` and the Mojo compiler emits
  // it under the dotted path `extensibility.managed_tensor_slice.IOSpec`.
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

/// Wrapper around a Mojo `OpaquePointer[MutAnyOrigin]` value.
class OpaquePointerRef {
public:
  explicit OpaquePointerRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo method symbol names on `UnsafePointer` / `OpaquePointer`.
  static constexpr llvm::StringLiteral kBitcast = "bitcast";
  static constexpr llvm::StringLiteral kTakePointee = "take_pointee";
  /// Type-level constructor (`OpaquePointer.unsafe_dangling()`); invoked on the
  /// type rather than an instance, so there's no member accessor for it.
  static constexpr llvm::StringLiteral kUnsafeDangling = "unsafe_dangling";

  /// `bitcast[Type]` — caller does `.bitcast(type)()` to invoke after
  /// supplying the type parameter.
  MojoValue bitcast(const MojoValue &targetType) const {
    return value.get(kBitcast)[targetType];
  }

  /// `take_pointee()` reads through the pointer.
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
  /// `UnsafePointer` it is built from).
  static constexpr llvm::StringLiteral kLayout = "layout";
  static constexpr llvm::StringLiteral kUnsafePtr = "unsafe_ptr";
  static constexpr llvm::StringLiteral kUnsafeOriginCast = "unsafe_origin_cast";
  static constexpr llvm::StringLiteral kDstAddressSpace = "dst_address_space";
  static constexpr llvm::StringLiteral kAddressSpace = "address_space";
  static constexpr llvm::StringLiteral kStaticProduct = "static_product";

  /// `val.layout` — the runtime tile layout.
  MojoValue layout() const { return value.get(kLayout); }

  /// `val.dim[d]()` — compile-time dimension query.
  MojoValue dim(unsigned d) const { return value.get("dim")[d](); }

  /// Emits the backing stack-allocated storage pointer for a local tile, in
  /// the address space the copier stages into:
  ///   `stack_allocation[LayoutType.static_product, dtype,
  ///   address_space = copier.dst_address_space]()`
  static MojoValue makeStorage() {
    MojoValue layoutProduct =
        MojoValue(kParameterLayoutType).get(kStaticProduct);
    MojoValue addressSpace =
        MojoValue(kAddressSpace)
            .equal(MojoValue(kParameterCopier).get(kDstAddressSpace));
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
  ///   `TileTensor((ptr).unsafe_origin_cast[MutExternalOrigin](), layout)`
  /// The pointer expression is parenthesised to handle compound expressions
  /// (e.g. `self.ptr_N + offset`) before the method dispatch.
  static MojoValue fromExternalPtr(const MojoValue &ptr,
                                   const MojoValue &layout) {
    MojoValue parenPtr(llvm::formatv("({})", ptr.text).str());
    MojoValue castPtr =
        parenPtr.get(kUnsafeOriginCast)[kComptimeMutExternalOrigin]();
    return MojoValue(kStructTileTensor)({}, {castPtr, layout});
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
