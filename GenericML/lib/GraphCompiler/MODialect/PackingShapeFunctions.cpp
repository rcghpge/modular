//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"

#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace M::MO;
using namespace mlir;

//===----------------------------------------------------------------------===//
// pack_matmul_b_shape_func
//===----------------------------------------------------------------------===//

struct PackMatmulShapeKgenAttrs {
  SmallVector<int64_t> aShape;
  SmallVector<int64_t> bShape;
  SmallVector<int64_t> cShape;

  TypeAttr aType;
  TypeAttr bType;
  TypeAttr cType;

  BoolAttr transposeIn0;
};

static std::optional<SmallVector<int64_t>> unpackDims(DictionaryAttr attr,
                                                      StringRef name) {
  if (auto dimsAttr = attr.getAs<DenseIntElementsAttr>(name))
    return llvm::to_vector(dimsAttr.getValues<int64_t>());

  return {};
}

template <typename AttrType>
static std::optional<AttrType> unpack(DictionaryAttr attr, StringRef name) {
  auto result = attr.getAs<AttrType>(name);
  if (!result)
    return {};
  return result;
}

std::optional<PackMatmulShapeKgenAttrs>
unpackMatmulKgenAttrs(DictionaryAttr kgenAttrs) {
  PackMatmulShapeKgenAttrs result;

#define PARSE_FIELD(field, name, parser)                                       \
  auto(field) = parser(kgenAttrs, name);                                       \
  if (!(field))                                                                \
    return {};                                                                 \
  result.field = std::move(*(field));

  PARSE_FIELD(aShape, "a_shape", unpackDims);
  PARSE_FIELD(bShape, "b_shape", unpackDims);
  PARSE_FIELD(cShape, "c_shape", unpackDims);

  PARSE_FIELD(aType, "a_type", unpack<TypeAttr>);
  PARSE_FIELD(bType, "b_type", unpack<TypeAttr>);
  PARSE_FIELD(cType, "c_type", unpack<TypeAttr>);

  PARSE_FIELD(transposeIn0, "transpose_in_0", unpack<BoolAttr>);

#undef PARSE_FIELD

  return result;
}

class TargetQueries {
public:
  TargetQueries(TargetInfoAttr target) : targetInfo{target} {
    assert(target != nullptr);
  }

  bool hasFeature(StringRef feat) const {
    return targetInfo && targetInfo.hasFeature(feat);
  }

  bool isArch(StringRef arch) const {
    return targetInfo && targetInfo.getArch() == arch;
  }

  bool isOS(StringRef os) const {
    return targetInfo && targetInfo.getOS() == os;
  }

  int64_t simdBitWidth() const {
    if (!targetInfo)
      return 0;
    return targetInfo.getSimdBitWidth();
  }

  int64_t simdWidthOf(Type type) const {
    return simdBitWidth() / type.getIntOrFloatBitWidth();
  }

  bool hasNeonInt8Dotprod() const {
    return hasFeature("neon") && hasFeature("dotprod");
  }

  bool hasNeonInt8Matmul() const {
    return hasFeature("neon") && hasFeature("i8mm");
  }

  bool osIsMacos() const { return isOS("darwin") || isOS("macosx"); }

  bool isNeoverseN1() const { return isArch("neoverse-n1"); }

private:
  TargetInfoAttr targetInfo;
};

static bool useVnniFn(const TargetQueries &target, Type aType, Type bType,
                      Type cType) {
  if (target.hasNeonInt8Dotprod() && !target.hasNeonInt8Matmul()) {
    return ((aType.isSignedInteger(8) && bType.isSignedInteger(8)) ||
            (aType.isUnsignedInteger(8) && bType.isUnsignedInteger(8))) &&
           cType.isSignedInteger(32);
  } else if (target.hasFeature("avx2")) {
    return aType.isUnsignedInteger(8) && bType.isSignedInteger(8) &&
           cType.isSignedInteger(32);
  } else {
    return false;
  }
}

static bool useI8mmFn(const TargetQueries &target, Type aType, Type bType,
                      Type cType) {
  bool hasNeonInt8Matmul =
      target.hasFeature("neon") && target.hasFeature("i8mm");

  auto neonTypesMatch =
      (aType.isUnsignedInteger(8) && bType.isUnsignedInteger(8)) ||
      (aType.isUnsignedInteger(8) && bType.isSignedInteger(8)) ||
      (aType.isSignedInteger(8) && bType.isSignedInteger(8));

  return hasNeonInt8Matmul && neonTypesMatch;
}

static bool getKernelType(const TargetQueries &target, int64_t m, int64_t n,
                          int64_t k) {
  if (target.hasFeature("avx512f"))
    return m > 0 && m <= 32;

  if (target.hasFeature("neon")) {
    if (target.isNeoverseN1())
      return (k % 4096) == 0;
    else
      return m > 32;
  }

  return false;
}

struct MicroKernelShape {
  int64_t simdRows;
  int64_t simdCols;
};

static MicroKernelShape getMatmulKernelShapeX86(const TargetQueries &target,
                                                bool kernelType) {
  if (target.hasFeature("avx512f")) {
    if (kernelType)
      return MicroKernelShape{8, 3};
    return MicroKernelShape{6, 4};
  }

  return MicroKernelShape{4, 3};
}

static MicroKernelShape getMatmulKernelShapeARM(const TargetQueries &target,
                                                Type aType, Type bType,
                                                Type cType, bool kernelType) {
  if (target.isNeoverseN1()) {
    if (kernelType)
      return MicroKernelShape{4, 4};
    else
      return MicroKernelShape{8, 2};
  } else {
    bool useI8mm = useI8mmFn(target, aType, bType, cType);
    if (useI8mm)
      return MicroKernelShape{4, 6};
    else if (kernelType)
      return MicroKernelShape{6, 4};
    else
      return MicroKernelShape{8, 2};
  }
}

static MicroKernelShape getMatmulKernelShape(const TargetQueries &target,
                                             Type aType, Type bType, Type cType,
                                             bool kernelType) {
  if (target.hasFeature("neon"))
    return getMatmulKernelShapeARM(target, aType, bType, cType, kernelType);
  return getMatmulKernelShapeX86(target, kernelType);
}

struct KernelConfig {
  int64_t kernelRows;
  int64_t kernelCols;
  int64_t simdSize;
};

static KernelConfig getKernelConfig(const TargetQueries &target, Type aType,
                                    Type bType, Type cType, bool kernelType) {
  int64_t simdSize = target.simdWidthOf(cType);

  MicroKernelShape kernelShape =
      getMatmulKernelShape(target, aType, bType, cType, kernelType);

  return KernelConfig{/*kernelRows=*/kernelShape.simdRows,
                      /*kernelCols=*/kernelShape.simdCols * simdSize,
                      /*simdSize=*/simdSize};
}

static int64_t getPackDataSize(const TargetQueries &target, Type type) {
  constexpr int64_t KB = 1024;

  int64_t sizeOfType = type.getIntOrFloatBitWidth() / CHAR_BIT;

  // TODO: How to implement this case?
  // @parameter
  // if is_debug_build():
  //     # Only use the large cache size for release build as debug build may
  //     # contain additional data could cause stack overflow.
  //     # Restrict it to 4K.
  //     return 4 * KB // sizeof[type]()
  //
  // This is not quite right, as it is whether this source file is compiled
  // in debug mode, not whether the Mojo code is compiled in debug mode.
  // #ifndef NDEBUG
  //   return (4 * KB) / sizeOfType;
  // #endif

  if (target.osIsMacos())
    return (64 * KB) / sizeOfType;

  if (target.hasFeature("neon") || target.hasFeature("avx512f"))
    return (512 * KB) / sizeOfType;

  return (256 * KB) / sizeOfType;
}

static int64_t ceilDiv(int64_t numerator, int64_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

static int64_t alignDown(int64_t value, int64_t alignment) {
  return (value / alignment) * alignment;
}

static int64_t alignUp(int64_t numerator, int64_t alignment) {
  return ceilDiv(numerator, alignment) * alignment;
}

static int64_t getMatmulArchFactor(bool useVnni, bool useI8mm) {
  if (useI8mm)
    return 8;
  if (useVnni)
    return 4;
  return 1;
}

static std::pair<int64_t, int64_t>
calculateTileNK(const TargetQueries &target, Type aType, Type bType, Type cType,
                int64_t kernelCols, int64_t n, int64_t k) {
  int64_t packCacheSize = getPackDataSize(target, bType);
  bool useVnni = useVnniFn(target, aType, bType, cType);
  bool useI8mm = useI8mmFn(target, aType, bType, cType);
  int64_t factor = getMatmulArchFactor(useVnni, useI8mm);

  int64_t leastTileN = kernelCols;

  int64_t largestTileK = alignDown(packCacheSize / leastTileN, factor);

  int64_t tileK = std::min(largestTileK, alignUp(k, factor));

  int64_t maxTileNInInnerSize = packCacheSize / tileK / kernelCols;

  int64_t fullDataTileNInInnerSize = ceilDiv(n, kernelCols);

  int64_t tileNInInnerSize =
      std::min(maxTileNInInnerSize, fullDataTileNInInnerSize);

  int64_t tileN = tileNInInnerSize * kernelCols;

  return {tileN, tileK};
}

static bool useAppleAccelerateLib(const TargetQueries &target, Type cType,
                                  Type aType, Type bType) {
  return target.osIsMacos() &&
         (aType.isF32() && bType.isF32() && cType.isF32());
}

std::optional<SmallVector<int64_t>>
materializePackMatmulShapeFunc(InvokeShapeFuncOp invoke,
                               TargetInfoAttr targetAttr) {
  std::optional<PackMatmulShapeKgenAttrs> maybeAttrs =
      unpackMatmulKgenAttrs(invoke.getKgenParamsAttr());

  if (!maybeAttrs) {
    invoke.emitError("malformed 'pack_matmul_b_shape_func' shape invocation");
    return {};
  }

  PackMatmulShapeKgenAttrs &attrs = *maybeAttrs;
  Type aType = attrs.aType.getValue();
  Type bType = attrs.bType.getValue();
  Type cType = attrs.cType.getValue();

  ArrayRef<int64_t> aShape = attrs.aShape;
  ArrayRef<int64_t> bShape = attrs.bShape;

  // Unable to materialize the packing logic for a dynamic shape, as we cannot
  // return to the runtime size in the materialize shape calculation.
  if (ShapedType::isDynamicShape(bShape))
    return {};

  // `getKernelType` expects `m == 0` for dynamic M.
  int64_t kernelTypeM = std::max<int64_t>(0, aShape[0]);
  int64_t n = attrs.transposeIn0.getValue() ? bShape[0] : bShape[1];
  int64_t k = attrs.transposeIn0.getValue() ? bShape[1] : bShape[0];

  TargetQueries target(targetAttr);

  bool kernelType = getKernelType(target, kernelTypeM, n, k);
  auto kernelConfig = getKernelConfig(target, aType, bType, cType, kernelType);

  auto tileNK =
      !attrs.transposeIn0.getValue()
          ? calculateTileNK(target, aType, bType, cType,
                            kernelConfig.kernelCols, bShape[1], bShape[0])
          : calculateTileNK(target, aType, bType, cType,
                            kernelConfig.kernelCols, bShape[0], bShape[1]);

  SmallVector<int64_t, 2> output = {bShape[0], bShape[1]};

  if (attrs.transposeIn0.getValue())
    std::swap(output[0], output[1]);

  if (!useAppleAccelerateLib(target, cType, aType, bType)) {
    bool useVnni = useVnniFn(target, aType, bType, cType);
    bool useI8mm = useI8mmFn(target, aType, bType, cType);
    int64_t factor = getMatmulArchFactor(useVnni, useI8mm);

    output[0] = alignUp(output[0], factor);
    output[1] = alignUp(output[1], std::get<0>(tileNK));
  } else {
    std::swap(output[0], output[1]);
  }

  return output;
}
