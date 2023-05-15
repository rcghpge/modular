//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOSupport/Broadcast.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "Support/MDialect/MDialect.h"
#include "gtest/gtest.h"
#include <mlir/IR/BuiltinOps.h>
#include <numeric>

using namespace M;
using namespace M::MO;

class BroadcastTest : public ::testing::Test {
public:
  BroadcastTest() : Test(), builder(&ctx), loc(UnknownLoc::get(&ctx)) {}

protected:
  void SetUp() override {
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
    loc.getContext()->allowUnregisteredDialects(true);
    loc.getContext()->loadDialect<MDialect>();
    loc.getContext()->loadDialect<MODialect>();
  }

  void TearDown() override { module->erase(); }

  MO::TensorValue getTestValue(ShapeAttr shape) {
    auto type = MO::TensorType::get(shape, FloatType::getF32(&ctx));
    SmallVector<APFloat> values(1, APFloat(0.f));
    MO::TensorValue out = builder.create<ConstantOp>(
        loc, ArrayRef<APFloat>(values), FloatType::getF32(&ctx));
    // Overwrite the output type
    out.setType(type);
    return out;
  }

  MO::TensorValue getTestValue(ArrayRef<int64_t> shape) {
    auto shapeTy = builder.getType<ShapeType>();
    auto shapeAttr = ShapeAttr::get(shape, shapeTy);
    return getTestValue(shapeAttr);
  }

  MO::TensorValue getTestValue(ArrayRef<TypedAttr> shape) {
    auto shapeTy = builder.getType<ShapeType>();
    auto shapeAttr = builder.getAttr<ShapeAttr>(shape, shapeTy);
    return getTestValue(shapeAttr);
  }

  TypedAttr getParametricDim(const std::string &name) {
    auto idxTy = builder.getIndexType();
    TypedAttr typeAttr = KGEN::ParamDeclRefAttr::get(name, idxTy);
    return typeAttr;
  }

  TypedAttr getUnknownDim() {
    auto idxTy = builder.getIndexType();
    TypedAttr typeAttr = KGEN::UnknownAttr::get(idxTy);
    return typeAttr;
  }

  TypedAttr getIntegerDim(int64_t v) {
    auto idxTy = builder.getIndexType();
    return IntegerAttr::get(idxTy, v);
  }

  MLIRContext ctx;
  ModuleOp module;
  OpBuilder builder;
  Location loc;
};

TEST_F(BroadcastTest, Failure) {
  MO::TensorValue x = getTestValue({1, 3});
  MO::TensorValue y = getTestValue({1, 1});
  EXPECT_DEATH(
      broadcastTo(builder, loc, x, y),
      "don't use this helper unless you can guarantee x broadcasts to y");
}

TEST_F(BroadcastTest, Basic) {
  MO::TensorValue x = getTestValue({1, 1});
  MO::TensorValue y = getTestValue({1, 3});
  MO::TensorValue bcastX = broadcastTo(builder, loc, x, y);
  SmallVector<int64_t> refDims{1ll, 3ll};
  ASSERT_EQ(bcastX.getType().getIntDims(), refDims);
}

TEST_F(BroadcastTest, Multi) {
  MO::TensorValue x = getTestValue({5, 1, 2});
  MO::TensorValue y = getTestValue({1, 1, 3, 1});
  auto [bcastX, bcastY] = reifyImplicitBroadcasting(builder, loc, x, y);
  SmallVector<int64_t> refDims{1ll, 5ll, 3ll, 2ll};
  ASSERT_EQ(bcastX.getType().getIntDims(), refDims);
  ASSERT_EQ(bcastY.getType().getIntDims(), refDims);
}

TEST_F(BroadcastTest, Dynamic1) {
  MO::TensorValue x = getTestValue({5, 1});
  MO::TensorValue y = getTestValue(SmallVector<TypedAttr>{
      getIntegerDim(1ll), getUnknownDim(), getIntegerDim(1ll)});
  auto [bcastX, bcastY] = reifyImplicitBroadcasting(builder, loc, x, y);
  SmallVector<int64_t> refDims{1, 5ll, 1ll};
  ASSERT_EQ(bcastX.getType().getIntDims(), refDims);
  ASSERT_EQ(bcastY.getType().getIntDims(), refDims);
}

TEST_F(BroadcastTest, Dynamic2) {
  MO::TensorValue x = getTestValue(SmallVector<TypedAttr>{
      getIntegerDim(5ll), getUnknownDim(), getUnknownDim()});
  MO::TensorValue y = getTestValue(SmallVector<TypedAttr>{
      getIntegerDim(1ll), getUnknownDim(), getIntegerDim(1ll)});
  auto [bcastX, bcastY] = reifyImplicitBroadcasting(builder, loc, x, y);
  auto refDims = SmallVector<TypedAttr>{getIntegerDim(5ll), getUnknownDim(),
                                        getUnknownDim()};
  ASSERT_EQ(bcastX.getType().getDims(), ArrayRef<TypedAttr>(refDims));
  ASSERT_EQ(bcastY.getType().getDims(), ArrayRef<TypedAttr>(refDims));
}

TEST_F(BroadcastTest, Parametric) {
  auto shapeTy = builder.getType<ShapeType>();
  SmallVector<TypedAttr> shapeX{getParametricDim("n"), getIntegerDim(1),
                                getParametricDim("m")};

  SmallVector<TypedAttr> shapeY{getParametricDim("n"), getUnknownDim(),
                                getIntegerDim(1ll)};

  SmallVector<TypedAttr> shapeRef{getParametricDim("n"), getUnknownDim(),
                                  getParametricDim("m")};
  ShapeAttr shapeRefAttr = builder.getAttr<ShapeAttr>(shapeRef, shapeTy);

  MO::TensorValue x = getTestValue(shapeX);
  MO::TensorValue y = getTestValue(shapeY);

  auto [bcastX, bcastY] = reifyImplicitBroadcasting(builder, loc, x, y);
  ASSERT_EQ(bcastX.getType().getTypedShapeAttr(), shapeRefAttr);
  ASSERT_EQ(bcastY.getType().getTypedShapeAttr(), shapeRefAttr);
}

TEST_F(BroadcastTest, ReshapeAndBroadcast) {
  MO::TensorValue x = getTestValue({5});
  MO::TensorValue y = getTestValue({3, 5, 1});

  auto [bcastX1, bcastY1] = reifyImplicitBroadcasting(builder, loc, x, y);

  SmallVector<int64_t> refDims1{3, 5, 5};
  SmallVector<int64_t> refDims2{3, 5, 1};

  ASSERT_EQ(bcastX1.getType().getIntDims(), refDims1);
  ASSERT_EQ(bcastY1.getType().getIntDims(), refDims1);

  auto bcastX2 =
      reshapeAndBroadcastTo(builder, loc, x, SmallVector<int64_t>{1, 5, 1}, y);

  ASSERT_EQ(bcastX2.getType().getIntDims(), refDims2);
}
