//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGMatchers.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/MDialect/MDialect.h"
#include "gtest/gtest.h"
#include <mlir/IR/BuiltinOps.h>
#include <numeric>

using namespace M;
using namespace M::MO;
using namespace mlir;
using namespace mlir::matchers;

class MatchersTest : public ::testing::Test {
public:
  MatchersTest() : Test(), builder(&ctx), loc(UnknownLoc::get(&ctx)) {}

protected:
  void SetUp() override {
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
    loc.getContext()->allowUnregisteredDialects(true);
    loc.getContext()->loadDialect<MDialect>();
    loc.getContext()->loadDialect<MODialect>();
    loc.getContext()->loadDialect<MOGG::MOGGDialect>();

    SmallVector<Type> inputTypes{
        MO::TensorType::get({1}, FloatType::getF32(&ctx)),
        MO::TensorType::get({1, 2}, FloatType::getF32(&ctx)),
        MO::TensorType::get({1, 2, 3}, FloatType::getF32(&ctx))};
    graph = builder.create<MO::GraphOp>(module.getLoc(), "test", inputTypes,
                                        SmallVector<Type>{});
    builder.setInsertionPointToEnd(graph.getBodyBlock());
  }

  void TearDown() override { module->erase(); }

  MO::TensorValue getFloatConst(ArrayRef<int64_t> shape) {
    auto type = MO::TensorType::get(shape, FloatType::getF32(&ctx));
    SmallVector<APFloat> values(1, APFloat(0.f));
    MO::TensorValue out = builder.create<ConstantOp>(
        loc, ArrayRef<APFloat>(values), FloatType::getF32(&ctx));
    // Overwrite the output type
    out.setType(type);
    return out;
  }

  MO::TensorValue getIntConst(ArrayRef<int64_t> shape, int64_t bitwidth = 64) {
    auto iType =
        IntegerType::get(loc.getContext(), bitwidth, IntegerType::Signed);
    auto type = MO::TensorType::get(shape, iType);
    SmallVector<int64_t> values{1};
    MO::TensorValue out =
        builder.create<ConstantOp>(loc, ArrayRef<int64_t>(values));
    // Overwrite the output type
    out.setType(type);
    return out;
  }

  MLIRContext ctx;
  ModuleOp module;
  GraphOp graph;
  OpBuilder builder;
  Location loc;
};

TEST_F(MatchersTest, Constants) {
  MO::TensorValue x = getIntConst({});
  MO::TensorValue x32 = getIntConst({}, 32);
  MO::TensorValue y = getFloatConst({});
  MO::TensorValue i = getIntConst({1});
  MO::TensorValue j = getFloatConst({1});
  Value c;
  auto match = m_ScalarConstantInt(1);
  auto capture = m_ScalarConstantInt(&c, 1);

  ASSERT_FALSE(match.match(y));
  ASSERT_FALSE(match.match(i));
  ASSERT_FALSE(match.match(j));
  ASSERT_TRUE(match.match(x));
  ASSERT_TRUE(match.match(x32));

  ASSERT_FALSE(capture.match(y));
  ASSERT_FALSE(capture.match(i));
  ASSERT_FALSE(capture.match(j));
  ASSERT_NE(c, x);
  ASSERT_TRUE(capture.match(x));
  ASSERT_EQ(c, x);
  ASSERT_NE(c, x32);
  ASSERT_TRUE(capture.match(x32));
  ASSERT_EQ(c, x32);
}

TEST_F(MatchersTest, Ranked) {
  Value ranked;
  auto pattern = m_Ranked(2, m_Any(&ranked));

  ASSERT_FALSE(pattern.match(graph.getArgument(0)));
  ASSERT_NE(graph.getArgument(1), ranked);
  ASSERT_TRUE(pattern.match(graph.getArgument(1)));
  ASSERT_EQ(graph.getArgument(1), ranked);
  ASSERT_FALSE(pattern.match(graph.getArgument(2)));
}

TEST_F(MatchersTest, MOVector_Matrix) {
  Value v, m;
  auto vector = m_MOVector(m_Any(&v));
  auto matrix = m_MOMatrix(m_Any(&m));

  ASSERT_FALSE(matrix.match(graph.getArgument(0)));
  ASSERT_NE(graph.getArgument(1), m);
  ASSERT_TRUE(matrix.match(graph.getArgument(1)));
  ASSERT_EQ(graph.getArgument(1), m);
  ASSERT_FALSE(matrix.match(graph.getArgument(2)));

  ASSERT_NE(graph.getArgument(0), v);
  ASSERT_TRUE(vector.match(graph.getArgument(0)));
  ASSERT_EQ(graph.getArgument(1), m);
  ASSERT_FALSE(vector.match(graph.getArgument(1)));
  ASSERT_FALSE(vector.match(graph.getArgument(2)));
}

TEST_F(MatchersTest, Optional) {
  Value ranked;
  auto any = m_Any(&ranked);
  auto pattern = m_Optional(m_Ranked(2, any), m_Ranked(1, any));

  ASSERT_NE(graph.getArgument(0), ranked);
  ASSERT_TRUE(pattern.match(graph.getArgument(0)));
  ASSERT_EQ(graph.getArgument(0), ranked);
  ASSERT_TRUE(pattern.match(graph.getArgument(1)));
  ASSERT_EQ(graph.getArgument(1), ranked);
  ASSERT_FALSE(pattern.match(graph.getArgument(2)));
}

TEST_F(MatchersTest, Typed) {
  Value typed;
  auto pattern1 = m_Typed<MO::TensorType>(m_Any(&typed));
  auto pattern2 = m_Typed<RankedTensorType>(m_Any(&typed));

  ASSERT_FALSE(pattern2.match(graph.getArgument(0)));
  ASSERT_NE(graph.getArgument(0), typed);
  ASSERT_TRUE(pattern1.match(graph.getArgument(0)));
  ASSERT_EQ(graph.getArgument(0), typed);
}

TEST_F(MatchersTest, Op) {
  MO::TensorValue x = getIntConst({});
  MO::TensorValue y = getIntConst({});
  MO::TensorValue z = getIntConst({});
  MO::TensorValue add = builder.create<MO::AddOp>(loc, x, y);
  MO::TensorValue mul = builder.create<MO::MulOp>(loc, add, z);
  Operation *pow3 = builder.create<MOGG::ExtOp>(
      loc, mul.getType(), "mop.kernels.pow3", SmallVector<Value>{mul});

  Value lhs, rhs, third;
  Operation *addop, *mulop, *pow3op;

  auto p_x = m_Any(&lhs);
  auto p_y = m_Any(&rhs);
  auto p_z = m_Any(&third);
  auto add_pattern = m_Op(&addop, "mo.add", p_x, p_y);
  auto mul_pattern = m_Op<MO::MulOp>(&mulop, add_pattern, p_z);
  auto pow3_pattern = m_StringAttr(&pow3op, "opName", "mop.kernels.pow3",
                                   m_Op("mogg.ext.op", mul_pattern));

  ASSERT_TRUE(add_pattern.match(add.getDefiningOp()));
  ASSERT_TRUE(mul_pattern.match(mul.getDefiningOp()));
  ASSERT_TRUE(pow3_pattern.match(pow3));
  ASSERT_EQ(pow3, pow3op);
  ASSERT_EQ(mul.getDefiningOp(), mulop);
  ASSERT_EQ(add.getDefiningOp(), addop);
  ASSERT_EQ(x, lhs);
  ASSERT_EQ(y, rhs);
  ASSERT_EQ(z, third);
}
