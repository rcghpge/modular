//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/GraphCompilerCLOptions.h"
#include "gtest/gtest.h"

using namespace M::MO;

class OverrideLoweringOptionsTest : public ::testing::Test {};

/// Regression test for #30415.
/// This tests that after overriding lowering options with command-line options,
/// `opt.hasValue()` returns true.
TEST_F(OverrideLoweringOptionsTest, HasValue) {
  MoToMgpOptions compilerOptions;
  EXPECT_FALSE(compilerOptions.cacheDir.hasValue());

  GraphCompilerOptions clOptions;
  clOptions.cacheDir = "foo";
  clOptions.overrideLoweringOptions(compilerOptions);
  EXPECT_TRUE(compilerOptions.cacheDir.hasValue());
}
