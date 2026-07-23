//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"

#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/MDialect/MDialect.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::MOGG;

namespace {

TEST(VirtualCpuTargetParse, DescriptorOverridesTripleCpuFeatures) {
  // Cross-arch: an x86_64 descriptor resolved against an arm64 host.
  auto specOr = parseVirtualCpuTarget(
      "triple=x86_64-unknown-linux-gnu;cpu=znver4;features=+avx2,+avx512f",
      "arm64-apple-macosx");
  ASSERT_FALSE(specOr.isError()) << specOr.takeError().get();
  auto spec = specOr.takeValue();
  EXPECT_EQ(spec.triple, "x86_64-unknown-linux-gnu");
  EXPECT_EQ(spec.cpu, "znver4");
  EXPECT_EQ(spec.features, "+avx2,+avx512f");
  EXPECT_TRUE(spec.hasExplicitFeatures);
}

TEST(VirtualCpuTargetParse, BareArchKeepsHostTripleNameOnly) {
  auto specOr = parseVirtualCpuTarget("x86-64-v3", "x86_64-unknown-linux-gnu");
  ASSERT_FALSE(specOr.isError()) << specOr.takeError().get();
  auto spec = specOr.takeValue();
  EXPECT_TRUE(spec.triple.empty());
  EXPECT_EQ(spec.cpu, "x86-64-v3");
  EXPECT_FALSE(spec.hasExplicitFeatures);
}

TEST(VirtualCpuTargetParse, GenericDesugarsAgainstDescriptorTriple) {
  // "generic" de-sugars against the descriptor's triple (x86-64), not the host.
  auto specOr = parseVirtualCpuTarget(
      "triple=x86_64-unknown-linux-gnu;cpu=generic", "arm64-apple-macosx");
  ASSERT_FALSE(specOr.isError()) << specOr.takeError().get();
  EXPECT_EQ(specOr.takeValue().cpu, "x86-64");
}

TEST(VirtualCpuTargetParse, NativeIsRejected) {
  // "native" re-leaks the host CPU; rejected in both forms.
  EXPECT_TRUE(
      parseVirtualCpuTarget("native", "x86_64-unknown-linux-gnu").isError());
  EXPECT_TRUE(parseVirtualCpuTarget("cpu=native", "x86_64-unknown-linux-gnu")
                  .isError());
}

TEST(VirtualCpuTargetParse, UnknownDescriptorFieldIsRejected) {
  EXPECT_TRUE(
      parseVirtualCpuTarget("cpu=x86-64;bogus=1", "x86_64-unknown-linux-gnu")
          .isError());
}

TEST(VirtualCpuTargetParse, DescriptorRequiresCpu) {
  EXPECT_TRUE(parseVirtualCpuTarget("triple=x86_64-unknown-linux-gnu",
                                    "x86_64-unknown-linux-gnu")
                  .isError());
}

TEST(VirtualCpuTargetResolve, DescriptorIsHonoredByGetTargetInfo) {
  // End-to-end: the descriptor must reach getTargetInfo's resolved arch/triple.
  // Host-independent: the resolved arch is the descriptor's, not the host's.
  initLLVM();
  mlir::MLIRContext ctx;
  ctx.loadDialect<MDialect>();

  llvm::scope_exit restore{[prev = M::Driver::getVirtualCpuTarget()] {
    M::Driver::setVirtualCpuTarget(prev);
  }};
  M::Driver::setVirtualCpuTarget(
      "triple=x86_64-unknown-linux-gnu;cpu=x86-64;features=+sse2");

  auto targetOr = getTargetInfo(&ctx);
  ASSERT_FALSE(targetOr.isError()) << targetOr.takeError().get();
  auto target = targetOr.takeValue();
  EXPECT_EQ(target.getArch(), "x86-64");
  EXPECT_TRUE(llvm::StringRef(target.getTripleStr()).starts_with("x86_64"))
      << target.getTripleStr();
}

} // namespace
