//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MEFCacheKey.h"

#include "GraphCompiler/GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/GraphCompiler/Passes.h"
#include "GraphCompiler/MOFOptions.h"
#include "Support/Context.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"

#include "gtest/gtest.h"

#include <fstream>

using namespace M;

namespace {

/// Test fixture that constructs the inputs the hasher needs: an MLIR
/// context with the dialects ``computeMEFCacheKey`` touches, default
/// `MOFLoweringOptions` / `GCPipelineOptions`, a fresh `Context`.
class MEFCacheKeyTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<mlir::BuiltinDialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<MOGG::MOGGDialect>();
    ctx.loadDialect<MO::MODialect>();
    maxContext = ContextRef::create();
  }

  /// Parse the given MLIR source as a `ModuleOp`. Returns an owned op so
  /// the caller can compute a key against it.
  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef src) {
    auto mod = mlir::parseSourceString<mlir::ModuleOp>(src, &ctx);
    EXPECT_TRUE(mod) << "failed to parse MLIR fixture:\n" << src.str();
    return mod;
  }

  /// Build a hasher with default-constructed options. Tests skip if
  /// `build()` fails — the hasher's prefix needs the built-in kernel
  /// packages on disk, which the bazel test sandbox may not surface.
  /// The optional `configureGC` callback runs before the build, so a
  /// caller can mutate `gcPipelineOptions` (e.g. populate
  /// `kernelLibraryPaths`).
  std::optional<MEFCacheKeyHasher>
  tryBuild(llvm::function_ref<void(MO::GCPipelineOptions &)> configureGC = {}) {
    MOF::MOFLoweringOptions mof;
    MO::GCPipelineOptions gc;
    if (configureGC)
      configureGC(gc);
    auto hasherOr = MEFCacheKeyHasher::build(ctx, mof, gc, maxContext);
    if (hasherOr.isError())
      return std::nullopt;
    return std::move(hasherOr).takeValue();
  }

  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ContextRef maxContext;
};

constexpr size_t kBLAKE3DigestBytes = 32;

constexpr llvm::StringLiteral kEmptyModule = "module {}";
// Symbols get canonicalized by `normalizeAllNames`, so the names alone
// don't differentiate modules. These fixtures differ in signature shape
// — argument count and type — which survives normalization.
constexpr llvm::StringLiteral kFuncFooModule = R"mlir(
  module {
    func.func private @foo(%a: i32) -> i32
  }
)mlir";
constexpr llvm::StringLiteral kFuncBarModule = R"mlir(
  module {
    func.func private @bar(%a: i64, %b: i64) -> i64
  }
)mlir";

} // namespace

TEST_F(MEFCacheKeyTest, KeyHasExpectedLength) {
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto keyOr = hasher->keyForModule(*mod);
  ASSERT_FALSE(keyOr.isError()) << keyOr.takeError().get();
  EXPECT_EQ(keyOr.takeValue().size(), kBLAKE3DigestBytes);
}

TEST_F(MEFCacheKeyTest, KeyIsDeterministic) {
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kFuncFooModule);
  auto first = hasher->keyForModule(*mod);
  auto second = hasher->keyForModule(*mod);
  ASSERT_FALSE(first.isError());
  ASSERT_FALSE(second.isError());
  EXPECT_EQ(first.takeValue(), second.takeValue());
}

TEST_F(MEFCacheKeyTest, DifferentModulesProduceDifferentKeys) {
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto a = parse(kFuncFooModule);
  auto b = parse(kFuncBarModule);
  auto keyA = hasher->keyForModule(*a);
  auto keyB = hasher->keyForModule(*b);
  ASSERT_FALSE(keyA.isError());
  ASSERT_FALSE(keyB.isError());
  EXPECT_NE(keyA.takeValue(), keyB.takeValue());
}

TEST_F(MEFCacheKeyTest, HasherStateIsImmutable) {
  // Computing a different key in between mustn't disturb the BLAKE3 state.
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto a = parse(kFuncFooModule);
  auto b = parse(kFuncBarModule);
  auto first = hasher->keyForModule(*a);
  (void)hasher->keyForModule(*b);
  auto third = hasher->keyForModule(*a);
  ASSERT_FALSE(first.isError());
  ASSERT_FALSE(third.isError());
  EXPECT_EQ(first.takeValue(), third.takeValue());
}

TEST_F(MEFCacheKeyTest, KernelLibraryPathInvalidatesKey) {
  // Two hashers differing only in `kernelLibraryPaths` should produce
  // distinct keys for the same module. Use a temp file with known
  // non-empty contents as a stand-in "kernel library" — pointing at a
  // real, readable path the hasher can open.
  llvm::SmallString<128> tmpPath;
  int tmpFd = -1;
  auto ec = llvm::sys::fs::createTemporaryFile("mef-cache-key-test", "dat",
                                               tmpFd, tmpPath);
  ASSERT_FALSE(ec) << "failed to create temp file: " << ec.message();
  {
    std::ofstream out(tmpPath.c_str(), std::ios::binary);
    out << "synthetic kernel library bytes for MEFCacheKeyTest";
  }
  llvm::sys::fs::closeFile(tmpFd);
  llvm::scope_exit cleanup{[&] { llvm::sys::fs::remove(tmpPath); }};

  auto plain = tryBuild();
  if (!plain)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  std::string tmpPathStr(tmpPath.str());
  auto withExtraHasher = tryBuild(
      [&](MO::GCPipelineOptions &gc) { gc.kernelLibraryPaths = {tmpPathStr}; });
  if (!withExtraHasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto plainKey = plain->keyForModule(*mod);
  auto extraKey = withExtraHasher->keyForModule(*mod);
  ASSERT_FALSE(plainKey.isError());
  ASSERT_FALSE(extraKey.isError());
  EXPECT_NE(plainKey.takeValue(), extraKey.takeValue());
}

TEST_F(MEFCacheKeyTest, ConvenienceMatchesIncremental) {
  // `computeMEFCacheKey` is documented to equal
  // `MEFCacheKeyHasher::build(...).keyForModule(...)`. Pin the
  // equivalence so any drift in the convenience surface is caught.
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kFuncFooModule);
  auto incrementalKey = hasher->keyForModule(*mod);
  ASSERT_FALSE(incrementalKey.isError());

  MOF::MOFLoweringOptions mof;
  MO::GCPipelineOptions gc;
  auto convenienceKey = computeMEFCacheKey(*mod, mof, gc, maxContext);
  ASSERT_FALSE(convenienceKey.isError());

  EXPECT_EQ(incrementalKey.takeValue(), convenienceKey.takeValue());
}
