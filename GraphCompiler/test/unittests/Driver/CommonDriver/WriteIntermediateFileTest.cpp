//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Driver/Common/Utils.h"
#include "Support/CommonCLOptions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/FileSystem.h"
#include "gtest/gtest.h"

#include <filesystem>

using namespace M;

namespace {

// Configures `opts` with saveTemps enabled and tempsDir set to `dir`.
static void setSaveTempsOptions(CommonOptions &opts, const std::string &dir) {
  opts.saveTemps = true;
  opts.tempsDir = dir;
}

// Creates a unique temporary directory that is removed when the object is
// destroyed.
class TemporaryDirectory {
public:
  TemporaryDirectory() {
    llvm::SmallString<256> path;
    std::error_code ec =
        llvm::sys::fs::createUniqueDirectory("WriteIntermediateFileTest", path);
    if (!ec)
      path_ = std::string(path.str());
  }

  ~TemporaryDirectory() {
    if (!path_.empty())
      std::filesystem::remove_all(path_);
  }

  bool valid() const { return !path_.empty(); }
  const std::string &path() const { return path_; }

private:
  std::string path_;
};

TEST(WriteIntermediateFileIfRequestedTest, CustomBaseNameNamesFileAccordingly) {
  mlir::MLIRContext ctx;
  mlir::OpBuilder builder(&ctx);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());

  TemporaryDirectory tmpDir;
  ASSERT_TRUE(tmpDir.valid());

  CommonOptions opts;
  setSaveTempsOptions(opts, tmpDir.path());
  writeIntermediateFileIfRequested("stage", *module, &opts, BufferRef(),
                                   "mymodel");

  EXPECT_TRUE(std::filesystem::exists(tmpDir.path() + "/mymodel.stage.mlir"));
}

TEST(WriteIntermediateFileIfRequestedTest, DefaultBaseNameUsesTemp) {
  mlir::MLIRContext ctx;
  mlir::OpBuilder builder(&ctx);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());

  TemporaryDirectory tmpDir;
  ASSERT_TRUE(tmpDir.valid());

  CommonOptions opts;
  setSaveTempsOptions(opts, tmpDir.path());
  writeIntermediateFileIfRequested("stage", *module, &opts, BufferRef());

  EXPECT_TRUE(std::filesystem::exists(tmpDir.path() + "/temp.stage.mlir"));
}

} // namespace
