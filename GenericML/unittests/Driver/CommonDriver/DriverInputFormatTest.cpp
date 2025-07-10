//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/CommonDriver/DriverInputFormat.h"
#include "llvm/Support/FileSystem.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <fstream>

using namespace M;

namespace {

class TemporaryDirectory {
public:
  TemporaryDirectory();
  TemporaryDirectory(const TemporaryDirectory &) = delete;
  ~TemporaryDirectory();
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  operator testing::AssertionResult() const;
  const std::filesystem::path &path() const;

private:
  static ErrorOr<std::filesystem::path> tryCreate();

  ErrorOr<std::filesystem::path> pathOr;
};

TemporaryDirectory::TemporaryDirectory() : pathOr(tryCreate()) {}

ErrorOr<std::filesystem::path> TemporaryDirectory::tryCreate() {
  auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
  if (!testInfo)
    return Error("unable to retrieve info about current test");
  llvm::SmallString<256> smallPath;
  std::error_code ec = llvm::sys::fs::createUniqueDirectory(
      Twine(testInfo->test_case_name()) + "." + testInfo->name(), smallPath);
  if (ec)
    return Error("unable to create temporary directory for test: " +
                 Twine(ec.message()));
  return std::filesystem::path(std::string_view(smallPath.str()));
}

TemporaryDirectory::~TemporaryDirectory() {
  if (!pathOr.isError())
    std::filesystem::remove_all(*pathOr);
}

TemporaryDirectory::operator testing::AssertionResult() const {
  if (pathOr.isError())
    return testing::AssertionFailure() << pathOr.getError();
  return testing::AssertionSuccess();
}

const std::filesystem::path &TemporaryDirectory::path() const {
  assert(!pathOr.isError());
  return *pathOr;
}

} // namespace
