//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Driver/DriverInputFormat.h"
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

static testing::AssertionResult touchFile(const std::filesystem::path &path) {
  std::ofstream of(path, std::ios::app);
  if (of.fail())
    return testing::AssertionFailure()
           << "Unable to open path " << path << " for appending";
  return testing::AssertionSuccess();
}

TEST(TFSavedModelCanonicalPath, basic) {
  TemporaryDirectory temp;
  ASSERT_TRUE(temp);
  auto path = temp.path();
  ASSERT_TRUE(std::filesystem::create_directory(path / "mymodel"));
  ASSERT_TRUE(touchFile(path / "mymodel" / "saved_model.pb"));
  auto expected = path / "mymodel";

  auto canonOr = canonicalizeTFSavedModelPath(path / "mymodel");
  ASSERT_FALSE(canonOr.isError()) << canonOr.getError();
  EXPECT_EQ(expected, *canonOr);

  canonOr = canonicalizeTFSavedModelPath(path / "mymodel" / "saved_model.pb");
  ASSERT_FALSE(canonOr.isError()) << canonOr.getError();
  EXPECT_EQ(expected, *canonOr);
}

TEST(TFSavedModelCanonicalPath, cpb) {
  TemporaryDirectory temp;
  ASSERT_TRUE(temp);
  auto path = temp.path();
  ASSERT_TRUE(std::filesystem::create_directory(path / "mymodel"));
  ASSERT_TRUE(touchFile(path / "mymodel" / "saved_model.cpb"));
  auto expected = path / "mymodel";

  auto canonOr = canonicalizeTFSavedModelPath(path / "mymodel");
  ASSERT_FALSE(canonOr.isError()) << canonOr.getError();
  EXPECT_EQ(expected, *canonOr);

  canonOr = canonicalizeTFSavedModelPath(path / "mymodel" / "saved_model.cpb");
  ASSERT_FALSE(canonOr.isError()) << canonOr.getError();
  EXPECT_EQ(expected, *canonOr);
}

TEST(TFSavedModelCanonicalPath, pbtxt) {
  TemporaryDirectory temp;
  ASSERT_TRUE(temp);
  auto path = temp.path();
  ASSERT_TRUE(std::filesystem::create_directory(path / "mymodel"));
  ASSERT_TRUE(touchFile(path / "mymodel" / "saved_model.pbtxt"));
  auto expected = path / "mymodel";

  auto canonOr = canonicalizeTFSavedModelPath(path / "mymodel");
  ASSERT_FALSE(canonOr.isError()) << canonOr.getError();
  EXPECT_EQ(expected, *canonOr);

  canonOr =
      canonicalizeTFSavedModelPath(path / "mymodel" / "saved_model.pbtxt");
  ASSERT_FALSE(canonOr.isError()) << canonOr.getError();
  EXPECT_EQ(expected, *canonOr);
}

TEST(TFSavedModelCanonicalPath, confusingNestedPbName) {
  TemporaryDirectory temp;
  ASSERT_TRUE(temp);
  auto path = temp.path();
  // Weird case: The user for some reason put their model in a DIRECTORY named
  // saved_model.pb, which, because it's a saved model, also contains a
  // saved_model.pb FILE.  Make sure canonicalization still works if the user
  // does this.
  ASSERT_TRUE(std::filesystem::create_directory(path / "saved_model.pb"));
  ASSERT_TRUE(touchFile(path / "saved_model.pb" / "saved_model.pb"));
  auto expected = path / "saved_model.pb";

  auto canonOr = canonicalizeTFSavedModelPath(path / "saved_model.pb");
  ASSERT_FALSE(canonOr.isError()) << canonOr.getError();
  EXPECT_EQ(expected, *canonOr);

  canonOr =
      canonicalizeTFSavedModelPath(path / "saved_model.pb" / "saved_model.pb");
  ASSERT_FALSE(canonOr.isError()) << canonOr.getError();
  EXPECT_EQ(expected, *canonOr);
}

TEST(TFSavedModelCanonicalPath, nonExistent) {
  TemporaryDirectory temp;
  ASSERT_TRUE(temp);
  auto path = temp.path();
  auto canonOr = canonicalizeTFSavedModelPath(path / "mymodel");
  ASSERT_TRUE(canonOr.isError()) << *canonOr;
  EXPECT_THAT(canonOr.getError(), testing::EndsWith(" does not exist"));
}

TEST(TFSavedModelCanonicalPath, directoryWithoutPb) {
  TemporaryDirectory temp;
  ASSERT_TRUE(temp);
  auto path = temp.path();
  ASSERT_TRUE(std::filesystem::create_directory(path / "mymodel"));
  auto canonOr = canonicalizeTFSavedModelPath(path / "mymodel");
  ASSERT_TRUE(canonOr.isError()) << *canonOr;
  EXPECT_THAT(canonOr.getError(),
              testing::EndsWith(
                  " does not contain the required file 'saved_model.pb'"));
}

TEST(TFSavedModelCanonicalPath, notDirectory) {
  TemporaryDirectory temp;
  ASSERT_TRUE(temp);
  auto path = temp.path();
  ASSERT_TRUE(touchFile(path / "mymodel"));
  auto canonOr = canonicalizeTFSavedModelPath(path / "mymodel");
  ASSERT_TRUE(canonOr.isError()) << *canonOr;
  EXPECT_THAT(canonOr.getError(),
              testing::EndsWith(
                  " is not a directory and cannot be loaded as a SavedModel"));
}
