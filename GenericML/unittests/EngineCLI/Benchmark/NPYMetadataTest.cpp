//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/EngineCLI/Benchmark/NPYMetadata.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/JSON.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::BenchmarkCLI;

template <typename T>
static ErrorOr<T> ok(T value) {
  return std::move(value);
}

static ErrorOr<llvm::json::Value> jsonErr(Twine errorMessage) {
  return Error(errorMessage);
}

static llvm::json::Value mustParseJSON(StringRef input) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(input);
  if (!parsed) {
    ADD_FAILURE() << "Failed to parse JSON value '" << input
                  << "': " << toString(parsed.takeError());
    return llvm::json::Value(nullptr);
  }
  return std::move(*parsed);
}

namespace M {
static llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const ErrorOr<llvm::json::Value> &valueOr) {
  if (valueOr.isError())
    return os << "error: " << valueOr.getError();
  return os << *valueOr;
}
} // namespace M

TEST(parseNPYMetadata, atoms) {
  EXPECT_EQ(ok(mustParseJSON("null")), parseNPYMetadata("None"));
  EXPECT_EQ(ok(mustParseJSON("false")), parseNPYMetadata("False"));
  EXPECT_EQ(ok(mustParseJSON("true")), parseNPYMetadata("True"));
}

TEST(parseNPYMetadata, numbers) {
  EXPECT_EQ(ok(mustParseJSON("1")), parseNPYMetadata("1"));
  EXPECT_EQ(ok(mustParseJSON("50")), parseNPYMetadata("50"));
  EXPECT_EQ(ok(mustParseJSON("-1")), parseNPYMetadata("-1"));
}

TEST(parseNPYMetadata, strings) {
  EXPECT_EQ(ok(mustParseJSON("\"\"")), parseNPYMetadata("''"));
  EXPECT_EQ(ok(mustParseJSON("\"abc\"")), parseNPYMetadata("'abc'"));
  EXPECT_EQ(jsonErr("Unterminated string"), parseNPYMetadata("'a"));
  EXPECT_EQ(jsonErr("Escapes not implemented"), parseNPYMetadata("'\\n'"));
  EXPECT_EQ(jsonErr("Newlines not allowed in string"),
            parseNPYMetadata("'\n'"));
  EXPECT_EQ(ok(mustParseJSON("\"\"")), parseNPYMetadata("\"\""));
  EXPECT_EQ(ok(mustParseJSON("\"abc\"")), parseNPYMetadata("\"abc\""));
  EXPECT_EQ(jsonErr("Unterminated string"), parseNPYMetadata("\"a"));
  EXPECT_EQ(jsonErr("Escapes not implemented"), parseNPYMetadata("\"\\n\""));
  EXPECT_EQ(jsonErr("Newlines not allowed in string"),
            parseNPYMetadata("\"\n\""));
}

TEST(parseNPYMetadata, dicts) {
  EXPECT_EQ(ok(mustParseJSON("{}")), parseNPYMetadata("{}"));
  EXPECT_EQ(jsonErr("Unexpected token comma: ,"), parseNPYMetadata("{,}"));
  EXPECT_EQ(ok(mustParseJSON("{\"a\": \"b\"}")),
            parseNPYMetadata("{'a': 'b'}"));
  EXPECT_EQ(ok(mustParseJSON("{\"a\": \"b\"}")),
            parseNPYMetadata("{'a': 'b',}"));
  EXPECT_EQ(ok(mustParseJSON("{\"a\": \"b\", \"c\": \"d\"}")),
            parseNPYMetadata("{'a': 'b', 'c': 'd'}"));
  EXPECT_EQ(jsonErr("Unexpected token comma: ,"),
            parseNPYMetadata("{'a': 'b',,}"));
  EXPECT_EQ(jsonErr("Unexpected token end of input: "), parseNPYMetadata("{"));
  EXPECT_EQ(jsonErr("Dict keys must be strings"), parseNPYMetadata("{1: 2}"));
  EXPECT_EQ(jsonErr("Duplicate key in dict"),
            parseNPYMetadata("{'a': 'b', 'a': 'c'}"));
}

TEST(parseNPYMetadata, lists) {
  EXPECT_EQ(ok(mustParseJSON("[]")), parseNPYMetadata("[]"));
  EXPECT_EQ(jsonErr("Unexpected token comma: ,"), parseNPYMetadata("[,]"));
  EXPECT_EQ(ok(mustParseJSON("[1]")), parseNPYMetadata("[1]"));
  EXPECT_EQ(ok(mustParseJSON("[1]")), parseNPYMetadata("[1,]"));
  EXPECT_EQ(jsonErr("Unexpected token comma: ,"), parseNPYMetadata("[1,,]"));
  EXPECT_EQ(ok(mustParseJSON("[1, 2]")), parseNPYMetadata("[1, 2]"));
  EXPECT_EQ(ok(mustParseJSON("[1, 2]")), parseNPYMetadata("[1, 2,]"));
  EXPECT_EQ(jsonErr("Unexpected token comma: ,"), parseNPYMetadata("[1, 2,,]"));
  EXPECT_EQ(jsonErr("Unexpected token end of input: "), parseNPYMetadata("["));
}

TEST(parseNPYMetadata, tuples) {
  EXPECT_EQ(ok(mustParseJSON("[]")), parseNPYMetadata("()"));
  EXPECT_EQ(jsonErr("Unexpected token comma: ,"), parseNPYMetadata("(,)"));
  EXPECT_EQ(jsonErr("Singleton tuple must end with a comma"),
            parseNPYMetadata("(1)"));
  EXPECT_EQ(ok(mustParseJSON("[1]")), parseNPYMetadata("(1,)"));
  EXPECT_EQ(jsonErr("Unexpected token comma: ,"), parseNPYMetadata("(1,,)"));
  EXPECT_EQ(ok(mustParseJSON("[1, 2]")), parseNPYMetadata("(1, 2)"));
  EXPECT_EQ(ok(mustParseJSON("[1, 2]")), parseNPYMetadata("(1, 2,)"));
  EXPECT_EQ(jsonErr("Unexpected token comma: ,"), parseNPYMetadata("(1, 2,,)"));
  EXPECT_EQ(jsonErr("Unexpected token end of input: "), parseNPYMetadata("("));
}

TEST(parseNPYMetadata, whitespace) {
  EXPECT_EQ(ok(mustParseJSON("null")), parseNPYMetadata("\n\t None \n "));
}

TEST(parseNPYMetadata, miscErrors) {
  EXPECT_EQ(jsonErr("Unexpected token end of input: "), parseNPYMetadata(""));
  EXPECT_EQ(jsonErr("Unexpected token end of input: "),
            parseNPYMetadata(" \n\t"));
  EXPECT_EQ(jsonErr("Unexpected letter '.'"), parseNPYMetadata("1.0"));
  EXPECT_EQ(jsonErr("Unexpected letter 'e'"), parseNPYMetadata("1e10"));
  EXPECT_EQ(jsonErr("Junk at end"), parseNPYMetadata("1 2"));
}
