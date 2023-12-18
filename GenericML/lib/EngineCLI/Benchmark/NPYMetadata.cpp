//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/EngineCLI/Benchmark/NPYMetadata.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/JSON.h"

using llvm::json::Value;
using namespace M;
using namespace M::BenchmarkCLI;

namespace M::BenchmarkCLI::NPYDetail {

enum class TokenType {
  Invalid,
  End,
  None,
  False,
  True,
  Number,
  String,
  Colon,
  Comma,
  OpenBrace,
  CloseBrace,
  OpenParen,
  CloseParen,
  OpenBracket,
  CloseBracket,
};

const char *tokenTypeName(TokenType tt);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, TokenType tt);

struct Token {
  TokenType type;
  StringRef spelling;

  std::string debugString() const;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Token &token);

class Tokenizer final {
public:
  explicit Tokenizer(StringRef input);

  Token next();
  ErrorOrSuccess takeError();

private:
  TokenType nextType();
  TokenType consumeString(char terminator);
  TokenType setError(Error error);

  StringRef input;
  ErrorOrSuccess savedError;
};

class Parser final {
public:
  explicit Parser(Tokenizer &tokenizer);

  ErrorOr<Value> parseDoc();
  ErrorOr<Value> parseItem();

private:
  Token peek();
  void advance();
  Error unexpected(const Token &tok);

  ErrorOr<Value> parseDict();
  ErrorOr<Value> parseSeq(bool tuple);

  Tokenizer *tokenizer;
  std::optional<Token> curToken;
};

} // namespace M::BenchmarkCLI::NPYDetail

using namespace M::BenchmarkCLI::NPYDetail;

const char *NPYDetail::tokenTypeName(TokenType tt) {
  switch (tt) {
  case TokenType::End:
    return "end of input";
  case TokenType::None:
    return "None";
  case TokenType::False:
    return "False";
  case TokenType::True:
    return "True";
  case TokenType::Number:
    return "number";
  case TokenType::String:
    return "string";
  case TokenType::Colon:
    return "colon";
  case TokenType::Comma:
    return "comma";
  case TokenType::OpenBrace:
    return "open brace";
  case TokenType::CloseBrace:
    return "close brace";
  case TokenType::OpenParen:
    return "open paren";
  case TokenType::CloseParen:
    return "close paren";
  case TokenType::OpenBracket:
    return "open bracket";
  case TokenType::CloseBracket:
    return "close bracket";
  default:
    return "invalid";
  }
}
llvm::raw_ostream &NPYDetail::operator<<(llvm::raw_ostream &os, TokenType tt) {
  return os << tokenTypeName(tt);
}

std::string Token::debugString() const {
  return (Twine(tokenTypeName(type)) + ": " + spelling).str();
}
llvm::raw_ostream &NPYDetail::operator<<(llvm::raw_ostream &os,
                                         const Token &token) {
  return os << token.debugString();
}

Tokenizer::Tokenizer(StringRef input) : input(input) {}

TokenType Tokenizer::setError(Error error) {
  savedError = std::move(error);
  return TokenType::Invalid;
}
ErrorOrSuccess Tokenizer::takeError() { return std::move(savedError); }

Token Tokenizer::next() {
  if (savedError.isError())
    return {TokenType::Invalid, StringRef()};
  input = input.ltrim();
  StringRef span = input;
  TokenType type = nextType();
  span = span.drop_back(input.size());
  return {type, span};
}

TokenType Tokenizer::nextType() {
  if (input.empty())
    return TokenType::End;
  if (input.consume_front(":"))
    return TokenType::Colon;
  if (input.consume_front(","))
    return TokenType::Comma;
  if (input.consume_front("{"))
    return TokenType::OpenBrace;
  if (input.consume_front("}"))
    return TokenType::CloseBrace;
  if (input.consume_front("("))
    return TokenType::OpenParen;
  if (input.consume_front(")"))
    return TokenType::CloseParen;
  if (input.consume_front("["))
    return TokenType::OpenBracket;
  if (input.consume_front("]"))
    return TokenType::CloseBracket;
  if (input.consume_front("None"))
    return TokenType::None;
  if (input.consume_front("False"))
    return TokenType::False;
  if (input.consume_front("True"))
    return TokenType::True;
  size_t numberLen = input.find_first_not_of("-0123456789");
  if (numberLen > 0) {
    input = input.drop_front(std::min(numberLen, input.size()));
    return TokenType::Number;
  }
  if (input.consume_front("'"))
    return consumeString('\'');
  if (input.consume_front("\""))
    return consumeString('"');
  return setError("Unexpected letter '" + Twine(input[0]) + "'");
}

TokenType Tokenizer::consumeString(char terminator) {
  size_t end = input.find(terminator);
  if (end == StringRef::npos)
    return setError("Unterminated string");
  StringRef content = input.take_front(end);
  input = input.drop_front(end + 1);
  if (content.find('\\') != StringRef::npos)
    return setError("Escapes not implemented");
  if (content.find('\n') != StringRef::npos)
    return setError("Newlines not allowed in string");
  return TokenType::String;
}

Parser::Parser(Tokenizer &tokenizer) : tokenizer(&tokenizer) {}

Token Parser::peek() {
  if (!curToken.has_value())
    curToken = tokenizer->next();
  return *curToken;
}
void Parser::advance() { curToken = std::nullopt; }

Error Parser::unexpected(const Token &tok) {
  return Error(Twine("Unexpected token ") + tok.debugString());
}

ErrorOr<Value> Parser::parseDoc() {
  auto itemOr = parseItem();
  if (itemOr.isError())
    return itemOr;
  Token tok = peek();
  if (tok.type != TokenType::End)
    return Error("Junk at end");
  return itemOr;
}

ErrorOr<Value> Parser::parseItem() {
  Token tok = peek();
  advance();
  switch (tok.type) {
  case TokenType::None:
    return Value(nullptr);
  case TokenType::False:
    return Value(false);
  case TokenType::True:
    return Value(true);
  case TokenType::Number: {
    int value;
    if (tok.spelling.getAsInteger(10, value))
      return Error("Invalid number '" + Twine(tok.spelling) + "'");
    return Value(value);
  }
  case TokenType::String:
    // Safe to use spelling because we do not support escapes.
    return Value(tok.spelling.substr(1, tok.spelling.size() - 2));
  case TokenType::OpenBrace:
    return parseDict();
  case TokenType::OpenParen:
    return parseSeq(true);
  case TokenType::OpenBracket:
    return parseSeq(false);
  default:
    return unexpected(tok);
  }
}

ErrorOr<Value> Parser::parseDict() {
  llvm::json::Object object;
  bool mustEnd = false;
  for (;;) {
    Token tok = peek();
    if (tok.type == TokenType::CloseBrace)
      break;
    if (mustEnd)
      return unexpected(tok);
    ErrorOr<Value> keyOr = parseItem();
    if (keyOr.isError())
      return keyOr;
    if (keyOr->kind() != Value::String)
      return Error("Dict keys must be strings");
    tok = peek();
    if (tok.type != TokenType::Colon)
      return unexpected(tok);
    advance();
    ErrorOr<Value> valueOr = parseItem();
    if (valueOr.isError())
      return valueOr;
    auto [it, success] =
        object.try_emplace(*keyOr->getAsString(), valueOr.takeValue());
    if (!success)
      return Error("Duplicate key in dict");
    mustEnd = true;
    tok = peek();
    if (tok.type == TokenType::Comma) {
      mustEnd = false;
      advance();
    }
  }
  advance();
  return object;
}

ErrorOr<Value> Parser::parseSeq(bool tuple) {
  TokenType terminator =
      tuple ? TokenType::CloseParen : TokenType::CloseBracket;
  llvm::json::Array array;
  bool mustEnd = false;
  for (;;) {
    Token tok = peek();
    if (tok.type == terminator)
      break;
    if (mustEnd)
      return unexpected(tok);
    ErrorOr<Value> valueOr = parseItem();
    if (valueOr.isError())
      return valueOr;
    array.push_back(valueOr.takeValue());
    mustEnd = true;
    tok = peek();
    if (tok.type == TokenType::Comma) {
      mustEnd = false;
      advance();
    }
  }
  advance();
  if (tuple && array.size() == 1 && mustEnd)
    return Error("Singleton tuple must end with a comma");
  return array;
}

ErrorOr<Value> M::BenchmarkCLI::parseNPYMetadata(StringRef input) {
  Tokenizer tokenizer(input);
  Parser parser(tokenizer);
  auto result = parser.parseDoc();
  auto tokResult = tokenizer.takeError();
  // Tokenization errors take priority over parser errors.
  if (tokResult.isError())
    return tokResult.takeError();
  return result;
}
