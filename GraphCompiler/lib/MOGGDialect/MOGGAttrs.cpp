//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace M;
using namespace M::MOGG;

//===----------------------------------------------------------------------===//
// MOGGDialect attribute registration
//===----------------------------------------------------------------------===//

void MOGGDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/MOGGDialect/MOGGAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// FusionInfoAttr verification
//===----------------------------------------------------------------------===//

LogicalResult
FusionInfoAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute root) {
  if (!isa<FusionLeafAttr, FusionSequenceAttr, FusionElementwiseAttr,
           FusionEpilogueAttr, FusionPrologueAttr, FusionViewAttr,
           FusionAffineAttr, FusionMutableLoadAttr>(root))
    return emitError() << "root must be a valid fusion info attribute";
  return success();
}

//===----------------------------------------------------------------------===//
// FusionInfoAttr::toString
//===----------------------------------------------------------------------===//

namespace {

// Strip the dialect prefix (e.g., "mo.add" -> "add") from an operation name
// to make trace descriptions more readable.
std::string stripDialectPrefix(StringRef opName) {
  auto dotPos = opName.find('.');
  if (dotPos != StringRef::npos)
    return opName.substr(dotPos + 1).str();
  return opName.str();
}

void nodeToString(Attribute node, llvm::raw_string_ostream &os,
                  bool includeLineNumbers, bool quote,
                  bool stripDialectPrefixes) {
  llvm::TypeSwitch<Attribute>(node)
      .Case<FusionLeafAttr>([&](FusionLeafAttr leaf) {
        if (quote)
          os << '"';
        std::string name = stripDialectPrefixes
                               ? stripDialectPrefix(leaf.getOpName().getValue())
                               : leaf.getOpName().getValue().str();
        llvm::printEscapedString(name, os);
        if (quote)
          os << '"';
        if (includeLineNumbers)
          if (auto fileLoc = dyn_cast_or_null<FileLineColLoc>(leaf.getLoc()))
            if (fileLoc.getLine() > 0)
              os << ':' << fileLoc.getLine();
      })
      .Case<FusionSequenceAttr, FusionElementwiseAttr, FusionViewAttr,
            FusionAffineAttr, FusionMutableLoadAttr>([&](auto attr) {
        os << decltype(attr)::getFusionName() << '(';
        llvm::interleaveComma(attr.getChildren(), os, [&](Attribute child) {
          nodeToString(child, os, includeLineNumbers, quote,
                       stripDialectPrefixes);
        });
        os << ')';
      })
      .Case<FusionEpilogueAttr, FusionPrologueAttr>([&](auto attr) {
        os << decltype(attr)::getFusionName() << '(';
        llvm::interleaveComma(attr.getChildren(), os, [&](Attribute child) {
          nodeToString(child, os, includeLineNumbers, quote,
                       stripDialectPrefixes);
        });
        os << ", operandIndices=[";
        llvm::interleaveComma(attr.getOperandIndices(), os);
        os << "])";
      });
}

/// Pretty-print a fusion info tree node with tree-drawing connectors.
/// If \p highlightLeaves is non-empty, leaf nodes whose op name appears in
/// the set are annotated with " <-- error" in the output.
void nodeToPrettyTree(Attribute node, llvm::raw_string_ostream &os,
                      const llvm::DenseSet<Attribute> &highlightLeaves,
                      StringRef prefix = "", bool isLast = true) {
  auto emitChildren = [&](auto attr, ArrayRef<Attribute> children,
                          ArrayRef<uint32_t> operandIndices = {}) {
    os << decltype(attr)::getFusionName();
    if (!operandIndices.empty()) {
      os << " (operandIndices=[";
      llvm::interleaveComma(operandIndices, os);
      os << "])";
    }
    os << '\n';
    StringRef connector = isLast ? "    " : "│   ";
    std::string childPrefix = (prefix + connector).str();
    for (size_t i = 0; i < children.size(); ++i) {
      bool childIsLast = (i == children.size() - 1);
      os << childPrefix << (childIsLast ? "└── " : "├── ");
      nodeToPrettyTree(children[i], os, highlightLeaves, childPrefix,
                       childIsLast);
    }
  };

  llvm::TypeSwitch<Attribute>(node)
      .Case<FusionLeafAttr>([&](FusionLeafAttr leaf) {
        os << leaf.getOpName().getValue();
        if (auto fileLoc = dyn_cast_or_null<FileLineColLoc>(leaf.getLoc())) {
          StringRef fn = fileLoc.getFilename().getValue();
          uint32_t line = fileLoc.getLine();
          if (!fn.empty() && line)
            os << " (" << fn << ':' << line << ')';
          else if (line)
            os << ':' << line;
        }
        if (auto sig = leaf.getSignature()) {
          if (auto funcTy = dyn_cast<FunctionType>(sig.getValue())) {
            os << " : (";
            llvm::interleaveComma(funcTy.getInputs(), os);
            os << ") -> ";
            if (funcTy.getResults().size() == 1)
              os << funcTy.getResults()[0];
            else {
              os << '(';
              llvm::interleaveComma(funcTy.getResults(), os);
              os << ')';
            }
          }
        }
        if (highlightLeaves.count(leaf))
          os << "  <-- error";
        os << '\n';
      })
      .Case<FusionSequenceAttr, FusionElementwiseAttr, FusionViewAttr,
            FusionAffineAttr, FusionMutableLoadAttr>(
          [&](auto attr) { emitChildren(attr, attr.getChildren()); })
      .Case<FusionEpilogueAttr, FusionPrologueAttr>([&](auto attr) {
        emitChildren(attr, attr.getChildren(), attr.getOperandIndices());
      });
}

} // namespace

std::string FusionInfoAttr::toString(bool includeLineNumbers, bool quote,
                                     bool stripDialectPrefixes) const {
  std::string result;
  llvm::raw_string_ostream os(result);
  nodeToString(getRoot(), os, includeLineNumbers, quote, stripDialectPrefixes);
  return result;
}

std::string
FusionInfoAttr::toPrettyTree(const llvm::DenseSet<Attribute> &highlightLeaves,
                             bool indent) const {
  std::string result;
  llvm::raw_string_ostream os(result);
  nodeToPrettyTree(getRoot(), os, highlightLeaves);
  while (!result.empty() && result.back() == '\n')
    result.pop_back();
  if (!indent)
    return result;
  std::string indented;
  llvm::raw_string_ostream ios(indented);
  for (auto line : llvm::split(result, '\n'))
    ios << "  " << line << '\n';
  return indented;
}

//===----------------------------------------------------------------------===//
// FusionInfoAttr::parseTree
//===----------------------------------------------------------------------===//

namespace {

ParseResult parseChildren(AsmParser &p, SmallVectorImpl<Attribute> &children,
                          SmallVectorImpl<uint32_t> &operandIndices);

/// Recursively parse a fusion info tree node. Group nodes (e.g. "Sequence",
/// "Epilogue") are parsed by matching the keyword, then calling parseChildren
/// which in turn calls parseNode for each child.
Attribute parseNode(AsmParser &p) {
  MLIRContext *ctx = p.getContext();

  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(
          &keyword,
          {FusionSequenceAttr::getFusionName(),
           FusionElementwiseAttr::getFusionName(),
           FusionEpilogueAttr::getFusionName(),
           FusionPrologueAttr::getFusionName(), FusionViewAttr::getFusionName(),
           FusionAffineAttr::getFusionName(),
           FusionMutableLoadAttr::getFusionName()}))) {
    SmallVector<Attribute> children;
    SmallVector<uint32_t> operandIndices;

    if (parseChildren(p, children, operandIndices))
      return {};

    if (keyword == FusionSequenceAttr::getFusionName())
      return FusionSequenceAttr::get(ctx, children);
    if (keyword == FusionElementwiseAttr::getFusionName())
      return FusionElementwiseAttr::get(ctx, children);
    if (keyword == FusionEpilogueAttr::getFusionName())
      return FusionEpilogueAttr::get(ctx, children, operandIndices);
    if (keyword == FusionPrologueAttr::getFusionName())
      return FusionPrologueAttr::get(ctx, children, operandIndices);
    if (keyword == FusionViewAttr::getFusionName())
      return FusionViewAttr::get(ctx, children);
    if (keyword == FusionAffineAttr::getFusionName())
      return FusionAffineAttr::get(ctx, children);
    if (keyword == FusionMutableLoadAttr::getFusionName())
      return FusionMutableLoadAttr::get(ctx, children);

    llvm_unreachable("unhandled fusion keyword");
  }

  // Leaf: parse op name with optional line number.
  std::string opName;
  if (p.parseOptionalKeywordOrString(&opName))
    return {};

  uint32_t lineNumber = 0;
  if (succeeded(p.parseOptionalColon())) {
    if (p.parseInteger(lineNumber))
      return {};
  }

  // NOTE: The custom textual format does not encode location or signature.
  // If this parser is ever used to round-trip through textual IR, those
  // fields will be lost. Currently only used in lit tests.
  LocationAttr loc;
  if (lineNumber)
    loc = FileLineColLoc::get(ctx, "", lineNumber, 0);
  return FusionLeafAttr::get(ctx, StringAttr::get(ctx, opName), loc,
                             /*signature=*/{});
}

/// Parse a parenthesized, comma-separated list of child nodes. Each child is
/// parsed via parseNode. An optional trailing "operand=<N>" is parsed for
/// epilogue/prologue groups.
ParseResult parseChildren(AsmParser &p, SmallVectorImpl<Attribute> &children,
                          SmallVectorImpl<uint32_t> &operandIndices) {
  if (p.parseLParen())
    return failure();

  if (succeeded(p.parseOptionalRParen()))
    return mlir::success();

  do {
    if (succeeded(p.parseOptionalKeyword("operandIndices"))) {
      if (p.parseEqual() || p.parseLSquare())
        return failure();
      do {
        uint32_t idx;
        if (p.parseInteger(idx))
          return failure();
        operandIndices.push_back(idx);
      } while (succeeded(p.parseOptionalComma()));
      if (p.parseRSquare())
        return failure();
      break;
    }

    Attribute child = parseNode(p);
    if (!child)
      return failure();
    children.push_back(child);
  } while (succeeded(p.parseOptionalComma()));

  return p.parseRParen();
}

} // namespace

//===----------------------------------------------------------------------===//
// FusionInfo custom printing/parsing for ops
//===----------------------------------------------------------------------===//

void M::MOGG::printFusionInfo(AsmPrinter &p, Operation *op,
                              FusionInfoAttr fusionInfo) {
  if (fusionInfo)
    p << " fusionInfo = "
      << fusionInfo.toString(/*includeLineNumbers=*/true, /*quote*/ true);
}

ParseResult M::MOGG::parseFusionInfo(AsmParser &p, FusionInfoAttr &fusionInfo) {
  if (succeeded(p.parseOptionalKeyword("fusionInfo"))) {
    if (p.parseEqual())
      return failure();
    Attribute root = parseNode(p);
    if (!root)
      return failure();
    fusionInfo = FusionInfoAttr::get(p.getContext(), root);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen-generated code
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MOGGDialect/MOGGAttrs.cpp.inc"
