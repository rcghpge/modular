//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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
#include "GenericML/GraphCompiler/MOGGDialect/MOGGAttrs.cpp.inc"
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

void nodeToString(Attribute node, llvm::raw_string_ostream &os,
                  bool includeLineNumbers) {
  llvm::TypeSwitch<Attribute>(node)
      .Case<FusionLeafAttr>([&](FusionLeafAttr leaf) {
        os << leaf.getOpName().getValue();
        if (includeLineNumbers)
          if (uint32_t line = leaf.getLineNumber())
            os << ':' << line;
      })
      .Case<FusionSequenceAttr, FusionElementwiseAttr, FusionViewAttr,
            FusionAffineAttr, FusionMutableLoadAttr>([&](auto attr) {
        os << decltype(attr)::getFusionName() << '(';
        llvm::interleaveComma(attr.getChildren(), os, [&](Attribute child) {
          nodeToString(child, os, includeLineNumbers);
        });
        os << ')';
      })
      .Case<FusionEpilogueAttr, FusionPrologueAttr>([&](auto attr) {
        os << decltype(attr)::getFusionName() << '(';
        llvm::interleaveComma(attr.getChildren(), os, [&](Attribute child) {
          nodeToString(child, os, includeLineNumbers);
        });
        os << ", operandIndices=[";
        llvm::interleaveComma(attr.getOperandIndices(), os);
        os << "])";
      });
}

/// Pretty-print a fusion info tree node with tree-drawing connectors.
void nodeToPrettyTree(Attribute node, llvm::raw_string_ostream &os,
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
      nodeToPrettyTree(children[i], os, childPrefix, childIsLast);
    }
  };

  llvm::TypeSwitch<Attribute>(node)
      .Case<FusionLeafAttr>([&](FusionLeafAttr leaf) {
        os << leaf.getOpName().getValue();
        if (uint32_t line = leaf.getLineNumber())
          os << ':' << line;
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

std::string FusionInfoAttr::toString(bool includeLineNumbers) const {
  std::string result;
  llvm::raw_string_ostream os(result);
  nodeToString(getRoot(), os, includeLineNumbers);
  return result;
}

std::string FusionInfoAttr::toPrettyTree() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  nodeToPrettyTree(getRoot(), os);
  while (!result.empty() && result.back() == '\n')
    result.pop_back();
  return result;
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

  return FusionLeafAttr::get(ctx, StringAttr::get(ctx, opName), lineNumber);
}

/// Parse a parenthesized, comma-separated list of child nodes. Each child is
/// parsed via parseNode. An optional trailing "operand=<N>" is parsed for
/// epilogue/prologue groups.
ParseResult parseChildren(AsmParser &p, SmallVectorImpl<Attribute> &children,
                          SmallVectorImpl<uint32_t> &operandIndices) {
  if (p.parseLParen())
    return failure();

  if (succeeded(p.parseOptionalRParen()))
    return success();

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
  if (fusionInfo && !isa<FusionLeafAttr>(fusionInfo.getRoot()))
    p << " fusionInfo = " << fusionInfo.toString(/*includeLineNumbers=*/true);
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
#include "GenericML/GraphCompiler/MOGGDialect/MOGGAttrs.cpp.inc"
