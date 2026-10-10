//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Assembly-format pieces shared by the MAP and MOGG dialects.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_ASMSYNTAX_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_ASMSYNTAX_H

#include "mlir/IR/OpImplementation.h"

namespace M::MOGG {

/// Parses `<keyword> {%inner = %outer : type, ...}`. Each `%outer` is an
/// operand appended to `outers`, and the region sees it as the block argument
/// `%inner`, appended to `entryArgs`. A binding without `: type` gets
/// `impliedType`. Without `keyword` nothing is parsed, unless `required` is
/// set.
mlir::ParseResult parseBindings(
    mlir::OpAsmParser &parser, llvm::StringRef keyword,
    llvm::SmallVectorImpl<mlir::OpAsmParser::Argument> &entryArgs,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &outers,
    llvm::SmallVectorImpl<mlir::Type> &outerTypes, mlir::Type impliedType,
    bool required);

/// Prints `<keyword> {%inner = %outer : type, ...}`, pairing each operand
/// with the block argument that stands for it in the region. Types equal to
/// `impliedType` are not printed. Prints nothing for no operands unless
/// `required` is set.
void printBindings(mlir::OpAsmPrinter &p, llvm::StringRef keyword,
                   mlir::ValueRange outers,
                   mlir::ArrayRef<mlir::BlockArgument> inners,
                   mlir::Type impliedType, bool required);

/// Parses `(%a : type, ...)`, the region arguments that have no operand.
mlir::ParseResult
parseLeadingArgs(mlir::OpAsmParser &parser,
                 llvm::SmallVectorImpl<mlir::OpAsmParser::Argument> &args);

/// Prints `(%a : type, ...)`.
void printLeadingArgs(mlir::OpAsmPrinter &p,
                      mlir::ArrayRef<mlir::BlockArgument> args);

/// Parses the `key = value, ...` body of `@callee[...]` into a dictionary.
mlir::ParseResult parseKernelParamsBody(mlir::OpAsmParser &parser,
                                        mlir::DictionaryAttr &params);

/// Prints the body of `@callee[...]`.
void printKernelParamsBody(mlir::OpAsmPrinter &p, mlir::Operation *op,
                           mlir::DictionaryAttr params);

/// Parses an optional `[key = value, ...]`. Without the brackets `params`
/// stays null, which consumers treat like an empty dictionary.
mlir::ParseResult parseOptionalKernelParams(mlir::OpAsmParser &parser,
                                            mlir::DictionaryAttr &params);

/// Prints `[key = value, ...]`, or nothing for null or empty `params`.
void printOptionalKernelParams(mlir::OpAsmPrinter &p, mlir::Operation *op,
                               mlir::DictionaryAttr params);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_ASMSYNTAX_H
