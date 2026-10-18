//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_PublicASTDecl_H
#define KGEN_MOJOTOOLING_PublicASTDecl_H

#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/SignatureModel.h"
#include "Mojo/MojoTooling/TypeMetadata.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <string>
namespace llvm {
namespace json {
class Object;
} // namespace json
} // namespace llvm

namespace M::KGEN {
enum class PassingKind : uint32_t;
enum class VariadicKind : uint32_t;
} // namespace M::KGEN

namespace M::KGEN::LIT {

using M::KGEN::TypeMetadata;

class ASTDecl;
class AliasASTDecl;
class MojoParserContext;
class PackageASTDecl;
class SharedState;
class StructASTDecl;
class StructFieldASTDecl;
class TraitASTDecl;
class VarASTDecl;
class FnTypeGeneratorType;

namespace {
/// Typical max trait composition size in Mojo APIs.
/// Currently only 1 case exceeds this: builtin.Comparable with 5 traits.
static constexpr size_t kTypicalTraitCompositionSize = 4;

/// Typical max parameter count for Mojo functions, structs, and aliases.
/// Currently about 98% of declarations use ≤2 parameters.
static constexpr size_t kTypicalParameterCount = 2;

/// Typical max argument count for Mojo functions.
/// Currently about 95% of functions use ≤2 arguments.
static constexpr size_t kTypicalArgumentCount = 2;
} // namespace
class PublicDecl;
enum class PublicDeclKind;

std::unique_ptr<PublicDecl> getPublicDecl(ASTDecl *decl);
std::optional<PublicDeclKind> getApproximatePublicDeclKind(ASTDecl *decl);

/// Rendered `where` conditions of a decl's conditional trait conformances,
/// keyed by the conformed trait's symbol. Unconditional conformances are
/// absent. The key is the symbol rather than the trait's short name because
/// distinct symbols can share one name.
using ConformanceConditionMap =
    llvm::DenseMap<KGEN::TraitSymbolAttr, std::string>;

//===----------------------------------------------------------------------===//
// PublicDecls
//
// The following classes provide simple and JSON-serializable representations of
// the most common Mojo decls. They also include structured documentation from
// the corresponding DocStrings and the ability to summarize themselves as code
// snippets.
//
// The API for these decls is expected to be somewhat stable because other tools
// rely on it, hence its `Public` name.
//===----------------------------------------------------------------------===//

/// Stability information extracted from the @stable decorator on a decl.
/// isStable=false means no @stable decorator. sinceVersion is empty when
/// the decorator was bare (@stable with no arguments).
/// isStabilityTracked=true means the symbol is in a package that has opted
/// into stability tracking (currently only "std").
struct StableInfo {
  bool isStable = false;
  bool isStabilityTracked = false;
  std::string sinceVersion;
};

/// The different kinds of public decls.
enum class PublicDeclKind {
  DK_PublicAliasDecl,
  DK_PublicArgumentDecl,
  DK_PublicFunctionDecl,
  DK_PublicModuleDecl,
  DK_PublicPackageDecl,
  DK_PublicParameterDecl,
  DK_PublicStructDecl,
  DK_PublicStructFieldDecl,
  DK_PublicTraitDecl,
  DK_PublicVariableDecl,
};

/// Base class of all public decls.
class PublicDecl {
public:
  virtual ~PublicDecl() = default;

  /// Generate a correct piece of code that summarizes this decl.
  virtual std::string getDeclarationSnippet(MojoParserContext &ctx) const {
    return {};
  }

  /// Get the name of the decl. It might be empty.
  StringRef getName() const { return name; }

  /// Return the declaration this public view represents.
  ASTDecl &getDecl() const { return *decl; }

  /// Get a string representation of the kind of decl, e.g., 'variable',
  /// 'function', etc.
  StringRef getKindAsString() const;

  static StringRef getKindAsString(PublicDeclKind kind);

  /// Serialize the fields in this decl to JSON.
  virtual llvm::json::Object toJSON(MojoParserContext &ctx) const;

  /// Return a nicely formatted markdown docstring of this declaration. It might
  /// be empty if no docstring is available.
  virtual std::string getMarkdownDocString() const;

  /// Return a nicely formatted markdown blob containing the declaration snippet
  /// and doc string of the decl.
  std::string getFullMarkdownString(MojoParserContext &ctx) const;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  PublicDeclKind getKind() const { return kind; }

protected:
  friend std::unique_ptr<PublicDecl> getPublicDecl(ASTDecl *decl);

  PublicDecl(PublicDeclKind kind, ASTDecl &decl, StringRef name)
      : kind(kind), decl(&decl), name(name) {}

private:
  PublicDeclKind kind;
  ASTDecl *decl;
  StringRef name;
};

/// Decl for `var`.
class PublicVariableDecl : public PublicDecl {
public:
  VarASTDecl &getASTDecl() const;

  /// Get the type string of this variable
  StringRef getType() const { return type; }

  /// Return if this variable is global.
  bool isGlobal() const { return isGlobalVariable; }

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "variable",
  ///    "name": string,
  ///    "type": string,
  ///    "path": string
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicVariableDecl;
  }

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend std::unique_ptr<PublicDecl> getPublicDecl(ASTDecl *decl);

  PublicVariableDecl(ASTDecl *decl);

  std::string type;
  TypeMetadata typeMetadata;
  bool isGlobalVariable;
  StringRef deprecated;
};

/// Decl for parameters of structs or functions.
class PublicParameterDecl : public PublicDecl {
public:
  /// Build a documented parameter from a fully-resolved `ParameterInfo`
  /// (which carries the type string with any conforms_to trait bounds
  /// already merged in). Type metadata for doc cross-linking is computed
  /// once during construction.
  PublicParameterDecl(ASTDecl &decl, KGEN::ParameterInfo info,
                      SharedState &sharedState);

  KGEN::PassingKind getPassingKind() const { return info.passingKind; }

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  /// Set the description of this decl.
  void setDescription(StringRef desc) { description = desc; }

  /// Get constraints of this declaration as string. It might be empty.
  StringRef getConstraints() const { return info.constraints; }

  /// The underlying signature-shape view of this parameter, used by both the
  /// rendering algorithm and JSON serialization.
  const KGEN::ParameterInfo &getInfo() const { return info; }

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "parameter",
  ///    "description": string,
  ///    "name": string,
  ///    "type": string,
  ///    "path": string,
  ///    "passingKind": string,
  ///    "defaultValue": string?
  ///    "constraints": string?
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicParameterDecl;
  }

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  KGEN::ParameterInfo info;
  TypeMetadata typeMetadata;
  SmallVector<TypeMetadata, kTypicalTraitCompositionSize> traitMetadata;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
};

/// Decl for function arguments, including varargs arguments.
class PublicArgumentDecl : public PublicDecl {
public:
  /// The convention this argument is passed with. Aliased to the shared
  /// signature model's `ArgumentConvention` so callers can use either name.
  using Convention = KGEN::ArgumentConvention;

  /// Build a documented argument from a fully-resolved `ArgumentInfo`.
  PublicArgumentDecl(ASTDecl &decl, KGEN::ArgumentInfo info,
                     SharedState &sharedState);

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  Convention getConvention() const { return info.convention; }

  KGEN::PassingKind getPassingKind() const { return info.passingKind; }
  void setPassingKind(KGEN::PassingKind kind) { info.passingKind = kind; }

  /// Set the description of this decl.
  void setDescription(StringRef desc) { description = desc; }

  /// The underlying signature-shape view of this argument, used by both the
  /// rendering algorithm and JSON serialization.
  const KGEN::ArgumentInfo &getInfo() const { return info; }
  KGEN::ArgumentInfo &getInfo() { return info; }

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "argument",
  ///    "name": string,
  ///    "description": string,
  ///    "convention": string, // "imm", "mut", "var"
  ///    "type": string,
  ///    "path": string,
  ///    "passingKind": string,
  ///    "defaultValue": string?
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicArgumentDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  KGEN::ArgumentInfo info;
  TypeMetadata typeMetadata;
  SmallVector<TypeMetadata, kTypicalTraitCompositionSize> traitMetadata;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
};

/// Decl for alias.
class PublicAliasDecl : public PublicDecl {
public:
  AliasASTDecl &getASTDecl() const;

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the declaration snippet for the alias. The positions of parameters
  /// within the printed snippet may be extracted via `parameterOffsets`, which
  /// may be null if parameter offsets are not desired.
  std::string getDeclarationSnippet(
      MojoParserContext &ctx,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) const;

  std::string getMarkdownDocString() const override;

  /// Return the parameters of this alias.
  ArrayRef<PublicParameterDecl> getParameters() const { return parameters; }

  StringRef getValue() const { return value; }

  /// Return the type of this alias if available.
  StringRef getType() const { return type; }

  /// Return if this alias is global.
  bool isGlobal() const { return isGlobalAlias; }

  /// Populate the parameter descriptions from a docstring.
  void augmentWithDocumentation(ArrayRef<StringRef> description);

  /// Generate a string for the signature of this alias, given its
  /// components. The positions of parameters within the printed signature may
  /// be extracted via the optionally null `parameterOffsets`.
  std::string getSignature(MojoParserContext &ctx,
                           SmallVectorImpl<std::pair<unsigned, unsigned>>
                               *parameterOffsets = nullptr) const;

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "alias",
  ///    "name": string,
  ///    "description": string,
  ///    "summary": string,
  ///    "parameters": PublicParameterDecl[],
  ///    "signature": string,
  ///    "value": string,
  ///    "type": string
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicAliasDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend std::unique_ptr<PublicDecl> getPublicDecl(ASTDecl *decl);

  PublicAliasDecl(ASTDecl *decl);

  SmallVector<PublicParameterDecl, kTypicalParameterCount> parameters;
  std::string aliasConstraints;
  std::string value;
  std::string type;
  std::string docPath;
  bool isGlobalAlias;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string description;
  StableInfo stableInfo;
  std::string summary;
};

/// Decl for functions of any kind, including closures, instance methods, fn
/// functions, def functions, etc.
class PublicFunctionDecl : public PublicDecl {
public:
  /// Return true if this is an async function.
  bool isAsync() const { return isAsyncFlag; }

  /// Return true if this is an __init__ function decorated with the @implicit
  /// decorator
  bool isImplicitConversion() const { return isImplicitConversionFlag; }

  /// Return true if this is a static struct method, i.e., marked with @static.
  bool isStatic() const { return isStaticFlag; }

  /// Return true if this is a non-static struct method.
  bool isMethod() const { return isMethodFlag; }

  /// Return true if this trait method has a default implementation.
  bool hasDefaultImplementation() const { return isDefaultImplFlag; }

  /// Return the list of arguments of this function.
  ArrayRef<PublicArgumentDecl> getArguments() const { return args; }
  const PublicArgumentDecl &getArgument(size_t i) const { return args[i]; }

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the declaration snippet for the function. The positions of parameters
  /// and arguments within the printed signature may be extracted via the
  /// optional `parameterOffsets` and `argumentOffsets`. nullptr should be
  /// provided if collecting the offsets isn't desired.
  std::string getDeclarationSnippet(
      MojoParserContext &ctx,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets =
          nullptr) const;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  /// Return the parameters of this function.
  ArrayRef<PublicParameterDecl> getParameters() const { return parameters; }

  // TODO: always return the type, including None. The doc and snippet
  // generation will need to be updated accordingly.
  /// Return the type of this function if it is not None.
  std::optional<StringRef> getReturnType() const { return returnType; }

  /// Generate a string for the signature of this function, given its
  /// components. The positions of parameters, arguments, and the return type
  /// within the printed signature may be extracted via the optionally null
  /// `parameterOffsets`, `argumentOffsets`, and `returnOffset` parameters.
  std::string getSignature(
      MojoParserContext &ctx,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets =
          nullptr,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets = nullptr,
      unsigned *returnOffset = nullptr) const;

  /// Return true if this function raises.
  bool raises() const { return raisesFlag; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "function",
  ///   "name": string,
  ///   "args": PublicArgumentDecl[],
  ///   "constraints": string,
  ///   "description": string,
  ///   "hasDefaultImplementation": boolean,
  ///   "isImplicitConversion", boolean,
  ///   "isStatic": boolean,
  ///   "parameters": PublicParameterDecl[],
  ///   "raises": boolean,
  ///   "raisesDoc": string,
  ///   "returns": {"type": string, "path": string?, "doc": string?},
  ///   "signature": string, // E.g., "baz() -> Int"
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicFunctionDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend std::unique_ptr<PublicDecl> getPublicDecl(ASTDecl *decl);

  PublicFunctionDecl(ASTDecl *decl);
  PublicFunctionDecl(ASTDecl *decl, FnTypeGeneratorType signature);

  /// Initialize the function decl with the given signature.
  void initFromSignature(ASTDecl *decl, FnTypeGeneratorType signature,
                         ArrayRef<Type> userArgTypes, Type userResultType);

  /// Augment this function decl with docstring documentation, as well as its
  /// parameters and args.
  void augmentWithDocumentation(ArrayRef<StringRef> description);

  SmallVector<PublicArgumentDecl, kTypicalArgumentCount> args;
  SmallVector<PublicParameterDecl, kTypicalParameterCount> parameters;
  std::string fnConstraints;
  std::optional<std::string> returnType;

  //===----------------------------------------------------------------------===//
  // Effects and modifiers
  //===----------------------------------------------------------------------===//

  bool isAsyncFlag = false;
  bool isImplicitConversionFlag = false;
  bool isMethodFlag = false;
  bool isStaticFlag = false;
  bool raisesFlag = false;
  StableInfo stableInfo;
  bool isInit = false; // Is init or moveinit or copyinit.
  bool isDefaultImplFlag =
      false; // True if trait method has default implementation.

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string constraints;
  std::string description;
  std::string raisesDoc;
  std::string returnsDoc;
  std::string summary;
};

/// Decl for struct field.
class PublicStructFieldDecl : public PublicDecl {
public:
  StructFieldASTDecl &getASTDecl() const;

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  std::string getMarkdownDocString() const override;

  /// Return the type of this field.
  StringRef getType() const { return type; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "field",
  ///   "name": string,
  ///   "description": string,
  ///   "summary": string,
  ///   "type": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicStructFieldDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend std::unique_ptr<PublicDecl> getPublicDecl(ASTDecl *decl);

  PublicStructFieldDecl(ASTDecl *decl);

  std::string type;
  std::string value;
};

/// A collection of overloaded functions in the same scope.
class FunctionDeclOverloadSet {
public:
  /// Create a list of function overload sets by grouping the given functions by
  /// their name. It's assumed that the input list is sorted by name.
  static SmallVector<FunctionDeclOverloadSet, 2>
  fromSortedFunctions(SmallVector<PublicFunctionDecl, 2> &&functions);

  /// Get the common name of the functions in this overload set.
  StringRef getBaseName() const { return baseName; }

  /// Get the functions in this overload set.
  ArrayRef<PublicFunctionDecl> getFunctions() const { return functions; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "function",
  ///   "name": string,
  ///   "overloads": PublicFunctionDecl[]
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const;

private:
  FunctionDeclOverloadSet(StringRef baseName) : baseName(baseName) {}

  void append(PublicFunctionDecl function) {
    functions.push_back(std::move(function));
  }

  std::string baseName;
  SmallVector<PublicFunctionDecl, 2> functions;
};

// Decl for trait decls.
class PublicTraitDecl : public PublicDecl {
public:
  TraitASTDecl &getASTDecl() const;

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  std::string getMarkdownDocString() const override;

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "trait",
  ///   "name": string,
  ///   "description": string,
  ///   "functions": FunctionDeclOverloadSet[],
  ///   "parentTraits": { "name": string, "path"?: string }[],
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicTraitDecl;
  }

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend std::unique_ptr<PublicDecl> getPublicDecl(ASTDecl *decl);

  PublicTraitDecl(ASTDecl *decl);

  StringRef deprecated;
  StableInfo stableInfo;
};

/// Decl for structs.
class PublicStructDecl : public PublicDecl {
public:
  StructASTDecl &getASTDecl() const;

  std::string getDeclarationSnippet(MojoParserContext &ctx) const override;

  /// Get the declaration snippet for the struct. The positions of parameters
  /// within the printed snippet may be extracted via `parameterOffsets`, which
  /// may be null if parameter offsets are not desired.
  std::string getDeclarationSnippet(
      MojoParserContext &ctx,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) const;

  std::string getMarkdownDocString() const override;

  /// Generate a string for the signature of this function, given its
  /// components. The positions of parameters within the printed signature may
  /// be extracted via the optionally null `parameterOffsets`.
  std::string getSignature(MojoParserContext &ctx,
                           SmallVectorImpl<std::pair<unsigned, unsigned>>
                               *parameterOffsets = nullptr) const;

  /// Return the parameters of this struct.
  ArrayRef<PublicParameterDecl> getParameters() const { return parameters; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "struct",
  ///   "name": string,
  ///   "aliases": PublicAliasDecl[],
  ///   "description": string,
  ///   "functions": FunctionDeclOverloadSet[],
  ///   "parameters": PublicParameterDecl[],
  ///   "parentTraits": {
  ///     "name": string,
  ///     "path"?: string,
  ///     // The `where` condition, present only for a conditional conformance
  ///     // such as `Writable where conforms_to(T, Writable)`.
  ///     "condition"?: string
  ///   }[],
  ///   "fields": PublicStructFieldDecl[],
  ///   "signature": string,
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static PublicDeclKind getKindStatic() {
    return PublicDeclKind::DK_PublicStructDecl;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const PublicDecl *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend std::unique_ptr<PublicDecl> getPublicDecl(ASTDecl *decl);

  PublicStructDecl(ASTDecl *decl);

  /// Augment this struct decl with docstring documentation, as well as its
  /// parameters.
  void augmentWithDocumentation(ArrayRef<StringRef> description);

  SmallVector<PublicParameterDecl, kTypicalParameterCount> parameters;
  std::string structConstraints;

  ConformanceConditionMap conformanceConditions;

  TypeConvention convention;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string constraints;
  std::string description;
  StableInfo stableInfo;
  std::string summary;
};

} // namespace M::KGEN::LIT

namespace M {
using KGEN::LIT::ConformanceConditionMap;
using KGEN::LIT::FunctionDeclOverloadSet;
using KGEN::LIT::PublicAliasDecl;
using KGEN::LIT::PublicArgumentDecl;
using KGEN::LIT::PublicDecl;
using KGEN::LIT::PublicDeclKind;
using KGEN::LIT::PublicFunctionDecl;
using KGEN::LIT::PublicParameterDecl;
using KGEN::LIT::PublicStructDecl;
using KGEN::LIT::PublicStructFieldDecl;
using KGEN::LIT::PublicTraitDecl;
using KGEN::LIT::PublicVariableDecl;
using KGEN::LIT::StableInfo;
} // namespace M

#endif // KGEN_MOJOTOOLING_PublicASTDecl_H
