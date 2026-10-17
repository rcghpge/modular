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
//
// This file provides the main entrypoints for the Mojo parser.
//
//===----------------------------------------------------------------------===//

#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// MojoASTDeclRef
//===----------------------------------------------------------------------===//

/// Get the SharedState for this decl if non-null.
KGEN::LIT::SharedState *MojoASTDeclRef::getShared() const {
  return decl ? &decl->getShared() : nullptr;
}

/// Return the signature type contained by this decl (e.g. if it's a function),
/// or null otherwise.
static FnTypeGeneratorType getSignatureFromDecl(ASTDecl *decl) {
  if (!decl)
    return nullptr;
  if (auto defDecl = dyn_cast<DefASTDecl>(decl))
    if (auto func = defDecl->getOp())
      return func.getFuncTypeGenerator();
  if (auto typeValue = decl->getIfTypeValue())
    return dyn_cast_or_null<FnTypeGeneratorType>(typeValue);
  return nullptr;
}

/// Return the index of the argument that corresponds to the given decl.
static std::optional<size_t> getDeclArgIndex(ASTDecl &decl, BlockArgument arg) {
  // If this is a normal argument, we can just return the argument number.
  if (arg.getParentRegion())
    return arg.getArgNumber();
  // Otherwise, we need to inspect the children of the parent decl. The parser
  // uses a shared block for all dangling arguments, so we need to find the
  // correct one manually.
  size_t argIndex = 0;
  for (auto [name, decls] : decl.getParentDecl()->getDeclsInScope()) {
    if (decls.size() != 1)
      continue;

    // Check if the decl is the one we're looking for.
    if (auto cv = decls.front()->getIfIRValue()) {
      // Ignore parameters for our indexing.
      if (decls.front()->getIfIRValue().getIfPValue())
        continue;

      if (cv.getMlirValue() == arg)
        return argIndex;
      ++argIndex;
    }
  }
  return std::nullopt;
}

/// If this decl corresponds to a not owned function argument, return its
/// corresponding BlockArgument. Otherwise, return null.
static BlockArgument getIfNotOwnedFunctionArgument(ASTDecl &decl) {
  Value val = decl.getIfIRValue().getMlirValue();
  if (!val)
    return {};

  // Look through rebinds of arguments, which may happen for certain
  // argument conventions.
  if (auto rebind = val.getDefiningOp<RebindOp>())
    val = rebind.getInput();

  // Check if this is a block argument of a function.
  if (auto bbArg = dyn_cast<BlockArgument>(val)) {
    if (isa_and_nonnull<FnOp>(bbArg.getOwner()->getParentOp()))
      return bbArg;
    // If this is a block without a proper owner, this is generally a
    // block argument for a function signature. These are detached from
    // normal IR.
    if (!bbArg.getOwner()->getParentOp())
      return bbArg;
  }

  return {};
}

static ParamDeclRefAttr getIfParameter(ASTDecl &decl) {
  if (auto val = decl.getIfIRValue().getIfPValue()) {
    if (auto paramRef = dyn_cast<ParamDeclRefAttr>(val.get()))
      return paramRef;
  }
  return {};
}

MojoASTTypeRef MojoASTDeclRef::getType() const {
  return decl->getToolingType();
}

std::optional<StringRef> MojoASTDeclRef::getName() const {
  return decl->getName();
}

std::optional<StringRef> MojoASTDeclRef::getDeprecationWarning() const {
  return decl->getDeprecationWarning();
}

llvm::SMLoc MojoASTDeclRef::getLoc() const { return decl->getLoc(); }

MojoASTDeclRef MojoASTDeclRef::getParent() const {
  return MojoASTDeclRef(decl->getParentDecl());
}

/// Create an Argument decl view for the given decl and argument index.
static std::unique_ptr<PublicArgumentDecl>
createPublicArgumentDecl(MojoASTDeclRef declRef, unsigned arg) {
  // The parent PublicFunctionDecl is the one who owns the docstring of this
  // argument, so it's easier just to construct that view and extract the
  // argument from it.
  MojoASTDeclRef parentDecl = declRef->getParentDecl();
  auto functionView =
      llvm::unique_dyn_cast_or_null<PublicFunctionDecl>(parentDecl.getDecl());
  if (!functionView || functionView->getArguments().size() <= arg)
    return nullptr;
  return std::make_unique<PublicArgumentDecl>(functionView->getArgument(arg));
}

/// Helper method for `getDeclImpl` that either returns a PublicDeclKind
/// or a new PublicDecl instance depending on the ResultType.
template <typename ResultType, typename PublicDeclT, typename... DeclArgs>
ResultType MojoASTDeclRef::createPublicDecl(DeclArgs &&...declArgs) const {
  if constexpr (std::is_same_v<ResultType, ApproximatePublicDeclKind>)
    return PublicDeclT::getKindStatic();
  else
    return std::unique_ptr<PublicDeclT>(
        new PublicDeclT(std::forward<DeclArgs>(declArgs)...));
}

/// Common implementation for `getDecl` and `getApproximateDeclKind`.
///
/// If parametrized with `PublicDeclInstance`, it will return a PublicDecl,
/// which can be an expensive operation for entities like function arguments,
/// but it is guaranteed to be correct. This is considered the correct but slow
/// path.
///
/// If parametrized with `ApproximatePublicDeclKind`, it will return a
/// `PublicDeclKind` by doing only cheap lookups. In general, expensive or
/// unbounded iterations are disallowed in this variant and only executed in
/// the `PublicDeclInstance` case. This is considered the approximate but fast
/// path, which is used by interactive tools like the LSP.
template <typename ResultType>
ResultType MojoASTDeclRef::getDeclImpl() const {
  static_assert(std::is_same_v<ResultType, ApproximatePublicDeclKind> ||
                    std::is_same_v<ResultType, PublicDeclInstance>,
                "Only ApproximatePublicDeclKind or PublicDeclInstance are "
                "valid parameters.");

  constexpr bool isApproximateResult =
      std::is_same_v<ResultType, ApproximatePublicDeclKind>;

  if (Operation *declOp = decl->getOperation()) {
    if (isa<AliasDeclOp>(declOp))
      return createPublicDecl<ResultType, PublicAliasDecl>(decl);

    if (isa<FnOp>(declOp))
      return createPublicDecl<ResultType, PublicFunctionDecl>(decl);

    // An import op names either a module or package; they ultimately resolve to
    // the same thing but classify it as a module so tooling sees the imported
    // name as a module rather than a variable.
    if (isa<FileModuleOp, ImportOp>(declOp))
      return createPublicDecl<ResultType, PublicModuleDecl>(decl);

    if (isa<StructDeclOp>(declOp))
      return createPublicDecl<ResultType, PublicStructDecl>(decl);

    if (isa<StructFieldOp>(declOp))
      return createPublicDecl<ResultType, PublicStructFieldDecl>(decl);

    if (auto varDecl = dyn_cast<VarDeclOp>(declOp)) {
      // Handle the case of an argument materialized in a variable.
      if (varDecl.getKind() == VarDeclKind::Arg) {
        if constexpr (isApproximateResult) {
          return PublicDeclKind::DK_PublicArgumentDecl;
        } else {
          auto parentFn = varDecl->getParentOfType<FnOp>();
          for (auto [idx, pogAttr] : llvm::enumerate(
                   parentFn.getFuncTypeGenerator().getArgListAttrs().getPogs()))
            if (pogAttr.getName() == varDecl.getNameAttr())
              return createPublicArgumentDecl(*this, idx);
        }
      }
      // Otherwise, this is a regular variable.
      return createPublicDecl<ResultType, PublicVariableDecl>(decl);
    }

    if (isa<PackageOp>(declOp))
      return createPublicDecl<ResultType, PublicPackageDecl>(decl);

    if (isa<TraitDeclOp>(declOp))
      return createPublicDecl<ResultType, PublicTraitDecl>(decl);
  }

  // If the decl corresponds to a signature, synthesize a function view for
  // it.
  if (auto signature = getSignatureFromDecl(decl))
    return createPublicDecl<ResultType, PublicFunctionDecl>(decl, signature);

  // After failing to match with regular Ops, we then inspect the IR to
  // identify if this decl is an argument.
  if (BlockArgument bbArg = getIfNotOwnedFunctionArgument(*decl)) {
    if constexpr (isApproximateResult) {
      return PublicDeclKind::DK_PublicArgumentDecl;
    } else {
      if (std::optional<size_t> argIdx = getDeclArgIndex(*decl, bbArg))
        return createPublicArgumentDecl(*this, *argIdx);
      return nullptr;
    }
  }

  // Now we inspect the IR checking for a parameter.
  if (ParamDeclRefAttr param = getIfParameter(*decl)) {
    if constexpr (isApproximateResult) {
      return PublicDeclKind::DK_PublicParameterDecl;
    } else {
      auto name = demangleParameterName(param.getName(), /*forUser*/ true);
      // The parent PublicFunctionDecl or PublicStructDecl is the one who owns
      // the docstring of this parameter, so it's easier to construct that view
      // and extract the parameter from it.
      auto getParamFromParent =
          [&](auto &parent) -> std::unique_ptr<PublicDecl> {
        for (const PublicParameterDecl &param : parent->parameters)
          if (param.getName() == name)
            return std::make_unique<PublicParameterDecl>(param);
        return nullptr;
      };
      std::unique_ptr<PublicDecl> parent = getParent().getDecl();
      if (!parent)
        return nullptr;
      return TypeSwitch<PublicDecl *, std::unique_ptr<PublicDecl>>(&*parent)
          .Case<PublicFunctionDecl, PublicStructDecl>(getParamFromParent)
          .Default({nullptr});
    }
  }

  return {};
}

std::unique_ptr<PublicDecl> MojoASTDeclRef::getDecl() const {
  return getDeclImpl<PublicDeclInstance>();
}

std::optional<PublicDeclKind> MojoASTDeclRef::getApproximateDeclKind() const {
  return getDeclImpl<ApproximatePublicDeclKind>();
}

//===----------------------------------------------------------------------===//
// MojoASTTypeRef
//===----------------------------------------------------------------------===//

MojoASTDeclRef MojoASTTypeRef::getDecl(SharedState &shared) {
  return MojoASTDeclRef(type.getDecl(shared));
}

std::string MojoASTTypeRef::getAsString(SharedState &shared) const {
  return type.getAsString(/*ctx=*/{&shared});
}

/// If the current type is a reference, return the type of the pointee. This
/// aborts if the current type isn't a reference.
MojoASTTypeRef MojoASTTypeRef::getReferenceElementType() const {
  return type.getReferenceElementType();
}

Type MojoASTTypeRef::getMLIRType() const { return type.mlirType; }
