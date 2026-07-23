//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace M::MOGG {

#define GEN_PASS_DEF_REWRITEOPAQUETYPES
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct RewriteOpaqueTypesPass
    : public impl::RewriteOpaqueTypesBase<RewriteOpaqueTypesPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    // Read the simple → fully-qualified mapping persisted by
    // `mogg-import-packages`. Absence is a valid no-op: any consumer is
    // free to run this pass on a module that has no opaque-type references.
    auto mappingAttr =
        module->getAttrOfType<mlir::DictionaryAttr>(kOpaqueTypeMappingAttrName);
    // Safety net for the kernel-struct exclusion in `buildOpaqueTypeMapping`:
    // collect the leaf name of every kernel/primitive decl in the module so an
    // `!mo.opaque<"Foo">` that names an (excluded) registered kernel struct
    // fails loudly here instead of silently passing through unqualified and
    // surfacing as a confusing far-from-the-cause failure during later
    // lowering. The decls are already materialized in the module at this
    // stage, so this needs no package rescan and no extra attribute.
    llvm::StringSet<> kernelLeafNames;
    auto leafOf = [](llvm::StringRef name) -> llvm::StringRef {
      return name.rsplit('.').second.empty() ? name : name.rsplit('.').second;
    };
    for (auto decl : module.getOps<KernelDeclOp>())
      kernelLeafNames.insert(leafOf(decl.getSymName()));
    for (auto decl : module.getOps<PrimitiveDeclOp>())
      kernelLeafNames.insert(leafOf(decl.getSymName()));

    bool sawKernelStructMisuse = false;
    if (mappingAttr && !mappingAttr.empty()) {
      mlir::AttrTypeReplacer replacer;
      replacer.addReplacement(
          [&](MO::OpaqueType opaqueTy) -> std::optional<mlir::Type> {
            llvm::StringRef symbol = opaqueTy.getSymbol().getValue();
            auto qualified = mappingAttr.getNamed(symbol);
            if (!qualified) {
              // An unqualified opaque type that names a registered kernel
              // struct is a misuse: kernel structs are implementations
              // resolved by registration, not opaque data types. Already-
              // qualified or genuinely-external opaque types carry dotted
              // names whose leaves never collide here, so this fires only on
              // the precise misuse.
              if (kernelLeafNames.contains(symbol)) {
                module.emitError()
                    << "opaque type '" << symbol
                    << "' refers to a registered kernel struct; kernel "
                       "structs are implementations resolved by registration, "
                       "not opaque data types";
                sawKernelStructMisuse = true;
              }
              return std::nullopt;
            }
            auto qualifiedStr =
                mlir::dyn_cast<mlir::StringAttr>(qualified->getValue());
            if (!qualifiedStr)
              return std::nullopt;
            return MO::OpaqueType::get(qualifiedStr, opaqueTy.getParameters());
          });
      replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/true);
      if (sawKernelStructMisuse)
        return signalPassFailure();
    }

    // The mapping has been applied (or there was nothing to apply);
    // downstream passes operate on fully-qualified names and don't need
    // it. Drop the attribute to keep the module clean.
    module->removeAttr(kOpaqueTypeMappingAttrName);
  }
};

} // namespace

} // namespace M::MOGG
