//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace M::MOGG {

#define GEN_PASS_DEF_REWRITEOPAQUETYPES
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

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
    if (mappingAttr && !mappingAttr.empty()) {
      mlir::AttrTypeReplacer replacer;
      replacer.addReplacement(
          [&](MO::OpaqueType opaqueTy) -> std::optional<mlir::Type> {
            auto qualified =
                mappingAttr.getNamed(opaqueTy.getSymbol().getValue());
            if (!qualified)
              return std::nullopt;
            auto qualifiedStr =
                mlir::dyn_cast<mlir::StringAttr>(qualified->getValue());
            if (!qualifiedStr)
              return std::nullopt;
            return MO::OpaqueType::get(qualifiedStr, opaqueTy.getParameters());
          });
      replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/true);
    }

    // The mapping has been applied (or there was nothing to apply);
    // downstream passes operate on fully-qualified names and don't need
    // it. Drop the attribute to keep the module clean.
    module->removeAttr(kOpaqueTypeMappingAttrName);
  }
};

} // namespace

} // namespace M::MOGG
