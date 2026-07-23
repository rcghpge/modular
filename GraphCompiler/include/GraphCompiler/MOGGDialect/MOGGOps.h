//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H

#include "GraphCompiler/Interfaces/EmitsMojoInterface.h"
#include "GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGEnums.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace M::KGEN {
class ParamDeclareRegionOp;
} // namespace M::KGEN

// Needed because ODS-generated build methods for mogg.kernel.decl use the
// unqualified name.
using M::KGEN::LIT::FnTypeGeneratorType;

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/MOGGDialect/MOGG.h.inc"

namespace M::MOGG {

template <typename Kernel>
SmallVector<KGEN::ParamDeclRefAttr> getKernelUsedParameters(Kernel kernel) {
  DenseSet<KGEN::ParamDeclRefAttr> seenParams;
  SmallVector<KGEN::ParamDeclRefAttr> usedParameters;

  auto params = kernel.getOutputParamDecls();
  auto isInOutputParams = [&](StringRef name) {
    for (KGEN::ParamDeclAttr attr : params) {
      if (attr.getName().str() == name)
        return true;
    }
    return false;
  };

  // A walker allows us to crawl the parameters used within expression
  // attributes.
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](KGEN::ParamDeclRefAttr param) {
    if (auto [_, inserted] = seenParams.insert(param); !inserted)
      return WalkResult::advance();

    if (isInOutputParams(param.getName()))
      return WalkResult::advance();

    usedParameters.push_back(param);
    return WalkResult::advance();
  });

  // Track all unique parameters from a given tensor that we haven't seen
  // already.
  auto addParamsFromTensor = [&](Type t) {
    MOSH::ShapeAttr shape;

    // The type can either be a shape or a tensor.
    MO::TensorType asTensor = MO::dynCastFromTensorLike(t);
    if (asTensor)
      shape = asTensor.getTypedShapeAttr();

    if (!shape)
      return;
    walker.walk(shape);
  };

  // Get all parameters used by an operation.
  auto addParamsFromOp = [&](Operation *op) {
    for (Type t : op->getOperandTypes())
      addParamsFromTensor(t);

    for (Type t : op->getResultTypes())
      addParamsFromTensor(t);

    if (auto getDim = dyn_cast<MOGG::GetDimOp>(op)) {
      auto ref = KGEN::ParamDeclRefAttr::get(
          getDim.getDimension(), MOSH::ShapeAttr::getDimType(op->getContext()));
      auto itr = seenParams.find(ref);

      if (itr == seenParams.end()) {
        usedParameters.push_back(ref);
        seenParams.insert(ref);
      }
    }
  };

  // Start with the inputs / outputs on the kernel. Generally this would be all
  // parameters.
  addParamsFromOp(kernel);

  // However rarely we could create a kernel which internally uses a parameter
  // which isn't known from the inputs. I.E

  // clang-format off
  // %0 = broadcast(%input) : tensor<[N]> -> tensor<N, J>
  // %1 = broadcast(%0) : tensor<[N, J]> -> tensor<N, B>
  // %3 = broadcast(%1) : tensor<[N, B, K]> -> tensor<N, B, K, M>
  // clang-format on

  // We would see `N` on the input and `B` / `K` / `M` on the outputs but the
  // intermediate `J` is not on either.

  // To capture the above legal composition we walk the kernel ops as well.
  kernel.walk(addParamsFromOp);

  return usedParameters;
}
} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
