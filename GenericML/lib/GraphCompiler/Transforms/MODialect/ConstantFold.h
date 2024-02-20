//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/Support/Assert.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"

#include <numeric>
namespace M::MO {

namespace {

// Convert a TensorType to a ShapedType so it can be used with functions in
// Support. TensorType does not implement the ShapedType protocol so this is
// necessary.
ShapedType typeToShaped(TensorType type) {
  auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(type.getShapeAttr());
  GENERICML_ASSERT(shapeAttr)
      << "Shapes in constants subgraph should all be infered";
  auto intDims = shapeAttr.getIntDims();
  int64_t size = std::accumulate(intDims.begin(), intDims.end(), 1,
                                 std::multiplies<int64_t>());
  return ArrayType::get(type.getContext(), size, type.getElementType());
}

} // namespace

struct Subgraph {
  llvm::SetVector<Operation *> ops;
  // Outputs are Value in the constant subgraph that have uses outside of the
  // subgraph. Outputs may depend on other outputs.
  llvm::SetVector<Operation *> outputs;
  Subgraph(llvm::SetVector<Operation *> ops) : ops(std::move(ops)) {
    findOutputs();
  }

  /// Remove operation from subgraph. Also removes any dependents in the
  /// subgraph.
  void remove(Operation *op) {
    for (auto dep : op->getUsers())
      if (ops.contains(dep))
        ops.remove(dep);
    ops.remove(op);
    outputs.clear();
    findOutputs();
  }

  /// Is the Value completely internal to this subgraph.
  bool fullyContains(Value v) const {
    return llvm::all_of(v.getUsers(),
                        [this](Operation *user) { return ops.contains(user); });
  }

  /// List of parameters declared and used in this subgraph. Parameters are
  /// returned in topological order.
  llvm::SetVector<StringAttr> paramDeclarations() const;

  /// List of all constants contained in this subgraph.
  llvm::SmallVector<ConstantOp> constants() const {
    auto filtered = llvm::make_filter_range(
        ops, [](Operation *op) { return dyn_cast<ConstantOp>(op); });
    return llvm::to_vector(llvm::map_range(
        filtered, [](Operation *op) { return dyn_cast<ConstantOp>(op); }));
  }

private:
  // Set outputs to those Values in vals which have uses outside the subgraph.
  void findOutputs() {
    for (Operation *op : ops) {
      // Constant ops cant be outputs
      if (dyn_cast<ConstantOp>(op))
        continue;
      for (auto user : op->getUsers()) {
        if (dyn_cast<MOControlOpInterface>(user) || !ops.contains(user)) {
          // If any user is not const, then this is an output.
          outputs.insert(op);
          break;
        }
      }
    }
  }
};

// Interface for controlling whether or not a subgraph is constant folded.
struct ConstantFoldPolicy {
  // Determines which subgraphs should be constant folded. Must return the
  // passed in subgraphs or subsets of each of them. Inputs subgraphs not
  // included in the output will not be constant folded at all. I.e. only
  // the returned subgraphs will be constant folded.
  virtual SmallVector<Subgraph>
  shouldFold(SmallVector<Subgraph> subgraphs) const = 0;
  virtual ~ConstantFoldPolicy() = default;
};

// Fold the largest subgraph that does not increase memory used by more than
// a factor of the space used by the non-folded subgraph.
struct FactorMemIncrease : public ConstantFoldPolicy {
  FactorMemIncrease(double factor, int64_t threshold)
      : factor(factor), threshold(threshold) {}

  // A factor of 1 indicates no increase, a factor of 2 indicates 100% more
  // memory usage max.
  double factor;
  /// Fold everything under this size in bytes.
  int64_t threshold;

  SmallVector<Subgraph>
  shouldFold(SmallVector<Subgraph> subgraphs) const override {
    SmallVector<Subgraph> foldThese;
    for (auto &subgraph : subgraphs) {
      int64_t initialMem = constantMemUsage(subgraph);
      // Greedily remove outputs with the largest size. Might be a smarter
      // way to do this.
      int64_t outputMem = outputMemUsage(subgraph);
      while (!subgraph.outputs.empty() && outputMem > initialMem * factor &&
             outputMem > threshold) {
        // TODO: this is N^2, we should improve it #18475
        Operation *maxValue = *std::max_element(
            subgraph.outputs.begin(), subgraph.outputs.end(),
            [&subgraph](Operation *a, Operation *b) {
              return valueSize(subgraph, a) < valueSize(subgraph, b);
            });
        // TODO: this is N^2, we should improve it #18475
        subgraph.remove(maxValue);
        outputMem = outputMemUsage(subgraph);
      }
      if (!llvm::all_of(subgraph.outputs, [](Operation *op) {
            return dyn_cast<ConstantOp>(op) ||
                   dyn_cast<KGEN::ParamDeclareOp>(op);
          }))
        foldThese.push_back(subgraph);
    }
    return foldThese;
  }
  ~FactorMemIncrease() override = default;

  // Size of Value in bytes.
  static int64_t valueSize(Value v) {
    auto tensor = dyn_cast<TensorType>(v.getType());
    if (tensor)
      return typeToShaped(tensor).getNumElements() *
             tensor.getDtype().getSizeInBytes();

    // index or shape have nominal sizes
    return 0;
  }

  static int64_t valueSize(const Subgraph &subgraph, Operation *op) {
    int64_t sum = 0;
    for (auto result : op->getResults()) {
      if (!subgraph.fullyContains(result))
        sum += valueSize(result);
    }
    return sum;
  }

  // Size of ConstantOp data in bytes.
  static int64_t valueSize(ConstantOp op) { return valueSize(op.getResult()); }

  // Calculate the size in bytes of memory used by the outputs of a
  // subgraph.
  static int64_t outputMemUsage(const Subgraph &subgraph) {
    int64_t outputSize = 0;
    for (auto output : subgraph.outputs)
      outputSize += valueSize(subgraph, output);
    return outputSize;
  }

  // Calculate the size in bytes of memory used by all constant ops in the
  // subgraph.
  static int64_t constantMemUsage(const Subgraph &subgraph) {
    int64_t internalSize = 0;
    for (auto op : subgraph.ops) {
      if (!subgraph.outputs.contains(op)) {
        auto constant = dyn_cast<ConstantOp>(op);
        if (constant)
          internalSize += valueSize(constant);
      }
    }
    return internalSize;
  }
};

SmallVector<Subgraph>
constantSubgraphs(MO::GraphOp graph,
                  KGEN::ParameterCollector::Analysis &paramCache);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H
