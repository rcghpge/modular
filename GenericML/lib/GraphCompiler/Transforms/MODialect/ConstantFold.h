//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallSet.h"

#include <numeric>
namespace M::MO {

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
    llvm::SmallVector<Operation *> opsToRemove = {op};
    while (!opsToRemove.empty()) {
      Operation *op = opsToRemove.pop_back_val();
      for (auto dep : op->getUsers())
        if (ops.contains(dep))
          opsToRemove.push_back(dep);
      ops.remove(op);
    }
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

namespace {
struct ValueParamCompare {
  bool operator()(const Operation *lhs, const Operation *rhs) const {
    return hash_value(lhs) < hash_value(rhs);
  }
};
} // namespace

// Consider the undirected version of the subgraph, return all connected
// components of the graph.
static SmallVector<Subgraph> getSubgraphComponents(const Subgraph &subgraph) {
  if (subgraph.ops.empty())
    return {};

  // Disjoint set containing connected components.
  llvm::EquivalenceClasses<Operation *, ValueParamCompare> newSubgraphs;
  for (Operation *curOp : subgraph.ops) {
    newSubgraphs.insert(curOp);
    for (Operation *other : curOp->getUsers())
      if (subgraph.ops.contains(other))
        newSubgraphs.unionSets(other, curOp);
  }

  // Create new subgraphs for each connected component.
  SmallVector<Subgraph> answer;
  for (auto it = newSubgraphs.begin(), end = newSubgraphs.end(); it != end;
       ++it) {
    if (!it->isLeader())
      continue;
    answer.emplace_back(llvm::SetVector<Operation *>{
        newSubgraphs.member_begin(it), newSubgraphs.member_end()});
  }
  return answer;
}

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
    while (!subgraphs.empty()) {
      Subgraph subgraph = subgraphs.pop_back_val();
      std::optional<int64_t> maybeInitialMem = constantMemUsage(subgraph);

      // Greedily remove outputs with the largest size and ensure no
      // outputs are the result of a direct broadcast.
      std::optional<int64_t> maybeOutputMem = outputMemUsage(subgraph);

      // Cannot determine memory growth if parameterized inputs.
      // In this case be safe and don't fold the subgraphs for now.
      if (!maybeInitialMem.has_value())
        continue;

      int64_t initialMem = *maybeInitialMem;

      int64_t factorLimit = floor(initialMem * factor);
      int64_t totalThreshold = std::max(factorLimit, threshold);

      auto memoryOk = [&]() {
        return maybeOutputMem.has_value() && *maybeOutputMem <= totalThreshold;
      };
      bool maybeBcastsOutputs = true;

      // TODO: this is N^2, we should improve it #18475
      while (!memoryOk() || maybeBcastsOutputs) {
        // Implies parameterized shape in output, cannot determine true size
        // so defensively don't fold this subgraph.
        if (!maybeOutputMem.has_value())
          break;

        if (subgraph.outputs.empty())
          break;

        // Remove largest value outputs to get under memory thresholds
        if (!memoryOk()) {
          // Find and remove the largest value
          Operation *maxValue = *std::max_element(
              subgraph.outputs.begin(), subgraph.outputs.end(),
              [&subgraph, totalThreshold](Operation *a, Operation *b) {
                std::optional<int64_t> maybeSizeA = valueSize(subgraph, a);
                std::optional<int64_t> maybeSizeB = valueSize(subgraph, b);

                // If unknown, set to a big enough number
                int64_t sizeA = maybeSizeA.has_value() ? maybeSizeA.value()
                                                       : totalThreshold + 1;
                int64_t sizeB = maybeSizeB.has_value() ? maybeSizeB.value()
                                                       : totalThreshold + 1;

                return sizeA < sizeB;
              });

          // TODO: this is N^2, we should improve it #18475
          subgraph.remove(maxValue);

          // New outputs means we might have broadcasts again.
          maybeBcastsOutputs = true;
        }

        // Broadcasts are not helpful to fold as they can be represented as view
        // ops and only increase or maintain memory usage. They are also used in
        // many symbolic optimizations under "broadcast sinking."
        if (maybeBcastsOutputs) {
          maybeBcastsOutputs = false;
          SmallVector<Operation *> initOutputs(subgraph.outputs.begin(),
                                               subgraph.outputs.end());
          for (Operation *op : initOutputs) {
            if (isa<BroadcastToOp, Static::BroadcastToOp>(op)) {
              maybeBcastsOutputs = true;
              subgraph.remove(op);
            }
          }
        }

        maybeOutputMem = outputMemUsage(subgraph);
      }

      // The removal process above could have separated our graph into multiple
      // components. These can help mask increasing memory use (imagine a
      // big memory component allowing a small memory component balloon in
      // size). Therefore, only fold single component graphs.
      SmallVector<Subgraph> result = getSubgraphComponents(subgraph);
      ASSERT_STREAM(!result.empty(), "Expect non-empty subgraph!");

      // NOTE: do not fold subgraphs with parameters in them.
      // We assume shape inference will clean up such shape inference graphs.
      if (result.size() == 1 && memoryOk() &&
          !llvm::all_of(subgraph.outputs, [](Operation *op) {
            return dyn_cast<ConstantOp>(op) ||
                   dyn_cast<KGEN::ParamDeclareOp>(op);
          })) {
        foldThese.push_back(subgraph);
      } else if (result.size() > 1) {
        // Implies we have multiple components, add them back to the work-list
        subgraphs.append(result.begin(), result.end());
      }
    }
    return foldThese;
  }
  ~FactorMemIncrease() override = default;

  // Size of Value in bytes.
  // Returns std::nullopt if it is unknown (e.g. parameterized shape).
  static std::optional<int64_t> valueSize(Value v) {
    auto tensor = dyn_cast<TensorType>(v.getType());
    if (tensor) {
      std::optional<int64_t> numElements = tensor.getNumElements();
      if (!numElements.has_value())
        return {};
      return numElements.value() * tensor.getDtype().getSizeInBytes();
    }

    // index or shape have nominal sizes
    return 0;
  }

  static std::optional<int64_t> valueSize(const Subgraph &subgraph,
                                          Operation *op) {
    int64_t sum = 0;
    for (auto result : op->getResults()) {
      if (!subgraph.fullyContains(result)) {
        std::optional<int64_t> localSize = valueSize(result);
        if (!localSize.has_value())
          return {};
        sum += localSize.value();
      }
    }
    return sum;
  }

  // Size of ConstantOp data in bytes.
  // Returns std::nullopt if it is unknown (e.g. parameterized shape).
  static std::optional<int64_t> valueSize(ConstantOp op) {
    return valueSize(op.getResult());
  }

  // Calculate the size in bytes of memory used by the outputs of a
  // subgraph.
  // Returns std::nullopt if it is unknown (e.g. parameterized shape).
  static std::optional<int64_t> outputMemUsage(const Subgraph &subgraph) {
    int64_t outputSize = 0;
    for (auto output : subgraph.outputs) {
      std::optional<int64_t> localSize = valueSize(subgraph, output);

      if (!localSize.has_value())
        return {};

      outputSize += localSize.value();
    }
    return outputSize;
  }

  // Calculate the size in bytes of memory used by all constant ops in the
  // subgraph.
  // Returns std::nullopt if it is unknown (e.g. parameterized shape).
  static std::optional<int64_t> constantMemUsage(const Subgraph &subgraph) {
    int64_t internalSize = 0;
    for (auto op : subgraph.ops) {
      if (!subgraph.outputs.contains(op)) {
        auto constant = dyn_cast<ConstantOp>(op);
        if (constant) {
          auto localSize = valueSize(constant);
          if (!localSize.has_value())
            return {};
          internalSize += localSize.value();
        }
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
