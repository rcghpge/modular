//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Base classes for building forwards/backwards dense dataflow analyses for
// relations over values and operands which understand the
// grt.{if,while,repeat.i32} control flow ops.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPTRANSFORMS_BACKWARDSDATAFLOWANALYSIS_H
#define GENERICML_MGPTRANSFORMS_BACKWARDSDATAFLOWANALYSIS_H

#include "GenericML/MGPDialect/MGPOps.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#include "./IndexSet.h"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// RelationDataflowAnalysisBase
//===----------------------------------------------------------------------===//

/// Base class to supply relation domain operations shared by both the forwards
/// and backwards dataflow analyses.
template <typename V>
class RelationDataflowAnalysisBase {
protected:
  using Set = typename IndexSet<V>::Set;
  using Equivalence = typename IndexSet<V>::Equivalence;

  /// A relation between V pairs, represented as:
  ///   (x, y) in relation iff x |-> set and y in set.
  ///
  /// The relation can also encode a predicate P over Vs via:
  ///    (v, distinguished) is in the relation iff P(v)
  /// This way we only need to implement one dataflow domain for all the
  /// analyses we need, and we can make all the transfer functions concrete
  /// on that domain.
  ///
  /// However we compress the map storage slightly:
  ///  1. The relation is implicitly reflexive, so if x |-> set then
  ///     x will not appear in set.
  ///  2. If the set is empty then the map has no entry for x.
  ///  3. If the set is {distinguished} then x maps to the empty set.
  using Relation = DenseMap<V, Set>;

  /// A view of a relation under definition as map from Vs to domain with:
  ///  - top as the empty set
  ///  - bottom as the full set (or just a set with just the distinguished V
  ///    item if we are using the relation to represent a predicate)
  ///  - least-upper bound as set intersection.
  ///
  /// (We may end up also needing the inverted domain, eg for a mayAlias
  /// predicate. If so we'll capture the domain orientation here and rename
  /// these fields.)
  struct DomainMap {
    Relation &relation;
    Set fullSet;
    Set emptySet;

    DomainMap(Relation &relation, Set &&fullSet, Set &&emptySet)
        : relation(relation), fullSet(std::move(fullSet)),
          emptySet(std::move(emptySet)) {}

    void setIntersection(Set &lhs, const Set &rhs) const { lhs &= rhs; }
    void setUnion(Set &lhs, const Set &rhs) const { lhs |= rhs; }
  };

  RelationDataflowAnalysisBase(IndexSet<V> *index)
      : index(index), equivalence(index->identityEquivalence()),
        changed(false) {}

  /// Updates domainMap so that v |-> set, up to equivalence. Marks changed true
  /// if domainMap was changed. Accounts for the above compression rules.
  void setEntriesAndId(DomainMap &domainMap, V v, const Set &set) {
#define DEBUG_TYPE "mgp-dataflow"

    assert(set.size() == index->size());

    LLVM_DEBUG(
        llvm::dbgs() << "setEntriesAndId("; printV(*index, llvm::dbgs(), v);
        llvm::dbgs() << ", {";
        llvm::interleaveComma(index->setToVector(set), llvm::dbgs(),
                              [&](V w) { printV(*index, llvm::dbgs(), w); });
        llvm::dbgs() << "})\n");

    v = index->getCanonical(equivalence, v);
    auto itr = domainMap.relation.find(v);

    size_t count = set.count();
    if (index->setContains(set, v))
      // No need to explicitly represent reflexivity.
      --count;

    // Represent {} as no mapping.
    if (count == 0) {
      if (itr != domainMap.relation.end()) {
        domainMap.relation.erase(itr);
        changed = true;
      }
      return;
    }

    // Represent {distinguished} as the empty bitvector.
    if (set.test(0) && count == 1) { // distinguished always has index 0
      if (itr == domainMap.relation.end()) {
        domainMap.relation.insert({v, Set()});
        changed = true;
      } else if (!itr->second.empty()) {
        itr->second = Set();
        changed = true;
      }
      return;
    }

    Set storedSet = set;
    // No need to explicitly represent reflexivity.
    index->removeFromSet(storedSet, v);
    if (itr == domainMap.relation.end()) {
      domainMap.relation.insert({v, std::move(storedSet)});
      changed = true;
    } else if (itr->second != storedSet) {
      LLVM_DEBUG(
          llvm::dbgs() << "not equal to existing set {";
          llvm::interleaveComma(index->setToVector(itr->second), llvm::dbgs(),
                                [&](V w) { printV(*index, llvm::dbgs(), w); });
          llvm::dbgs() << "})\n");
      itr->second = std::move(storedSet);
      changed = true;
    }
#undef DEBUG_TYPE
  }

  /// If using the relation to represent a predicate, includes v in the
  /// predicate, up to equivalence.
  void setTrue(DomainMap &domainMap, V v) {
    setEntriesAndId(domainMap, v, domainMap.fullSet);
  }

  /// Returns the set bound to v, up to equivalence. Accounts for the
  /// above compression rules.
  Set getEntries(const DomainMap &domainMap, V v) const {
    v = index->getCanonical(equivalence, v);
    auto itr = domainMap.relation.find(v);
    if (itr == domainMap.relation.end())
      // Recover implicit reflexivity.
      return index->unitSet(v);
    if (itr->second.empty()) {
      // Recover implicit reflexivity and the distinguished entry.
      Set result = index->unitSet(v);
      result.set(0); // distinguished always has index 0
      return result;
    }
    // Recover implicit reflexivity.
    Set result = itr->second;
    index->addToSet(result, v);
    return result;
  }

  /// If using the relation to represent a predicate, returns true if v is
  /// in the predicate, up to equivalence.
  bool isTrue(const Relation &relation, V v) const {
    v = index->getCanonical(equivalence, v);
    auto itr = relation.find(v);
    if (itr == relation.end())
      return false;
    if (itr->second.empty())
      return true;
    if (itr->second.test(0)) // distinguished always has index 0
      return true;
    return false;
  }

  /// Returns true if (lhs, rhs) is in relation, up to equivalence.
  bool containsEntryOrId(const Relation &relation, V lhs, V rhs) const {
    lhs = index->getCanonical(equivalence, lhs);
    rhs = index->getCanonical(equivalence, rhs);
    if (lhs == rhs)
      // Relation is implicitly reflexive.
      return true;
    auto itr = relation.find(lhs);
    if (itr == relation.end())
      return false;
    if (itr->second.empty())
      return rhs == index->distinguished;
    return index->setContains(itr->second, rhs);
  }

  /// Updates the analysis equivalence such that if (x, y) is in relation then
  /// x and y are considered equivalent. In other words, makes the equivalence
  /// represent relation closed under reflexivity, symmetry, and transitivity.
  void addRelationToEquivalence(const Relation &relation) {
    for (const auto &[lhs, set] : relation) {
      size_t lhsIndex = index->getIndex(lhs);
      if (set.empty()) {
        equivalence.unionSets(lhsIndex,
                              0); // distinguished always has index 0
      } else {
        for (auto itr = set.set_bits_begin(), e = set.set_bits_end(); itr != e;
             ++itr)
          equivalence.unionSets(lhsIndex, *itr);
      }
    }
  }

  /// Grows domainMap for the given operation, calling into the following
  /// helpers where appropriate.
  using FlowOp = std::function<void(Operation *, DomainMap &)>;

  IndexSet<V> *index;
  Equivalence equivalence;
  bool changed;
};

//===----------------------------------------------------------------------===//
// ForwardsRelationDataflowAnalysis
//===----------------------------------------------------------------------===//

/// Base class to supply control flow transfer functions for all forwards
/// dataflow analyses for relations over values.
class ForwardsRelationDataflowAnalysis
    : public RelationDataflowAnalysisBase<Value> {
public:
  ForwardsRelationDataflowAnalysis(IndexSet<Value> *index)
      : RelationDataflowAnalysisBase<Value>(index) {}

protected:
  /// Grows domainMap for one iteration of the body sub-block of cfOp, which
  /// must have one sub-block. Sets changed to true if domainMap was changed.
  void flowSimpleLoop(GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
                      const FlowOp &flowOp);

  /// Grows domainMap for executing the condition sub-block of cfOp, which must
  /// have two sub-blocks. Sets changed to true if domainMap was changed.
  void flowCondition(GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
                     const FlowOp &flowOp);

  /// Grows domainMap for executing the body then condition sub-block of cfOp,
  /// which must have two sub-blocks. Sets changed to true if domainMap was
  /// changed.
  void flowConditionedLoop(GRT::GRTControlFlowInterface cfOp,
                           DomainMap &domainMap, const FlowOp &flowOp);

  /// Grows domainMap for the sub-blocks of branch control flow op.
  /// Sets changed to true if domainMap was changed.
  void flowBranch(GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
                  const FlowOp &flowOp);

  /// Grows domainMap using the control flow closure rules. Sets changed to
  /// true if domainMap was changed.
  void flowControlFlow(GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
                       const FlowOp &flowOp);

  /// Grows domainMap to account for the results of op.
  void flowDefault(Operation *op, DomainMap &domainMap);

  /// Grows domainMap for the given block.
  void flowBlock(Block *block, DomainMap &domainMap, const FlowOp &flowOp);

  /// Grows domainMap for modelOp.
  void flowModel(ModelOp modelOp, DomainMap &domainMap, const FlowOp &flowOp);
};

//===----------------------------------------------------------------------===//
// BackwardsRelationDataflowAnalysis
//===----------------------------------------------------------------------===//

/// Base class to supply control flow transfer functions for all backwards
/// dataflow analyses for relations over operands.
class BackwardsRelationDataflowAnalysis
    : public RelationDataflowAnalysisBase<const OpOperand *> {
public:
  BackwardsRelationDataflowAnalysis(IndexSet<const OpOperand *> *index)
      : RelationDataflowAnalysisBase<const OpOperand *>(index) {}

protected:
  /// Returns the intersection of all the operand occurrences of value
  /// (possibly none) and the top set.
  Set operandIntersection(const DomainMap &domainMap, Value value) const;

  /// Grows domainMap for one iteration of the body sub-block of cfOp, which
  /// must have one sub-block. Sets changed to true if domainMap was changed.
  void flowSimpleLoop(GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
                      const FlowOp &flowOp);

  /// Grows domainMap for executing the condition sub-block of cfOp, which must
  /// have two sub-blocks. Sets changed to true if domainMap was changed.
  void flowCondition(GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
                     const FlowOp &flowOp);

  /// Grows domainMap for executing the body then condition sub-block of cfOp,
  /// which must have two sub-blocks. Sets changed to true if domainMap was
  /// changed.
  void flowConditionedLoop(GRT::GRTControlFlowInterface cfOp,
                           DomainMap &domainMap, const FlowOp &flowOp);

  /// Grows domainMap for the sub-blocks of branch control flow op.
  /// Sets changed to true if domainMap was changed.
  void flowBranch(GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
                  const FlowOp &flowOp);

  /// Grows domainMap using the control flow closure rules. Sets changed to
  /// true if domainMap was changed.
  void flowControlFlow(GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
                       const FlowOp &flowOp);

  /// Graws domainMap to account for the operands of op.
  void flowDefault(Operation *op, DomainMap &domainMap);

  /// Grows domainMap for the given block.
  void flowBlock(Block *block, DomainMap &domainMap, const FlowOp &flowOp);

  /// Grows domainMap for modelOp.
  void flowModel(ModelOp modelOp, DomainMap &domainMap, const FlowOp &flowOp);
};

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_BACKWARDSDATAFLOWANALYSIS_H
