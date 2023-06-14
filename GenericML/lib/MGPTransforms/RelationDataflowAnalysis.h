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
public:
  IndexSet<V> *index;

protected:
  using Set = typename IndexSet<V>::Set;
  using Equivalence = typename IndexSet<V>::Equivalence;

  /// A relation R between pair of Vs, represented as:
  ///   (x, y) in R iff x |-> set and y in set.
  ///
  /// Can also encode a predicate P over Vs, represented as:
  ///   P(v) iff v |-> set and distinguished in set.
  /// This way we only need to implement one concrete dataflow domain for all
  /// the analyses we need, and we can make all the transfer functions
  /// similarly concrete on that domain.
  ///
  /// However we compress the map storage slightly:
  ///  1. The relation is implicitly reflexive, so if x |-> set then
  ///     x will not appear in set.
  ///  2. If, after the above, the set is empty, then the map has no entry
  ///     for x.
  ///  3. If the set is just {distinguished} then x maps to the empty set
  ///     (ie the default zero sized bitvector).
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
    Set emptySet;
    Set fullSet;

    DomainMap(Relation &relation, Set &&emptySet, Set &&fullSet)
        : relation(relation), emptySet(std::move(emptySet)),
          fullSet(std::move(fullSet)) {}

    void setIntersection(Set &lhs, const Set &rhs) const { lhs &= rhs; }
    void setUnion(Set &lhs, const Set &rhs) const { lhs |= rhs; }
  };

  RelationDataflowAnalysisBase(IndexSet<V> *index)
      : index(index), equivalence(index->identityEquivalence()),
        changed(false) {}

  /// Updates domainMap so that v |-> {v}, up to equivalence. Marks changed true
  /// if domainMap was changed. Accounts for the above compression rules.
  void setId(DomainMap &domainMap, V v) {
    v = index->getCanonical(equivalence, v);
    auto itr = domainMap.relation.find(v);
    if (itr != domainMap.relation.end()) {
      domainMap.relation.erase(itr);
      changed = true;
    }
  }

  /// Returns the compressed representation of set, suitable for
  /// storing in a relation map with key v. Returns nullopt if no
  /// entry for v should be stored.
  std::optional<Set> compress(V v, const Set &set) const {
    assert(set.size() == index->size());

    size_t count = set.count();
    if (index->setContains(set, v))
      // No need to explicitly represent reflexivity.
      --count;

    // Represent {} as no mapping.
    if (count == 0)
      return std::nullopt;

    // Represent {distinguished} as the empty bitvector.
    if (set.test(0) && count == 1) // distinguished always has index 0
      return Set();

    Set compressedSet = set;
    // No need to explicitly represent reflexivity.
    index->removeFromSet(compressedSet, v);
    return compressedSet;
  }

  /// Inverse of compress.
  Set expand(V v, const std::optional<Set> &compressedSet) const {
    if (!compressedSet)
      // Recover implicit reflexivity.
      return index->unitSet(v);

    if (compressedSet->empty()) {
      // Recover implicit reflexivity and the distinguished entry.
      Set set = index->unitSet(v);
      set.set(0); // distinguished always has index 0
      return set;
    }

    // Recover implicit reflexivity.
    Set set = *compressedSet;
    index->addToSet(set, v);
    return set;
  }

  /// Updates domainMap so that v |-> set, up to equivalence. Marks changed true
  /// if domainMap was changed. Accounts for the above compression rules.
  void setEntriesAndId(DomainMap &domainMap, V v, const Set &set) {
#define DEBUG_TYPE "mgp-dataflow"
    LLVM_DEBUG({
      llvm::dbgs() << "setEntriesAndId(";
      index->printV(llvm::dbgs(), v);
      llvm::dbgs() << ", ";
      index->printVs(llvm::dbgs(), index->setToVector(set));
      llvm::dbgs() << ")\n";
    });

    v = index->getCanonical(equivalence, v);
    auto itr = domainMap.relation.find(v);
    std::optional<Set> compressedSet = compress(v, set);

    if (!compressedSet) {
      if (itr != domainMap.relation.end()) {
        domainMap.relation.erase(itr);
        changed = true;
      }
      return;
    }

    if (itr == domainMap.relation.end()) {
      domainMap.relation.insert({v, std::move(*compressedSet)});
      changed = true;
      return;
    }

    if (itr->second != *compressedSet) {
      LLVM_DEBUG({
        llvm::dbgs() << "not equal to existing set ";
        index->printVs(llvm::dbgs(), index->setToVector(itr->second));
        llvm::dbgs() << ")\n";
      });
      itr->second = std::move(*compressedSet);
      changed = true;
    }
#undef DEBUG_TYPE
  }

  /// If using the relation to represent a predicate, includes v in the
  /// predicate, up to equivalence.
  void setTrue(DomainMap &domainMap, V v) {
#define DEBUG_TYPE "mgp-dataflow"
    LLVM_DEBUG({
      llvm::dbgs() << "setTrue(";
      index->printV(llvm::dbgs(), v);
      llvm::dbgs() << ")\n";
    });

    v = index->getCanonical(equivalence, v);
    auto itr = domainMap.relation.find(v);
    if (itr == domainMap.relation.end()) {
      domainMap.relation.insert({v, Set()});
      changed = true;
      return;
    }

    if (!itr->second.empty()) {
      itr->second = Set();
      changed = true;
    }
#undef DEBUG_TYPE
  }

  /// Returns the set bound to v, up to equivalence. Accounts for the
  /// above compression rules.
  Set getEntries(const DomainMap &domainMap, V v) const {
    return getEntries(domainMap.relation, v);
  }

  Set getEntries(const Relation &relation, V v) const {
    v = index->getCanonical(equivalence, v);
    auto itr = relation.find(v);
    if (itr == relation.end())
      return expand(v, std::nullopt);
    else
      return expand(v, itr->second);
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

  /// Updates equivalence such that if (x, y) is in relation then x and y are
  /// considered equivalent. In other words, makes the equivalence represent
  /// relation closed under reflexivity, symmetry, and transitivity.
  void addRelationToEquivalence(const Relation &relation) {
    for (const auto &[lhs, compressedSet] : relation) {
      SmallVector<Value> rhss = index->setToVector(expand(lhs, compressedSet));
      for (auto rhs : rhss)
        index->setEquivalent(equivalence, lhs, rhs);
    }
  }

  /// Returns inverse of relation.
  Relation inverse(const Relation &relation) {
    // First build result in non-compressed form.
    Relation uncompressed;
    for (const auto &[lhs, compressedSet] : relation) {
      Set set = expand(lhs, compressedSet);
      for (const auto rhs : index->setToVector(set)) {
        auto itr = uncompressed.find(rhs);
        if (itr == uncompressed.end())
          uncompressed.insert({rhs, index->unitSet(lhs)});
        else
          index->addToSet(itr->second, lhs);
      }
    }
    // Now compress the relation.
    Relation result;
    for (const auto &[rhs, set] : uncompressed) {
      std::optional<Set> compressedSet = compress(rhs, set);
      if (compressedSet)
        result.insert({rhs, std::move(*compressedSet)});
    }
    return result;
  }

  /// Pretty-prints equivalence, using the stable form for canonical values.
  /// Singleton entries are not printed since implied. For testing only.
  void printEquivalence(llvm::raw_ostream &os, StringRef equivName,
                        const Equivalence &equivalence) {
    os << equivName << " begin\n";
    for (V canon : index->getAllVs()) {
      if (canon != index->getStableCanonical(equivalence, canon))
        continue;
      SmallVector<Value> equivClass =
          index->setToVector(index->getEquivalenceClass(equivalence, canon));
      if (equivClass.size() == 1)
        continue;
      os << "  ";
      index->printVs(os, equivClass);
      os << "\n";
    }
    os << equivName << " end\n";
  }

  /// Converts relation to use stable canonical Vs w.r.t. equivalence. For
  /// testing only.
  Relation toStableForm(const Relation &relation) {
    Relation result;
    for (const auto &[v, compressedSet] : relation) {
      Set set = expand(v, compressedSet);
      Set stableCanonSet(set.size());
      for (auto x : index->setToVector(set))
        index->addToSet(stableCanonSet,
                        index->getStableCanonical(equivalence, x));
      Value canonV = index->getStableCanonical(equivalence, v);
      std::optional<Set> compressedStableSet = compress(canonV, stableCanonSet);
      if (compressedStableSet)
        result.insert({canonV, *compressedStableSet});
    }
    return result;
  }

  /// Pretty-prints relation in index order. For testing only.
  void printRelation(llvm::raw_ostream &os, StringRef relName,
                     const Relation &relation) {
    os << relName << " begin\n";
    for (V v : index->getAllVs()) {
      auto itr = relation.find(v);
      if (itr == relation.end())
        continue;
      os << "  ";
      index->printV(os, v);
      os << " |-> ";
      index->printVs(os, index->setToVector(expand(v, itr->second)));
      os << "\n";
    }
    os << relName << " end\n";
  }

  /// Pretty-prints relation encoding a predicate in index order. For testing
  /// only.
  void printPredicate(llvm::raw_ostream &os, StringRef predName,
                      const Relation &relation) {
    Set predicate = index->emptySet();
    for (const auto &[v, compressedSet] : relation) {
      Set set = expand(v, compressedSet);
      if (index->setContains(set, index->distinguished))
        index->addToSet(predicate, v);
    }
    os << predName << "(";
    index->printVs(os, index->setToVector(predicate));
    os << ")\n";
  }

  /// Grows domainMap for the given operation, calling into helpers where
  /// appropriate.
  using FlowOp = std::function<void(Operation *, DomainMap &)>;

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
  Set operandCombination(const DomainMap &domainMap, Value value) const;

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
