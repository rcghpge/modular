//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Base classes for building forwards/backwards dense dataflow analyses for
// relations over values and operands which understand control flow operations.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPTRANSFORMS_BACKWARDSDATAFLOWANALYSIS_H
#define GENERICML_MGPTRANSFORMS_BACKWARDSDATAFLOWANALYSIS_H

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/IndexSet.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "Support/AssertStream.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// RelationDataflowAnalysisBase
//===----------------------------------------------------------------------===//

/// Base class to supply relation domain operations shared by both the forwards
/// and backwards dataflow analyses. Captures the index representing the
/// bijection from Vs to indexes needed by the set (bitvector) and equivalence
/// (over size_t) representations. Supplies DomainMap to represent the
/// relation under construction. The same 'Analysis' instance can be used
/// across multiple DomainMaps.
template <typename V>
class RelationDataflowAnalysisBase {
#define DEBUG_TYPE "mgp-dataflow"
public:
  using Set = typename IndexSet<V>::Set;
  using Equivalence = typename IndexSet<V>::Equivalence;

  IndexSet<V> *index;

  /// A relation R between pair of Vs, represented as:
  ///   (x, y) in R iff x |-> set and y in set.
  ///
  /// Can also encode a predicate P over Vs, represented as:
  ///   P(v) iff v |-> set and distinguished in set.
  /// This way we only need to implement one concrete dataflow domain for all
  /// the analyses we need, and we can make all the transfer functions
  /// similarly concrete on that domain.
  using Relation = DenseMap<V, Set>;

  /// Orientation for the set domain.
  enum SetOrientation {
    // Use when soundness requires over-approximation.
    //   lub = union, glb = intersection, bottom = {}, top = all elems
    // It is possible for top to contain just the distinguished value when
    // trying to solve for a predicate instead of a relation.
    kLubIsUnion,
    // Use when soundness requires under-approximation.
    //   lub = intersection, glb = union, bottom = all elems, top = {}
    // It is possible for bottom to contain just the distinguished value when
    // trying to solve for a predicate instead of a relation.
    kLubIsIntersection
  };

  /// A view of a relation under definition as map from Vs to a lattice domain
  /// with a bottom, top, lub (least upper bound) operation, and glb (greatest
  /// lower bound) operation. The interpretation of the domain values and
  /// operations depends on the given orientation.
  struct DomainMap {
    /// The relation under definition.
    Relation &relation;
    /// How to implement lub and glb.
    SetOrientation orientation;
    /// Bottom and top elements, consistent with orientation.
    Set bottom;
    Set top;

    /// If true, account for the mutual-exclusivity of the sub-blocks of a
    /// kBranch control flow op which can be executed at most once.
    ///
    /// For example, given:
    ///    %a = ...
    ///    %b = ...
    ///    %c = ...
    ///    %d = if ... (%b, %c) {
    ///      %w = ...
    ///      mef.output (%x)
    ///    } else {
    ///      %y = ...
    ///      mef.output (%z)
    ///    }
    /// assume we have %x |-> {%x, %w, %b, %a} and %z |-> {%z, %y, %c, %a}.
    ///
    /// With handleDisjointBranches false we'll take %d to be:
    ///    %d |->  {%x, %w, %b, %a} lub {%z, %y, %c, %a}.
    ///
    /// However with assumeDisjoint true we'll take %d to be:
    ///    %d |-> {%x, %w} glb {%z, %y} glb ({%b, %a} lub {%c, %a})
    /// where the sets for %x and %z have been split according to whether each
    /// element is bound within vs without the kBronch control flow op
    /// sub-blocks.
    ///
    /// Whether this is semantically meaningful depends on the semantics of
    /// the relation and how it should account for values which may never be
    /// bound at runtime.
    bool handleDisjointBranches;

    /// Set to true if the relation is changed by leastUpperBoundEntries.
    bool changed = false;

    DomainMap(Relation &relation, SetOrientation orientation, Set &&emptySet,
              Set &&fullSet, bool handleDisjointBranches = false)
        : relation(relation), orientation(orientation),
          handleDisjointBranches(handleDisjointBranches) {
      switch (orientation) {
      case kLubIsUnion:
        bottom = std::move(emptySet);
        top = std::move(fullSet);
        break;
      case kLubIsIntersection:
        bottom = std::move(fullSet);
        top = std::move(emptySet);
        break;
      }
    }

    void leastUpperBound(Set &lhs, const Set &rhs) const {
      switch (orientation) {
      case kLubIsUnion:
        lhs |= rhs;
        break;
      case kLubIsIntersection:
        lhs &= rhs;
        break;
      }
    }

    void greatestLowerBound(Set &lhs, const Set &rhs) const {
      switch (orientation) {
      case kLubIsUnion:
        lhs &= rhs;
        break;
      case kLubIsIntersection:
        lhs |= rhs;
        break;
      }
    }
  };

  RelationDataflowAnalysisBase(IndexSet<V> *index) : index(index) {}

  /// Returns the relation with v |-> {} for all vs.
  Relation emptyRelation() const {
    Relation relation;
    for (V v : index->getAllVs()) {
      if (v == index->distinguished)
        continue;
      relation.insert({v, index->emptySet()});
    }
    return relation;
  }

  /// Returns the set bound to v.
  const Set &getEntries(const Relation &relation, V v) const {
    auto itr = relation.find(v);
    LLVM_DEBUG({
      if (itr == relation.end()) {
        llvm::dbgs() << "getEntries: no entry for ";
        index->printV(llvm::dbgs(), v);
        llvm::dbgs() << "\n";
      }
    });

    assert(itr != relation.end());
    return itr->second;
  }

  const Set &getEntries(const DomainMap &domainMap, V v) const {
    return getEntries(domainMap.relation, v);
  }

  /// Returns the set bound to v, unioned with v itself.
  Set getEntriesAndId(const Relation &relation, V v) const {
    Set set = getEntries(relation, v);
    index->addToSet(set, v);
    return set;
  }

  Set getEntriesAndId(const DomainMap &domainMap, V v) const {
    return getEntriesAndId(domainMap.relation, v);
  }

  /// If using the relation to represent a predicate, returns true if v is
  /// in the predicate.
  bool isTrue(const Relation &relation, V v) const {
    auto itr = relation.find(v);
    LLVM_DEBUG({
      if (itr == relation.end()) {
        llvm::dbgs() << "isTrue: no entry for ";
        index->printV(llvm::dbgs(), v);
        llvm::dbgs() << "\n";
      }
    });
    assert(itr != relation.end());
    return itr->second.test(0); // distinguished always has index 0
  }

  /// Returns true if (lhs, rhs) is in relation.
  bool containsEntry(const Relation &relation, V lhs, V rhs) const {
    auto itr = relation.find(lhs);
    LLVM_DEBUG({
      if (itr == relation.end()) {
        llvm::dbgs() << "containsEntry: no entry for ";
        index->printV(llvm::dbgs(), lhs);
        llvm::dbgs() << "\n";
      }
    });
    assert(itr != relation.end());
    return index->setContains(itr->second, rhs);
  }

  /// Updates relation so that v |-> set.
  void setEntries(Relation &relation, V v, const Set &set) const {
    LLVM_DEBUG({
      llvm::dbgs() << "setEntries(";
      index->printV(llvm::dbgs(), v);
      llvm::dbgs() << ", ";
      index->printVs(llvm::dbgs(), index->setToVector(set));
      llvm::dbgs() << ")\n";
    });

    auto itr = relation.find(v);
    if (itr == relation.end()) {
      relation.insert({v, set});
    } else {
      // Since may be setting within a fixed-point iteration, it is valid
      // for itr->second to not equal set.
      itr->second = set;
    }
  }

  void setEntries(DomainMap &domainMap, V v, const Set &set) const {
    setEntries(domainMap.relation, v, set);
  }

  /// Updates relation so that v |-> {}.
  void setEmpty(Relation &relation, V v) const {
    LLVM_DEBUG({
      llvm::dbgs() << "setEmpty(";
      index->printV(llvm::dbgs(), v);
      llvm::dbgs() << ")\n";
    });

    auto itr = relation.find(v);
    if (itr == relation.end()) {
      relation.insert({v, index->emptySet()});
    } else {
      // Even if we are setting to empty within a fixed-point iteration, an
      // empty entry must always be empty.
      assert(itr->second.count() == 0);
    }
  }

  void setEmpty(DomainMap &domainMap, V v) const {
    setEmpty(domainMap.relation, v);
  }

  /// If the relation does not have an entry for v, set it to {}.
  void setDefault(Relation &relation, V v) const {
    if (!relation.contains(v))
      setEmpty(relation, v);
  }

  void setDefault(DomainMap &domainMap, V v) const {
    setDefault(domainMap.relation, v);
  }

  /// Given entry v |-> existing, updates domainMap so that
  /// v |-> (set lub existing), and sets the domainMap.changed to true
  /// if the new value is different from the old.
  void leastUpperBoundEntries(DomainMap &domainMap, V v, const Set &set) const {
    auto itr = domainMap.relation.find(v);
    LLVM_DEBUG({
      if (itr == domainMap.relation.end()) {
        llvm::dbgs() << "leastUpperBoundEntries: no entry for ";
        index->printV(llvm::dbgs(), v);
        llvm::dbgs() << "\n";
      }
    });
    assert(itr != domainMap.relation.end());

    LLVM_DEBUG({
      llvm::dbgs() << "leastUpperBoundEntries(";
      index->printV(llvm::dbgs(), v);
      llvm::dbgs() << ", ";
      index->printVs(llvm::dbgs(), index->setToVector(set));
      llvm::dbgs() << ") with existing ";
      index->printVs(llvm::dbgs(), index->setToVector(itr->second));
      llvm::dbgs() << "\n";
    });

    size_t before = itr->second.count();
    domainMap.leastUpperBound(itr->second, set);
    size_t after = itr->second.count();
    if (after != before) {
      LLVM_DEBUG(llvm::dbgs() << "^^^ changed!\n");
      domainMap.changed = true;
    }
  }

  /// If using the relation to represent a predicate, includes v in the
  /// predicate.
  void setTrue(Relation &relation, V v) const {
    setEntries(relation, v, index->unitSet(index->distinguished));
  }

  void setTrue(DomainMap &domainMap, V v) const {
    setTrue(domainMap.relation, v);
  }

  /// Add (lhs, rhs) to relation.
  void add(Relation &relation, V lhs, V rhs) const {
    auto itr = relation.find(lhs);
    LLVM_DEBUG({
      if (itr == relation.end()) {
        llvm::dbgs() << "add: no entry for ";
        index->printV(llvm::dbgs(), lhs);
        llvm::dbgs() << "\n";
      }
    });
    assert(itr != relation.end());
    index->addToSet(itr->second, rhs);
  }

  /// Adds { (lhs, rhs) | rhs in set } to relation.
  void addAll(Relation &relation, V lhs, const Set &set) const {
    auto itr = relation.find(lhs);
    LLVM_DEBUG({
      if (itr == relation.end()) {
        llvm::dbgs() << "addAll: no entry for ";
        index->printV(llvm::dbgs(), lhs);
        llvm::dbgs() << "\n";
      }
    });
    assert(itr != relation.end());
    itr->second |= set;
  }

  /// Updates equivalence such that if (x, y) is in relation then x and y are
  /// considered equivalent. In other words, makes the equivalence represent
  /// relation closed under reflexivity, symmetry, and transitivity.
  void addRelationToEquivalence(Equivalence &equivalence,
                                const Relation &relation) const {
    for (const auto &[lhs, set] : relation) {
      for (auto rhs : index->setToVector(set))
        index->setEquivalent(equivalence, lhs, rhs);
    }
  }

  /// Returns inverse of relation.
  [[nodiscard]] Relation inverse(const Relation &relation) const {
    Relation result = emptyRelation();
    for (const auto &[lhs, set] : relation) {
      size_t lhsIndex = index->getIndex(lhs);
      for (auto rhsItr = set.set_bits_begin(), e = set.set_bits_end();
           rhsItr != e; ++rhsItr)
        result[index->getV(*rhsItr)].set(lhsIndex);
    }
    return result;
  }

  /// Updates relation to include union with other.
  void unionInPlace(Relation &relation, const Relation &other) const {
    for (const auto &[lhs, set] : other)
      addAll(relation, lhs, set);
  }

  /// Adds to set all Vs which are isTrue in relation.
  void addToPredicate(Set &set, const Relation &relation) const {
    for (V v : index->getAllVs()) {
      if (v == index->distinguished)
        continue;
      if (isTrue(relation, v))
        index->addToSet(set, v);
    }
  }

  /// Pretty-prints equivalence, using the stable form for canonical values.
  /// Singleton entries are not printed since implied. For testing only.
  void printEquivalence(llvm::raw_ostream &os, StringRef equivName,
                        const Equivalence &equivalence) const {
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

  /// Pretty-prints relation in index order. For testing only.
  void printRelation(llvm::raw_ostream &os, StringRef relName,
                     const Relation &relation) const {
    os << relName << " begin\n";
    for (V v : index->getAllVs()) {
      auto itr = relation.find(v);
      if (itr == relation.end() || itr->second.count() == 0)
        continue;
      os << "  ";
      index->printV(os, v);
      os << " |-> ";
      index->printVs(os, index->setToVector(itr->second));
      os << "\n";
    }
    os << relName << " end\n";
  }

  /// Pretty-prints predicate (ie set) in index order. For testing only.
  void printPredicate(llvm::raw_ostream &os, StringRef predName,
                      const Set &set) const {
    os << predName << "(";
    index->printVs(os, index->setToVector(set));
    os << ")\n";
  }

  /// Grows domainMap for the given operation, calling into helpers where
  /// appropriate.
  using FlowOp = std::function<void(Operation *, DomainMap &)>;

#undef DEBUG_TYPE
};

//===----------------------------------------------------------------------===//
// ForwardsRelationDataflowAnalysis
//===----------------------------------------------------------------------===//

/// Class to supply control flow transfer functions for all forwards dataflow
/// analyses for relations over values.
class ForwardsRelationDataflowAnalysis
    : public RelationDataflowAnalysisBase<Value> {
public:
  ForwardsRelationDataflowAnalysis(IndexSet<Value> *index)
      : RelationDataflowAnalysisBase<Value>(index) {}

  /// Grows domainMap for one iteration of the body sub-block of cfOp, which
  /// must have one sub-block. Sets changed to true if domainMap was changed.
  void flowSimpleLoop(GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
                      const FlowOp &flowOp);

  /// Grows domainMap for executing the condition THEN body sub-block of cfOp,
  /// which must have two sub-blocks. Sets changed to true if domainMap was
  /// changed.
  void flowConditionedLoop(GRT::GRTControlFlowInterface cfOp,
                           DomainMap &domainMap, const FlowOp &flowOp);

  /// Returns partition of set as (within, without) such that %x in within
  /// iff %x is bound by an op in block or one of its sub-blocks.
  std::pair<Set, Set> splitByScope(const Set &set, Block *block);

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

/// Class to supply control flow transfer functions for all backwards dataflow
/// analyses for relations over operands.
class BackwardsRelationDataflowAnalysis
    : public RelationDataflowAnalysisBase<const OpOperand *> {
public:
  BackwardsRelationDataflowAnalysis(IndexSet<const OpOperand *> *index)
      : RelationDataflowAnalysisBase<const OpOperand *>(index) {}

  /// Returns the combination of all the operand occurrences of value
  /// (possibly none) and the top set.
  Set operandCombination(const DomainMap &domainMap, Value value) const;

  /// Grows domainMap for one iteration of the body sub-block of cfOp, which
  /// must have one sub-block. Sets changed to true if domainMap was changed.
  void flowSimpleLoop(GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
                      const FlowOp &flowOp);

  /// Grows domainMap for executing the condition ThEN body sub-block of cfOp,
  /// which must have two sub-blocks. Sets changed to true if domainMap was
  /// changed.
  void flowConditionedLoop(GRT::GRTControlFlowInterface cfOp,
                           DomainMap &domainMap, const FlowOp &flowOp);

  /// Returns partition of set as (within, without) such that operand using %x
  /// is in within iff %x is bound by an op in block or one of its sub-blocks.
  std::pair<Set, Set> splitByScope(const Set &set, Block *block);

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
