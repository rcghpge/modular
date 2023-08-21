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

    /// If true, interpret 2-subregion kLoop control flow ops as if all
    /// execution traces are of the form:
    ///   (cond_block | body_block)*
    /// This interpretation is useful when the results of the analysis may
    /// cause the type of a loop input, result, sub-block argument or sub-block
    /// result to change. In that case the change can only be made in one place
    /// if it can also be made in all other places, otherwise the resulting
    /// loop would be ill-formed.
    ///
    /// Otherwise, interpret 2-subregion kLoop control flom ops as if all
    /// execution traces are of the form:
    ///   cond_block (body_block cond_block)*
    /// This interpretation is more precise than the above, but does not
    /// guarantee all control flow op input, result, sub-block argument and
    /// sub-block results have the same domain entry.
    bool approximateLoops;

    /// Set to true whenever relation is changed.
    bool changed = false;

    DomainMap(Relation &relation, SetOrientation orientation, Set &&emptySet,
              Set &&fullSet, bool handleDisjointBranches = false,
              bool approximateLoops = false)
        : relation(relation), orientation(orientation),
          handleDisjointBranches(handleDisjointBranches),
          approximateLoops(approximateLoops) {
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

  /// Returns the set bound to v.
  Set getEntries(const DomainMap &domainMap, V v) const {
    return getEntries(domainMap.relation, v);
  }

  /// Returns the set bound to v.
  Set getEntries(const Relation &relation, V v) const {
    auto itr = relation.find(v);
    if (itr == relation.end())
      return index->emptySet();
    else
      return itr->second;
  }

  /// Returns the set bound to v, unioned with v itself.
  Set getEntriesAndId(const DomainMap &domainMap, V v) const {
    return getEntriesAndId(domainMap.relation, v);
  }

  /// Returns the set bound to v, unioned with v itself.
  Set getEntriesAndId(const Relation &relation, V v) const {
    Set set = getEntries(relation, v);
    index->addToSet(set, v);
    return set;
  }

  /// If using the relation to represent a predicate, returns true if v is
  /// in the predicate.
  bool isTrue(const Relation &relation, V v) const {
    auto itr = relation.find(v);
    return itr != relation.end() &&
           itr->second.test(0); // distinguished always has index 0
  }

  /// Returns true if (lhs, rhs) is in relation.
  bool containsEntry(const Relation &relation, V lhs, V rhs) const {
    auto itr = relation.find(lhs);
    return itr != relation.end() && index->setContains(itr->second, rhs);
  }

  /// Updates domainMap so that v |-> set. Marks changed true if domainMap
  /// was changed.
  void setEntries(DomainMap &domainMap, V v, const Set &set) const {
#define DEBUG_TYPE "mgp-dataflow"
    LLVM_DEBUG({
      llvm::dbgs() << "setEntries(";
      index->printV(llvm::dbgs(), v);
      llvm::dbgs() << ", ";
      index->printVs(llvm::dbgs(), index->setToVector(set));
      llvm::dbgs() << ")\n";
    });

    bool setIsEmpty = set.count() == 0;
    auto itr = domainMap.relation.find(v);
    if (itr == domainMap.relation.end()) {
      if (!setIsEmpty) {
        LLVM_DEBUG(llvm::dbgs() << "not equal to existing empty set\n");
        domainMap.relation.insert({v, set});
        domainMap.changed = true;
      }
    } else if (setIsEmpty) {
      LLVM_DEBUG({
        llvm::dbgs() << "not equal to existing set ";
        index->printVs(llvm::dbgs(), index->setToVector(itr->second));
        llvm::dbgs() << "\n";
      });
      domainMap.relation.erase(itr);
      domainMap.changed = true;
    } else if (itr->second != set) {
      LLVM_DEBUG({
        llvm::dbgs() << "not equal to existing set ";
        index->printVs(llvm::dbgs(), index->setToVector(itr->second));
        llvm::dbgs() << "\n";
      });
      itr->second = set;
      domainMap.changed = true;
    }
#undef DEBUG_TYPE
  }

  /// Given v |-> existing, updates domainMap so that v |-> (set lub existing).
  /// Marks changed true if domainMap was changed.
  void leastUpperBoundEntries(DomainMap &domainMap, V v, const Set &set) const {
    Set newSet = set;
    domainMap.leastUpperBound(newSet, getEntries(domainMap, v));
    setEntries(domainMap, v, newSet);
  }

  /// Updates domainMap so that v |-> {}. Marks changed true
  /// if domainMap was changed.
  void setEmpty(DomainMap &domainMap, V v) const {
#define DEBUG_TYPE "mgp-dataflow"
    LLVM_DEBUG({
      llvm::dbgs() << "setEmpty(";
      index->printV(llvm::dbgs(), v);
      llvm::dbgs() << ")\n";
    });
    auto itr = domainMap.relation.find(v);
    if (itr != domainMap.relation.end()) {
      LLVM_DEBUG({
        llvm::dbgs() << "not equal to existing set ";
        index->printVs(llvm::dbgs(), index->setToVector(itr->second));
        llvm::dbgs() << "\n";
      });
      domainMap.relation.erase(itr);
      domainMap.changed = true;
    }
#undef DEBUG_TYPE
  }

  /// If using the relation to represent a predicate, includes v in the
  /// predicate.
  void setTrue(DomainMap &domainMap, V v) const {
    setEntries(domainMap, v, index->unitSet(index->distinguished));
  }

  /// Add (lhs, rhs) to relation.
  void add(Relation &relation, V lhs, V rhs) const {
    auto itr = relation.find(lhs);
    if (itr == relation.end())
      relation.insert({lhs, index->unitSet(rhs)});
    else
      index->addToSet(itr->second, rhs);
  }

  /// Adds { (lhs, rhs) | rhs in set } to relation.
  void addAll(Relation &relation, V lhs, const Set &set) const {
    auto itr = relation.find(lhs);
    if (itr == relation.end()) {
      if (set.count())
        relation.insert({lhs, set});
      return;
    }
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
    Relation result;
    for (const auto &[lhs, set] : relation) {
      for (const auto rhs : index->setToVector(set))
        add(result, rhs, lhs);
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
      if (itr == relation.end())
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

  /// Grows domainMap for executing the condition OR body sub-block of cfOp,
  /// which must have two sub-blocks. Sets changed to true if domainMap was
  /// changed.
  void flowApproximateConditionedLoop(GRT::GRTControlFlowInterface cfOp,
                                      DomainMap &domainMap,
                                      const FlowOp &flowOp);

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

  /// Grows domainMap for executing the condition OR body sub-block of cfOp,
  /// which must have two sub-blocks. Sets changed to true if domainMap was
  /// changed.
  void flowApproximateConditionedLoop(GRT::GRTControlFlowInterface cfOp,
                                      DomainMap &domainMap,
                                      const FlowOp &flowOp);

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
