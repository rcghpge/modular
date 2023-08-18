//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPTRANSFORMS_VALUEINDEX_H
#define GENERICML_MGPTRANSFORMS_VALUEINDEX_H

#include "GenericML/MGPDialect/MGPOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/raw_ostream.h"

namespace M::MGP {

/// A bijection from all Values, OpOperands or Operations in an MGP::ModelOp
/// to unique indexes. Supports using a BitVector to represent sets of such,
/// and an EquivalenceRelation over indexes to represent equivalence relations
/// over such.
///
/// Also supports pretty-printing values/operands/operations, where SSA names
/// are globally disambiguated by a block index. Blocks indexes are assigned
/// by operation order and pre-order.
template <typename V>
class IndexSet {
public:
  /// The model.
  ModelOp model;

  /// A distinguished V not equal to any V which appears in the model.
  /// This can be used to encode predicates as relations, where (x, {})
  /// represents false and (x, {distinguished}) true. We just use null.
  /// The distinguished V always has value index 0.
  V distinguished = nullptr;

  /// Returns the number of Vs in model, plus one for the distinguished
  /// value.
  size_t size() const { return allVs.size(); }

  bool contains(V v) const { return vIndexMap.contains(v); }

  /// Returns the unique index of V in model, or the distinguished value.
  size_t getIndex(V v) const {
    auto itr = vIndexMap.find(v);
    assert(itr != vIndexMap.end());
    return itr->second;
  }

  /// Returns the V from model for unique index, or the distinguished value.
  V getV(size_t index) const {
    assert(index < allVs.size());
    return allVs[index];
  }

  /// Returns all the Vs in model. See also getAlLCanonicalVs below.
  ArrayRef<V> getAllVs() const { return allVs; }

  //===--------------------------------------------------------------------===//
  // Pretty-printing
  //===--------------------------------------------------------------------===//

  /// Prints a V. For testing only.
  void printV(llvm::raw_ostream &os, V v) { printVImpl(*this, os, v); }

  /// Prints Vs. For testing only.
  void printVs(llvm::raw_ostream &os, ArrayRef<V> vs) {
    os << "{";
    llvm::interleaveComma(vs, os, [&](V v) { printV(os, v); });
    os << "}";
  }

  /// Prints the model in raw form so that the raw SSA names will match up
  /// with those used by printV.
  void printModel(llvm::raw_ostream &os) {
    model.print(os, mlir::OpPrintingFlags().printGenericOpForm());
    os << "\n\n";
  }

  //===--------------------------------------------------------------------===//
  // Representing sets of Vs using bitvectors.
  //===--------------------------------------------------------------------===//

  using Set = llvm::BitVector;

  /// Add v to set.
  void addToSet(Set &set, V v) const { set.set(getIndex(v)); }

  /// Remove v from set.
  void removeFromSet(Set &set, V v) const { set.reset(getIndex(v)); }

  /// Return true if v is in set.
  bool setContains(const Set &set, V v) const { return set.test(getIndex(v)); }

  /// Returns empty set.
  Set emptySet() const { return Set(allVs.size()); }

  /// Returns the set containing all Vs.
  Set fullSet() const { return Set(allVs.size(), true); }

  /// Return set containing only v.
  Set unitSet(V v) const {
    Set set(allVs.size());
    set.set(getIndex(v));
    return set;
  }

  /// Returns set as vector of Vs. Entries will always be in index order.
  SmallVector<V> setToVector(const Set &set) const {
    SmallVector<V> result;
    result.reserve(set.count());
    for (auto itr = set.set_bits_begin(), e = set.set_bits_end(); itr != e;
         ++itr)
      result.emplace_back(allVs[*itr]);
    return result;
  }

  //===--------------------------------------------------------------------===//
  // Representing equivalence relations over Vs using indexes.
  //===--------------------------------------------------------------------===//

  using Equivalence = llvm::EquivalenceClasses<size_t>;

  /// Returns the identity equivalence over all Vs in model, plus the
  /// distinguished value.
  Equivalence identityEquivalence() const {
    Equivalence result;
    for (size_t i = 0; i < allVs.size(); ++i)
      result.insert(i);
    return result;
  }

  /// Returns true if lhs and rhs are equivalent.
  bool areEquivalent(const Equivalence &equivalence, V lhs, V rhs) const {
    return equivalence.isEquivalent(getIndex(lhs), getIndex(rhs));
  }

  /// Returns the canonical representation for V w.r.t. equivalence for use
  /// during analysis.
  ///
  /// CAUTION: Not stable, use getStableCanonical when printing for unit tests.
  V getCanonical(const Equivalence &equivalence, V v) const {
    auto itr = equivalence.findLeader(getIndex(v));
    assert(itr != equivalence.member_end());
    return getV(*itr);
  }

  /// Returns the canonical representation for V w.r.t. equivalence for use
  /// by unit test printing when a stable result is required. This is simply
  /// the V in the equivalence class for v with the lowest index.
  ///
  /// CAUTION: Slow, use getCanonical during analysis.
  V getStableCanonical(const Equivalence &equivalence, V v) const {
    auto itr = equivalence.findLeader(getIndex(v));
    assert(itr != equivalence.member_end());
    size_t index = *itr;
    ++itr;
    while (itr != equivalence.member_end()) {
      index = std::min(index, *itr);
      ++itr;
    }
    return getV(index);
  }

  /// Update equivalence to reflect that lhs and rhs are equivalent Vs.
  void setEquivalent(Equivalence &equivalence, V lhs, V rhs) const {
    equivalence.unionSets(getIndex(lhs), getIndex(rhs));
  }

  /// Returns the set of Vs which are equivalent to v under equivalence,
  /// including v itself.
  Set getEquivalenceClass(const Equivalence &equivalence, V v) const {
    Set result = emptySet();
    for (auto itr = equivalence.findLeader(getIndex(v));
         itr != equivalence.member_end(); ++itr) {
      result.set(*itr);
    }
    return result;
  }

  /// Returns the set of all canonical Vs in model under equivalence. See also
  /// getAlLVs above.
  ///
  /// CAUTION: Not stable.
  Set getAllCanonicalVs(const Equivalence &equivalence) const {
    Set result = emptySet();
    for (auto itr = equivalence.begin(), e = equivalence.end(); itr != e;
         ++itr) {
      if (itr->getData() == 0)
        // Ignore distinguished value.
        continue;
      if (itr->isLeader())
        result.set(itr->getData());
    }
    return result;
  }

  /// Returns copy of set closed under the given equivalence.
  [[nodiscard]] Set closeSetUnderEquivalence(const Equivalence &equivalence,
                                             const Set &set) const {
    Set result(set.size());
    for (auto setItr = set.set_bits_begin(), e = set.set_bits_end();
         setItr != e; ++setItr) {
      for (auto equivItr = equivalence.findLeader(*setItr);
           equivItr != equivalence.member_end(); ++equivItr)
        result.set(*equivItr);
    }
    return result;
  }

  /// Returns copy of set using only canonical Vs w.r.t. equivalance.
  [[nodiscard]] Set toCanonicalForm(const Equivalence &equivalence,
                                    const Set &set) const {
    Set result(set.size());
    for (auto itr = set.set_bits_begin(), e = set.set_bits_end(); itr != e;
         ++itr)
      result.set(equivalence.getLeaderValue(*itr));
    return result;
  }

protected:
  IndexSet(ModelOp model) : model(model) { addV(distinguished); }

  /// Add v to index.
  void addV(V v) {
    size_t index = vIndexMap.size();
    vIndexMap.insert({v, index});
    allVs.emplace_back(v);
  };

  /// Maps each V to its value index (for bit vector representations).
  DenseMap<V, size_t> vIndexMap;

  /// All the Vs in model in index order.
  SmallVector<V> allVs;

  /// Maps each block to its block index (for printing).
  DenseMap<Block *, size_t> blockIndexMap;

  friend void visitValues(IndexSet<Value> &indexSet, Block *block);
  friend void visitOperands(IndexSet<const OpOperand *> &indexSet,
                            Block *block);
  friend void visitOperations(IndexSet<Operation *> &indexSet, Block *block);

  friend void printVImpl(IndexSet<Value> &indexSet, llvm::raw_ostream &os,
                         Value value);
  friend void printVImpl(IndexSet<const OpOperand *> &indexSet,
                         llvm::raw_ostream &os, const OpOperand *value);
  friend void printVImpl(IndexSet<Operation *> &indexSet, llvm::raw_ostream &os,
                         Operation *op);
  friend IndexSet<Value> createValueIndexSet(ModelOp model);
  friend IndexSet<const OpOperand *> createOperandIndexSet(ModelOp model);
  friend IndexSet<Operation *> createOperationIndexSet(ModelOp model);
};

/// Visits all the values in block and adds them to indexSet.
void visitValues(IndexSet<Value> &indexSet, Block *block);

/// Prints a debug representation of value to os, using indexSet for
/// disambiguation.
///  %x    is %x bound in the 'execute' main block
///  %x[3] is %x bound in the block with index 3
void printVImpl(IndexSet<Value> &indexSet, llvm::raw_ostream &os, Value value);

/// Returns an IndexSet<Value> for model.
IndexSet<Value> createValueIndexSet(ModelOp model);

/// Visits all the operands in block and adds them to indexSet.
void visitOperands(IndexSet<const OpOperand *> &indexSet, Block *block);

/// Prints a debug representation of operand to os, using indexSet for
/// disambiguation.
///   %x      is the unique use of %x in the 'execute' main block
///   %x[3,2] is the second use of %x in the block with index 3
void printVImpl(IndexSet<const OpOperand *> &indexSet, llvm::raw_ostream &os,
                const OpOperand *operand);

/// Returns an IndexSet<const OpOperand*> for model.
IndexSet<const OpOperand *> createOperandIndexSet(ModelOp model);

/// Visits all the operations in block and adds them to indexSet.
void visitOperations(IndexSet<Operation *> &indexSet, Block *block);

/// Prints a debug representation of operation to os, using indexSet for
/// disambiguation. We just print the operation's first result, using the
/// same convention as for Values above. Zero-result operations print as [].
void printVImpl(IndexSet<Operation *> &indexSet, llvm::raw_ostream &os,
                Operation *op);

/// Returns an IndexSet<Operation*> for model.
IndexSet<Operation *> createOperationIndexSet(ModelOp model);

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_VALUEINDEX_H
