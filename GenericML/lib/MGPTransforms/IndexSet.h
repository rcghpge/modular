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

/// A bijection from all Values or OpOperands in an MGP::ModelOp to unique
/// indexes. Supports using a BitVector to represent sets of such, and an
/// EquivalenceRelation over indexes to represent equivalence relations over
/// such.
///
/// Also supports pretty-printing values/operands using their SSA name plus a
/// block index to disambiguate identically named values from different blocks.
/// Blocks indexes are allocated by operation order and pre-order.
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

  //===--------------------------------------------------------------------===//
  // Pretty-printing
  //===--------------------------------------------------------------------===//

  /// Prints the model in raw form so that SSA names and nested block arguments
  /// will match up with the printed forms for printV, printPredicate and
  /// printRelation below (modulo block index disambiguation). For testing only.
  void printModel(llvm::raw_ostream &os) {
    model.print(os, mlir::OpPrintingFlags().printGenericOpForm());
    os << "\n\n";
  }

  /// Prints the result of predicate for every value in model. For testing only.
  void printPredicate(llvm::raw_ostream &os, StringRef predName,
                      const std::function<bool(V)> &pred) {
    // Skip the distinguished value.
    for (size_t i = 1, n = allVs.size(); i < n; ++i) {
      V v = allVs[i];
      os << predName << "(";
      printV(*this, os, v);
      os << ") = " << (pred(v) ? "true" : "false") << "\n";
    }
  }

  /// Prints the result of relation for every value in model. For testing only.
  void printRelation(llvm::raw_ostream &os, StringRef relName,
                     const std::function<bool(Value, Value)> &rel) {
    // Skip the distinguished value.
    for (size_t i = 1, n = allVs.size(); i < n; ++i) {
      for (size_t j = 1; j < n; ++j) {
        V lhs = allVs[i];
        V rhs = allVs[j];
        os << relName << "(";
        printV(*this, os, lhs);
        os << ", ";
        printV(*this, os, rhs);
        os << ") = " << (rel(lhs, rhs) ? "true" : "false") << "\n";
      }
    }
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

  /// Returns set as vector of Vs.
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

  /// Returns the canonical representation for V w.r.t. equivalence.
  V getCanonical(const Equivalence &equivalence, V v) const {
    auto itr = equivalence.findLeader(getIndex(v));
    assert(itr != equivalence.member_end());
    return getV(*itr);
  }

  /// Update equivalence to reflect that lhs and rhs are equivalent Vs.
  void setEquivalent(Equivalence &equivalence, V lhs, V rhs) const {
    equivalence.unionSets(getIndex(lhs), getIndex(rhs));
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
  friend void printV(IndexSet<Value> &indexSet, llvm::raw_ostream &os,
                     Value value);
  friend void printV(IndexSet<const OpOperand *> &indexSet,
                     llvm::raw_ostream &os, const OpOperand *value);
  friend IndexSet<Value> createValueIndexSet(ModelOp model);
  friend IndexSet<const OpOperand *> createOperandIndexSet(ModelOp model);
};

/// Visits all the values in block and adds them to indexSet.
void visitValues(IndexSet<Value> &indexSet, Block *block);

/// Prints a debug representation of value to os, using indexSet for
/// disambiguation.
void printV(IndexSet<Value> &indexSet, llvm::raw_ostream &os, Value value);

/// Returns an IndexSet<Value> for model.
IndexSet<Value> createValueIndexSet(ModelOp model);

/// Visits all the operands in block and adds them to indexSet.
void visitOperands(IndexSet<const OpOperand *> &indexSet, Block *block);

/// Prints a debug representation of operand to os, using indexSet for
/// disambiguation.
void printV(IndexSet<const OpOperand *> &indexSet, llvm::raw_ostream &os,
            Value value);

/// Returns an IndexSet<const OpOperand*> for model.
IndexSet<const OpOperand *> createOperandIndexSet(ModelOp model);

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_VALUEINDEX_H
