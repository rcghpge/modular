//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "Support/AssertStream.h"

namespace M::MGP {

struct DependencyTreeNode {
  Operation *op;
  DependencyTreeNode *parent{nullptr};
  SmallVector<DependencyTreeNode *> children;

  int preIndex{-1};
  int postIndex{-1};
  int depth{-1};

  inline void addChild(DependencyTreeNode *child) {
    children.push_back(child);
    child->parent = this;
  }

  explicit DependencyTreeNode(Operation *operation) : op(operation) {}
};

/**
 * @class DependencyTree
 * @brief Represents a dependency tree for a block of operations.
 *
 * The DependencyTree class computes and represents the dependency
 * relationships between operations in a given block, forming a tree that allows
 * for efficient queries about execution order guarantees between operations.
 * Each node represents an operation and its ancestors in the tree are the
 * operations that depend on it, meaning that they are guaranteed to execute
 * afterward.
 *
 * Construction complexity:
 * The tree is built using an iterative dataflow analysis algorithm. In the
 * worst case, this algorithm has a time complexity of O(N^2).
 *
 * Usage notes:
 * - The tree is constructed statically and does not update with changes to the
 * block
 * - Dominance queries are performed in constant time
 * - LCA computation between 2 nodes is done in O(h) time where h is the height
 * of the tree
 * - LCA computation among a list of Q nodes is done in O(Q*h) time
 */
class DependencyTree {
public:
  DependencyTree();
  DependencyTree(const DependencyTree &other);
  DependencyTree &operator=(const DependencyTree &other);
  DependencyTree(DependencyTree &&other) noexcept;
  DependencyTree &operator=(DependencyTree &&other) noexcept;

  // Builds the dependency tree for the given block starting with the block
  // terminator
  static DependencyTree getDependencyTree(Block *block);

  // Dumps the tree as a GraphViz dot file, with dashed red edges
  // indicating uses of allocs
  void dumpToDot(llvm::raw_ostream &os) const;

  // Return the DependencyTreeNode* corresponding to the given Operation*,
  // and nullptr if it doesn't exist in the tree.
  DependencyTreeNode *get(Operation *) const;

  Block *getBlock() const;

  // Return true if the tree contains the node corresponding to the Operation*
  bool contains(Operation *) const;

  // Returns true if a is guaranteed to execute after b
  bool executesAfter(DependencyTreeNode *a, DependencyTreeNode *b) const;

  // Find the least common ancestor of two nodes in the tree.
  DependencyTreeNode *findLCA(DependencyTreeNode *a,
                              DependencyTreeNode *b) const;

  // Find the LCA of a list of nodes by repeatedly applying the above function
  DependencyTreeNode *
  findLCA(const SmallVector<DependencyTreeNode *> &nodes) const;
  DependencyTreeNode *findLCA(const SmallVector<Operation *> &nodes) const;

private:
  DependencyTreeNode *root{nullptr};
  DenseMap<Operation *, std::unique_ptr<DependencyTreeNode>> nodes;

  // Clones a node including its children
  DependencyTreeNode *cloneNode(DependencyTreeNode *node,
                                DependencyTreeNode *parent);

  // Return the corresponding DependencyTreeNode* if it exists; otherwise,
  // create a new node and return it
  DependencyTreeNode *getOrCreate(Operation *);
};

// Return a list of alloc uses including uses of aliases. If aliases cannot be
// tracked, return an error.
ErrorOr<SmallVector<Operation *>>
getAllocUses(MGP::BufferAllocDynamicOp allocOp);

// Analysis pass to do dataflow analysis on buffer uses to assist
// with memory planning
struct BufferLifetimeAnalysisPass {
  BufferLifetimeAnalysisPass(Operation *op);

  DenseMap<Block *, DependencyTree> dominatorTrees;
  DependencyTree &getDepTree(Block *b) {
    if (!dominatorTrees.contains(b))
      dominatorTrees.try_emplace(b, DependencyTree::getDependencyTree(b));
    return dominatorTrees[b];
  }
};

} // namespace M::MGP
