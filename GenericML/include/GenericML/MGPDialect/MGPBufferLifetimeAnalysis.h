//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "Support/AssertStream.h"

namespace M::MGP {

struct PostDominatorTreeNode {
  Operation *op;
  PostDominatorTreeNode *parent{nullptr};
  SmallVector<PostDominatorTreeNode *> children;

  int preIndex{-1};
  int postIndex{-1};
  int depth{-1};

  inline void addChild(PostDominatorTreeNode *child) {
    children.push_back(child);
    child->parent = this;
  }

  explicit PostDominatorTreeNode(Operation *operation) : op(operation) {}
};

/**
 * @class PostDominatorTree
 * @brief Represents a postdominator tree for a block of operations.
 *
 * The PostDominatorTree class computes and represents the postdominator
 * relationship between operations in a given block. It analyzes the dataflow
 * through operation result uses, considering an operation to postdominate
 * another if all paths from the latter to the block terminator must pass
 * through the former.
 *
 * MLIR offers an existing `mlir::DominanceInfo` class, but it is only suitable
 * for dominance across basic blocks or within SSACFG regions. In a graph
 * region, like our MGP graphs are, every op in the same block is considered to
 * dominate every other op in the block.
 *
 * LLVM offers an efficient generic implementation of a (post)dominator tree in
 * `llvm/Support/GenericDomTree.h`. However, although it is called generic, it
 * is difficult to adapt for this use case of MLIR op-level dataflow analysis.
 * For example, it requires bidirectional iterators for node children, but MLIR
 * stores uses as linked lists which only have forward iterators.
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
 *
 * This implementation prioritizes simplicity and ease of understanding over
 * optimal performance. More sophisticated algorithms for tree construction like
 * Lengauer-Tarjan or Semi-NCA could be considered in the future, or Tarjan's
 * offline LCA algorithm for computing the LCA of a list of nodes.
 */
class PostDominatorTree {
public:
  PostDominatorTree();
  PostDominatorTree(const PostDominatorTree &other);
  PostDominatorTree &operator=(const PostDominatorTree &other);
  PostDominatorTree(PostDominatorTree &&other) noexcept;
  PostDominatorTree &operator=(PostDominatorTree &&other) noexcept;

  // Builds the postdominator tree for the given block starting with the block
  // terminator
  static PostDominatorTree getPostDominatorTree(Block *block);

  // Dumps the dominator tree as a GraphViz dot file, with dashed red edges
  // indicating uses of allocs
  void dumpToDot(llvm::raw_ostream &os) const;

  // Return the PostDominatorTreeNode* corresponding to the given Operation*,
  // and nullptr if it doesn't exist in the tree.
  PostDominatorTreeNode *get(Operation *) const;

  Block *getBlock() const;

  // Return true if the tree contains the node corresponding to the Operation*
  bool contains(Operation *) const;

  // Returns true if a postdominates b, meaning that all paths to the block
  // terminator from a must pass through b
  bool postdominates(PostDominatorTreeNode *a, PostDominatorTreeNode *b) const;

  // Find the least common ancestor of two nodes in the dominator tree. Can be
  // used to find the nearest common postdominator.
  PostDominatorTreeNode *findLCA(PostDominatorTreeNode *a,
                                 PostDominatorTreeNode *b) const;

  // Find the LCA of a list of nodes by repeatedly applying the above function
  PostDominatorTreeNode *
  findLCA(const SmallVector<PostDominatorTreeNode *> &nodes) const;
  PostDominatorTreeNode *findLCA(const SmallVector<Operation *> &nodes) const;

private:
  PostDominatorTreeNode *root{nullptr};
  DenseMap<Operation *, std::unique_ptr<PostDominatorTreeNode>> nodes;

  // Clones a node including its children
  PostDominatorTreeNode *cloneNode(PostDominatorTreeNode *node,
                                   PostDominatorTreeNode *parent);

  // Return the corresponding PostDominatorTreeNode* if it exists; otherwise,
  // create a new node and return it
  PostDominatorTreeNode *getOrCreate(Operation *);
};

// Return a list of alloc uses including uses of aliases. If aliases cannot be
// tracked, return an error.
ErrorOr<SmallVector<Operation *>>
getAllocUses(MGP::BufferAllocDynamicOp allocOp);

// Analysis pass to do dataflow postdominator analysis on buffer uses to assist
// with memory planning
struct BufferLifetimeAnalysisPass {
  BufferLifetimeAnalysisPass(Operation *op);

  DenseMap<Block *, PostDominatorTree> dominatorTrees;
  PostDominatorTree &getDomTree(Block *b) {
    if (!dominatorTrees.contains(b))
      dominatorTrees.try_emplace(b, PostDominatorTree::getPostDominatorTree(b));
    return dominatorTrees[b];
  }
};

} // namespace M::MGP
