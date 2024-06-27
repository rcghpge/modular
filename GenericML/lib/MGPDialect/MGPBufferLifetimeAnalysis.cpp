//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPBufferLifetimeAnalysis.h"
#include "GraphRT/GRTDialect/GRTControlFlowInterface.h"
#include "llvm/ADT/SetVector.h"
#include <llvm/ADT/SetOperations.h>

using namespace M;
using namespace MGP;

PostDominatorTree::PostDominatorTree() = default;

PostDominatorTree::PostDominatorTree(const PostDominatorTree &other) {
  root = cloneNode(other.root, nullptr);
}

PostDominatorTree &
PostDominatorTree::operator=(const PostDominatorTree &other) {
  if (this != &other) {
    PostDominatorTree temp(other);
    std::swap(root, temp.root);
    std::swap(nodes, temp.nodes);
  }
  return *this;
}

PostDominatorTree::PostDominatorTree(PostDominatorTree &&other) noexcept
    : root(other.root), nodes(std::move(other.nodes)) {
  other.root = nullptr;
}

PostDominatorTree &
PostDominatorTree::operator=(PostDominatorTree &&other) noexcept {
  if (this != &other) {
    root = other.root;
    nodes = std::move(other.nodes);
    other.root = nullptr;
  }
  return *this;
}

PostDominatorTreeNode *
PostDominatorTree::cloneNode(PostDominatorTreeNode *node,
                             PostDominatorTreeNode *parent) {
  if (!node)
    return nullptr;

  auto newNode = std::make_unique<PostDominatorTreeNode>(node->op);
  newNode->preIndex = node->preIndex;
  newNode->postIndex = node->postIndex;
  newNode->depth = node->depth;
  newNode->parent = parent;

  PostDominatorTreeNode *newNodePtr = newNode.get();
  nodes[node->op] = std::move(newNode);

  for (auto *child : node->children) {
    newNodePtr->children.push_back(cloneNode(child, newNodePtr));
  }

  return newNodePtr;
}
PostDominatorTree PostDominatorTree::getPostDominatorTree(Block *block) {
  // Build dominator tree starting with block terminator
  // Does an iterative dataflow analysis for computing postdominators
  // There are better algorithms for doing this (Lengauer-Tarjan is almost
  // linear, and LLVM's GenericDomTreeConstruction.h uses the Semi-NCA
  // algorithm which is quadratic in the worst case but often faster) but they
  // are more complex to implement. We can revisit this if this pass turns out
  // to have a significant impact on compile time.

  PostDominatorTree tree;

  Operation *terminator = block->getTerminator();

  llvm::DenseMap<Operation *, unsigned> operationToIndex;
  SmallVector<Operation *> indexToOperation;
  unsigned bindex = 0;
  for (Operation &op : block->getOperations()) {
    operationToIndex[&op] = bindex;
    indexToOperation.push_back(&op);
    bindex++;
  }

  unsigned numOperations = operationToIndex.size();

  // Initialize postDom sets
  SmallVector<llvm::BitVector> postDom(numOperations,
                                       llvm::BitVector(numOperations, true));

  llvm::BitVector noUserPostdom(numOperations);
  noUserPostdom.set(operationToIndex[terminator]);

  // Initialize terminator postdom set with only itself
  postDom[operationToIndex[terminator]] = noUserPostdom;

  // Allocate outside loop to prevent re-allocation
  llvm::SetVector<Operation *> worklist;
  llvm::BitVector newPostdom(numOperations);
  SmallVector<Operation *, 4> users;

  for (Operation &op : llvm::reverse(block->getOperations())) {
    worklist.insert(&op);
  }

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    unsigned opIndex = operationToIndex[op];

    auto &oldPostdom = postDom[opIndex];
    newPostdom = oldPostdom;

    // Find all operations in this block that use this op's results
    users.clear();
    for (auto result : op->getResults())
      for (auto &use : result.getUses())
        if (use.getOwner()->getBlock() == block)
          users.push_back(use.getOwner());

    // If there are no users in this block, this op post-dominates only itself
    // and the terminator
    if (users.empty()) {
      newPostdom = noUserPostdom;
      newPostdom.set(opIndex);
    } else {
      // Otherwise, this op postdominates the intersection of the
      // postdominators of its uses
      for (Operation *user : users)
        newPostdom &= postDom[operationToIndex[user]];

      newPostdom.set(opIndex); // Ensure op is in its own postdom set
    }

    // Update the real postDom set if modified
    if (oldPostdom != newPostdom) {
      oldPostdom = std::move(newPostdom);

      // Add operands to the worklist
      for (Value operand : op->getOperands()) {
        if (Operation *definingOp = operand.getDefiningOp()) {
          if (definingOp->getBlock() == block) {
            worklist.insert(definingOp);
          }
        }
      }
    }
  }

  // Find the immediate postdominator for each op
  DenseMap<Operation *, Operation *> immediatePostdoms;
  immediatePostdoms.reserve(block->getOperations().size() - 1);

  for (Operation &n : block->getOperations()) {
    if (&n == terminator)
      continue;

    const auto &nPostDom = postDom[operationToIndex[&n]];
    Operation *immediatePostdom = nullptr;
    unsigned ipdomIndex = numOperations;

    for (unsigned i = 0; i < numOperations; ++i) {
      Operation *op = indexToOperation[i];

      // Skip all operations that aren't strict postdominators of n
      if (op == &n || !nPostDom.test(i))
        continue;

      // The immediate postdominator is the one with the largest postDom set
      if (immediatePostdom == nullptr ||
          postDom[i].count() > postDom[ipdomIndex].count()) {
        immediatePostdom = op;
        ipdomIndex = i;
      }
    }

    assert(immediatePostdom != nullptr);
    immediatePostdoms[&n] = immediatePostdom;
  }

  // Create the root node (terminator)
  tree.root = tree.getOrCreate(terminator);

  // Build the tree based on immediatePostdoms
  for (Operation &n : block->getOperations()) {
    if (&n == terminator)
      continue;
    auto parent = immediatePostdoms.at(&n);
    assert(parent);
    auto parentNode = tree.getOrCreate(parent);
    auto childNode = tree.getOrCreate(&n);
    parentNode->addChild(childNode);
    assert(tree.contains(&n));
  }

  // Do a pre and postorder traversal and set indices on nodes for constant time
  // domination lookup
  using VisitState = std::tuple<PostDominatorTreeNode *, bool, int>;
  SmallVector<VisitState> stack;
  DenseSet<PostDominatorTreeNode *> visited;

  stack.push_back({tree.root, true, 0});
  int index = 0;

  while (!stack.empty()) {
    auto [top, isFirstVisit, depth] = stack.pop_back_val();
    if (!top)
      continue;

    if (isFirstVisit) {
      // preorder
      if (!visited.contains(top)) {
        top->preIndex = index++;
        top->depth = depth;

        // Schedule to visit again for post-order
        stack.push_back({top, false, depth});

        for (auto child : llvm::reverse(top->children)) {
          stack.push_back({child, true, depth + 1});
        }

        visited.insert(top);
      }
    } else {
      // postorder
      if (visited.contains(top) && top->postIndex == -1)
        top->postIndex = index++;
    }
  }

  return tree;
}

void PostDominatorTree::dumpToDot(llvm::raw_ostream &os) const {
  static unsigned int maxNodeNameLength = 20;

  os << "digraph PostDominatorTree {\n";
  os << "  node [shape=ellipse];\n";

  auto format = [](Operation *op) {
    std::string str = llvm::formatv("{0}", *op);
    for (char &c : str)
      if (c == '%' || c == '"')
        c = ' ';

    if (str.size() > maxNodeNameLength)
      str = str.substr(0, maxNodeNameLength);
    return str;
  };

  // Draw main dominator tree
  std::function<void(PostDominatorTreeNode *)> dumpNode =
      [&](PostDominatorTreeNode *node) {
        if (!node)
          return;
        std::string nodeName = format(node->op);
        os << "  \"" << nodeName << "\";\n";
        for (auto *child : node->children) {
          std::string childName = format(child->op);
          os << "  \"" << nodeName << "\" -> \"" << childName << "\";\n";
          dumpNode(child);
        }
      };

  dumpNode(root);

  // Draw alloc use edges
  for (auto &op : root->op->getBlock()->getOperations()) {
    auto alloc = dyn_cast<MGP::BufferAllocDynamicOp>(op);
    if (!alloc)
      continue;
    auto uses = getAllocUses(alloc);
    if (uses.isError())
      continue;
    std::string nodeName = format(alloc);
    for (auto use : *uses) {
      std::string useName = format(use);

      os << "  \"" << nodeName << "\" -> \"" << useName
         << "\" [color=\"red\", constraint=false, style=dotted];\n";
    }
  }

  os << "}\n";
}

PostDominatorTreeNode *PostDominatorTree::get(Operation *op) const {
  auto it = nodes.find(op);
  if (it == nodes.end())
    return nullptr;
  return it->second.get();
}

Block *PostDominatorTree::getBlock() const {
  assert(root && root->op);
  return root->op->getBlock();
}

PostDominatorTreeNode *PostDominatorTree::getOrCreate(Operation *op) {
  auto it = nodes.find(op);
  if (it != nodes.end()) {
    return it->second.get();
  } else {
    auto newNode = std::make_unique<PostDominatorTreeNode>(op);
    nodes[op] = std::move(newNode);
    return nodes[op].get();
  }
}

bool PostDominatorTree::contains(Operation *op) const {
  return nodes.contains(op);
}

bool PostDominatorTree::postdominates(PostDominatorTreeNode *a,
                                      PostDominatorTreeNode *b) const {
  assert(a && b);
  return a->preIndex <= b->preIndex && a->postIndex >= b->postIndex;
}

PostDominatorTreeNode *
PostDominatorTree::findLCA(PostDominatorTreeNode *a,
                           PostDominatorTreeNode *b) const {
  if (!a || !b)
    return nullptr;

  int depthA = a->depth;
  int depthB = b->depth;

  // Bring both nodes to the same depth
  while (depthA > depthB) {
    a = a->parent;
    --depthA;
  }
  while (depthB > depthA) {
    b = b->parent;
    --depthB;
  }

  // Traverse up until we find the common ancestor
  while (a != b) {
    a = a->parent;
    b = b->parent;
  }

  return a;
}

PostDominatorTreeNode *PostDominatorTree::findLCA(
    const SmallVector<PostDominatorTreeNode *> &nodes) const {
  assert(!nodes.empty());

  PostDominatorTreeNode *lca = nodes[0];
  for (size_t i = 1; i < nodes.size(); ++i) {
    lca = findLCA(lca, nodes[i]);
    if (!lca)
      return nullptr;
  }
  return lca;
}

PostDominatorTreeNode *
PostDominatorTree::findLCA(const SmallVector<Operation *> &nodes) const {
  assert(!nodes.empty());

  PostDominatorTreeNode *lca = get(nodes[0]);
  for (size_t i = 1; i < nodes.size(); ++i) {
    lca = findLCA(lca, get(nodes[i]));
    if (!lca)
      return nullptr;
  }
  return lca;
}

ErrorOr<SmallVector<Operation *>>
M::MGP::getAllocUses(MGP::BufferAllocDynamicOp allocOp) {
  // Determine the uses of the allocated buffer by searching through all direct
  // uses as well as those of possible aliases

  SmallVector<Operation *> uses;
  SmallVector<Operation *> stack;

  stack.push_back(allocOp);

  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();

    for (auto &use : top->getUses()) {
      if (!isa<MGP::TensorExtractTensorSpecOp>(*use.getOwner()))
        uses.push_back(use.getOwner());

      const bool isBufferOperand =
          isa<MGP::DeviceBufferType>(use.get().getType());

      if (isa<GRT::GRTControlFlowInterface>(*use.getOwner()))
        return Error("control flow op may alias");

      if (isBufferOperand &&
          isa<GRT::GRTMayAliasInterface, GRT::GRTExtendsLifetimeInterface>(
              *use.getOwner()))
        stack.push_back(use.getOwner());
    }
  }
  return uses;
}

BufferLifetimeAnalysisPass::BufferLifetimeAnalysisPass(Operation *op) {
  auto modelOp = llvm::dyn_cast<ModelOp>(*op);

  ASSERT_STREAM(modelOp,
                "BufferLifetimeAnalysisPass can only run on an MGP::ModelOp");

  modelOp.walk([&](Block *block) {
    dominatorTrees[block] = PostDominatorTree::getPostDominatorTree(block);
  });
}
