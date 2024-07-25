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

DependencyTree::DependencyTree() = default;

DependencyTree::DependencyTree(const DependencyTree &other) {
  root = cloneNode(other.root, nullptr);
}

DependencyTree &DependencyTree::operator=(const DependencyTree &other) {
  if (this != &other) {
    DependencyTree temp(other);
    std::swap(root, temp.root);
    std::swap(nodes, temp.nodes);
  }
  return *this;
}

DependencyTree::DependencyTree(DependencyTree &&other) noexcept
    : root(other.root), nodes(std::move(other.nodes)) {
  other.root = nullptr;
}

DependencyTree &DependencyTree::operator=(DependencyTree &&other) noexcept {
  if (this != &other) {
    root = other.root;
    nodes = std::move(other.nodes);
    other.root = nullptr;
  }
  return *this;
}

DependencyTreeNode *DependencyTree::cloneNode(DependencyTreeNode *node,
                                              DependencyTreeNode *parent) {
  if (!node)
    return nullptr;

  auto newNode = std::make_unique<DependencyTreeNode>(node->op);
  newNode->preIndex = node->preIndex;
  newNode->postIndex = node->postIndex;
  newNode->depth = node->depth;
  newNode->parent = parent;

  DependencyTreeNode *newNodePtr = newNode.get();
  nodes[node->op] = std::move(newNode);

  for (auto *child : node->children) {
    newNodePtr->children.push_back(cloneNode(child, newNodePtr));
  }

  return newNodePtr;
}
DependencyTree DependencyTree::getDependencyTree(Block *block) {
  // Build the dependency tree starting with block terminator
  // Does an iterative dataflow analysis for computing `runsAfter` sets.

  DependencyTree tree;

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

  // Initialize runsAfter sets
  SmallVector<llvm::BitVector> runsAfter(numOperations,
                                         llvm::BitVector(numOperations, false));

  llvm::BitVector noUserRunsAfter(numOperations);
  noUserRunsAfter.set(operationToIndex[terminator]);

  // Initialize terminator runsAfter set with only itself
  runsAfter[operationToIndex[terminator]] = noUserRunsAfter;

  // Allocate outside loop to prevent re-allocation
  llvm::SetVector<Operation *> worklist;
  llvm::BitVector newRunsAfter(numOperations);
  SmallVector<Operation *, 4> users;

  for (Operation &op : block->getOperations()) {
    worklist.insert(&op);
  }

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    unsigned opIndex = operationToIndex[op];

    auto &oldRunsAfter = runsAfter[opIndex];
    newRunsAfter = oldRunsAfter;

    // Find all operations in this block that use this op's results
    users.clear();
    for (auto result : op->getResults())
      for (auto &use : result.getUses())
        if (use.getOwner()->getBlock() == block)
          users.push_back(use.getOwner());

    // If there are no users in this block, this op's runsAfter set only
    // consists of the block terminator
    if (users.empty()) {
      newRunsAfter = noUserRunsAfter;
      newRunsAfter.set(opIndex);
    } else {
      // Otherwise, this op's runsAfter set is the union of all of its uses'
      // sets
      for (Operation *user : users)
        newRunsAfter |= runsAfter[operationToIndex[user]];

      newRunsAfter.set(opIndex); // Ensure op is in its own runsAfter set
    }

    // Update the real runsAfter set if modified
    if (oldRunsAfter != newRunsAfter) {
      oldRunsAfter = std::move(newRunsAfter);

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

  // Find the immediate runsAfter for each op
  DenseMap<Operation *, Operation *> immediateRunsAfters;
  immediateRunsAfters.reserve(block->getOperations().size() - 1);

  for (Operation &n : block->getOperations()) {
    if (&n == terminator)
      continue;

    const auto &nRunsAfter = runsAfter[operationToIndex[&n]];
    Operation *immediateRunsAfter = nullptr;
    unsigned ipdomIndex = numOperations;

    for (unsigned i = 0; i < numOperations; ++i) {
      Operation *op = indexToOperation[i];

      // Skip all operations that aren't in the runsAfter set
      if (op == &n || !nRunsAfter.test(i))
        continue;

      // The immediate runsAfter is the one with the largest runsAfter set
      if (immediateRunsAfter == nullptr ||
          runsAfter[i].count() > runsAfter[ipdomIndex].count()) {
        immediateRunsAfter = op;
        ipdomIndex = i;
      }
    }

    assert(immediateRunsAfter != nullptr);
    immediateRunsAfters[&n] = immediateRunsAfter;
  }

  // Create the root node (terminator)
  tree.root = tree.getOrCreate(terminator);

  // Build the tree based on immediateRunsAfters
  for (Operation &n : block->getOperations()) {
    if (&n == terminator)
      continue;
    auto parent = immediateRunsAfters.at(&n);
    assert(parent);
    auto parentNode = tree.getOrCreate(parent);
    auto childNode = tree.getOrCreate(&n);
    parentNode->addChild(childNode);
    assert(tree.contains(&n));
  }

  // Do a pre and postorder traversal and set indices on nodes for constant time
  // domination lookup
  using VisitState = std::tuple<DependencyTreeNode *, bool, int>;
  SmallVector<VisitState> stack;
  DenseSet<DependencyTreeNode *> visited;

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

void DependencyTree::dumpToDot(llvm::raw_ostream &os) const {
  static unsigned int maxNodeNameLength = 20;

  os << "digraph DependencyTree {\n";
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
  std::function<void(DependencyTreeNode *)> dumpNode =
      [&](DependencyTreeNode *node) {
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

DependencyTreeNode *DependencyTree::get(Operation *op) const {
  auto it = nodes.find(op);
  if (it == nodes.end())
    return nullptr;
  return it->second.get();
}

Block *DependencyTree::getBlock() const {
  assert(root && root->op);
  return root->op->getBlock();
}

DependencyTreeNode *DependencyTree::getOrCreate(Operation *op) {
  auto it = nodes.find(op);
  if (it != nodes.end()) {
    return it->second.get();
  } else {
    auto newNode = std::make_unique<DependencyTreeNode>(op);
    nodes[op] = std::move(newNode);
    return nodes[op].get();
  }
}

bool DependencyTree::contains(Operation *op) const {
  return nodes.contains(op);
}

bool DependencyTree::executesAfter(DependencyTreeNode *a,
                                   DependencyTreeNode *b) const {
  assert(a && b);
  return a->preIndex <= b->preIndex && a->postIndex >= b->postIndex;
}

DependencyTreeNode *DependencyTree::findLCA(DependencyTreeNode *a,
                                            DependencyTreeNode *b) const {
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

DependencyTreeNode *
DependencyTree::findLCA(const SmallVector<DependencyTreeNode *> &nodes) const {
  assert(!nodes.empty());

  DependencyTreeNode *lca = nodes[0];
  for (size_t i = 1; i < nodes.size(); ++i) {
    lca = findLCA(lca, nodes[i]);
    if (!lca)
      return nullptr;
  }
  return lca;
}

DependencyTreeNode *
DependencyTree::findLCA(const SmallVector<Operation *> &nodes) const {
  assert(!nodes.empty());

  DependencyTreeNode *lca = get(nodes[0]);
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
    dominatorTrees[block] = DependencyTree::getDependencyTree(block);
  });
}
