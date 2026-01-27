//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPBufferLifetimeAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.h"
#include "Support/Compiler/Threading.h"

#include "llvm/ADT/SparseBitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace MGP;

DependencyTree::DependencyTree() = default;

DependencyTree::DependencyTree(const DependencyTree &other) {
  // First create the root and all nodes
  root = cloneNode(other.root, nullptr);

  // Copy the remaining members
  runsAfter = other.runsAfter;
  runsAfterCounts = other.runsAfterCounts;
  operationToIndex = other.operationToIndex;
  indexToOperation = other.indexToOperation;
}

DependencyTree &DependencyTree::operator=(const DependencyTree &other) {
  if (this != &other) {
    DependencyTree temp(other);
    std::swap(root, temp.root);
    std::swap(nodes, temp.nodes);
    std::swap(runsAfter, temp.runsAfter);
    std::swap(runsAfterCounts, temp.runsAfterCounts);
    std::swap(operationToIndex, temp.operationToIndex);
    std::swap(indexToOperation, temp.indexToOperation);
  }
  return *this;
}

DependencyTree::DependencyTree(DependencyTree &&other) noexcept
    : root(other.root), nodes(std::move(other.nodes)),
      runsAfter(std::move(other.runsAfter)),
      runsAfterCounts(std::move(other.runsAfterCounts)),
      operationToIndex(std::move(other.operationToIndex)),
      indexToOperation(std::move(other.indexToOperation)) {
  other.root = nullptr;
}

DependencyTree &DependencyTree::operator=(DependencyTree &&other) noexcept {
  if (this != &other) {
    root = other.root;
    nodes = std::move(other.nodes);
    runsAfter = std::move(other.runsAfter);
    runsAfterCounts = std::move(other.runsAfterCounts);
    operationToIndex = std::move(other.operationToIndex);
    indexToOperation = std::move(other.indexToOperation);
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

namespace {
static void buildRunsAfter(
    Block *block,
    llvm::DenseMap<Operation *, llvm::SparseBitVector<64>> &runsAfter,
    llvm::DenseMap<Operation *, unsigned> &opIndex,
    llvm::SmallVectorImpl<Operation *> &indexToOp) {

  // Collect all operations in block order first
  llvm::SmallVector<Operation *> blockOrderOps;
  for (Operation &op : block->getOperations()) {
    blockOrderOps.push_back(&op);
  }

  const unsigned numOps = blockOrderOps.size();
  llvm::DenseSet<Operation *> visited;
  llvm::DenseSet<Operation *> tempMark;
  llvm::SmallVector<Operation *> topoOrder;

  std::function<void(Operation *)> visit = [&](Operation *op) {
    if (visited.count(op))
      return;
    if (tempMark.count(op)) {
      llvm::report_fatal_error("Cycle detected in graph region: "
                               "buffer lifetime analysis assumes a DAG");
    }
    tempMark.insert(op);

    // Visit operand defs first
    for (Value operand : op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (defOp->getBlock() == block)
          visit(defOp);
      }
    }

    visited.insert(op);
    tempMark.erase(op);
    topoOrder.push_back(op);
  };

  // Visit operations in reverse order since it's likely that
  // operations that run later are already toward the end of the block
  for (Operation *op : llvm::reverse(blockOrderOps)) {
    if (!visited.count(op))
      visit(op);
  }

  // Populate the index maps
  indexToOp.resize(numOps);
  for (unsigned i = 0; i < numOps; ++i) {
    opIndex[topoOrder[i]] = i;
    indexToOp[i] = topoOrder[i];
  }

  // 2) Pre‑compute in‑block users for each op.
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *, 4>> users;
  users.reserve(numOps);
  for (auto *def : topoOrder) {
    for (Value res : def->getResults()) {
      for (auto &use : res.getUses()) {
        Operation *user = use.getOwner();
        if (user->getBlock() == block)
          users[def].push_back(user);
      }
    }
  }

  // 3) Initialise each runsAfter bit‑vector with {self, terminator}.
  Operation *terminator = block->getTerminator();
  const unsigned termIdx = opIndex[terminator];
  for (auto *op : topoOrder) {
    auto &bv = runsAfter[op];
    bv.set(opIndex[op]);
    bv.set(termIdx);
  }

  // 4) Single reverse sweep – topo guarantees users are later in the list.
  for (int i = static_cast<int>(numOps) - 1; i >= 0; --i) {
    Operation *def = topoOrder[i];
    auto &defSet = runsAfter[def];
    for (Operation *user : users[def])
      defSet |= runsAfter[user]; // SparseBitVector union (cheap)
  }
}
} // namespace

DependencyTree DependencyTree::getDependencyTree(Block *block) {
  DependencyTree tree;

  // Build runsAfter sets
  llvm::DenseMap<Operation *, llvm::SparseBitVector<64>> runsAfterSparse;
  llvm::DenseMap<Operation *, unsigned> opToIdx;
  llvm::SmallVector<Operation *> idxToOp;
  buildRunsAfter(block, runsAfterSparse, opToIdx, idxToOp);

  const unsigned numOps = idxToOp.size();

  tree.runsAfter.resize(numOps);
  tree.runsAfterCounts.resize(numOps);

  for (auto [index, op] : llvm::enumerate(idxToOp)) {
    tree.runsAfter[index] = std::move(runsAfterSparse[op]);
    tree.runsAfterCounts[index] = tree.runsAfter[index].count();
  }

  // Immediate runsAfter (closest successor in the set with largest size)
  llvm::DenseMap<Operation *, Operation *> immediate;
  Operation *terminator = block->getTerminator();
  for (auto *op : idxToOp) {
    if (op == terminator)
      continue;
    const auto &mySet = tree.runsAfter[opToIdx[op]];
    Operation *closest = nullptr;
    uint32_t maxSz = 0;

    // Iterate over all bits in the SparseBitVector
    for (unsigned i : mySet) {
      Operation *cand = idxToOp[i];
      if (cand == op)
        continue;

      uint32_t sz = tree.runsAfterCounts[i];
      if (sz > maxSz) {
        closest = cand;
        maxSz = sz;
      }
    }

    assert(closest && "terminator must at least be present");
    immediate[op] = closest;
  }

  // Build tree structure.
  tree.root = tree.getOrCreate(terminator);
  for (auto *op : idxToOp) {
    if (op == terminator)
      continue;
    auto *parent = tree.getOrCreate(immediate[op]);
    auto *child = tree.getOrCreate(op);
    parent->addChild(child);
  }

  // DFS to set pre/post indices + depths for constant‑time queries.
  using VisitState = std::tuple<DependencyTreeNode *, bool, int>;
  SmallVector<VisitState> stack;
  int index = 0;

  stack.push_back({tree.root, true, 0});

  while (!stack.empty()) {
    auto [node, isFirstVisit, depth] = stack.pop_back_val();
    if (!node)
      continue;

    if (isFirstVisit) {
      // preorder
      node->depth = depth;
      node->preIndex = index++;

      // Schedule to visit again for post-order
      stack.push_back({node, false, depth});

      for (auto child : llvm::reverse(node->children)) {
        stack.push_back({child, true, depth + 1});
      }
    } else {
      // postorder
      node->postIndex = index++;
    }
  }

  // Persist mapping for external queries.
  tree.operationToIndex.swap(opToIdx);
  tree.indexToOperation.swap(idxToOp);

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
    auto alloc = dyn_cast<MGP::BufferAllocOp>(op);
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
  if (!root || !root->op)
    return nullptr;
  return root->op->getBlock();
}

DependencyTreeNode *DependencyTree::getOrCreate(Operation *op) {
  std::unique_ptr<DependencyTreeNode> &node = nodes[op];
  if (node == nullptr)
    node = std::make_unique<DependencyTreeNode>(op);
  return node.get();
}

bool DependencyTree::contains(Operation *op) const {
  return nodes.contains(op);
}

bool DependencyTree::executesAfter(Operation *a, Operation *b) const {
  assert(a && b && "Invalid operations provided");

  // Use operationToIndex directly for faster lookup
  auto itA = operationToIndex.find(a);
  auto itB = operationToIndex.find(b);

  if (itA == operationToIndex.end() || itB == operationToIndex.end()) {
    // If either operation is not in the map, we can't determine execution order
    return false;
  }

  unsigned int indexA = itA->second;
  unsigned int indexB = itB->second;

  // The 'runsAfter' set for an operation (node 'b') contains all operations
  // that execute after it.
  return runsAfter[indexB].test(indexA);
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
DependencyTree::findLCA(ArrayRef<DependencyTreeNode *> nodes) const {
  assert(!nodes.empty());

  DependencyTreeNode *lca = nodes[0];
  for (size_t i = 1; i < nodes.size(); ++i) {
    lca = findLCA(lca, nodes[i]);
    if (!lca)
      return nullptr;
  }
  return lca;
}

DependencyTreeNode *DependencyTree::findLCA(ArrayRef<Operation *> nodes) const {
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
M::MGP::getAllocUses(MGP::BufferAllocOp allocOp) {
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

      if (isa<MGP::CallOp>(*use.getOwner()))
        return Error("call op may alias");

      if (use.getOwner()->hasTrait<mlir::OpTrait::IsTerminator>() &&
          use.getOwner()->getParentOfType<MGP::ModelOp>().getIsSubgraph())
        return Error("terminator op of subgraph may alias");

      if (isBufferOperand &&
          isa<GRT::GRTMayAliasInterface, GRT::GRTExtendsLifetimeInterface>(
              *use.getOwner()))
        stack.push_back(use.getOwner());
    }
  }
  return uses;
}

BufferLifetimeAnalysisPass::BufferLifetimeAnalysisPass(Operation *op) {
  ASSERT_STREAM(isa<ModelOp>(op),
                "BufferLifetimeAnalysisPass can only run on an MGP::ModelOp");

  SmallVector<Block *> blocks;
  op->walk<mlir::WalkOrder::PreOrder>([&](Block *block) {
    if (isa_and_present<MOGG::GenericCompiledRegionOp>(block->getParentOp()))
      return WalkResult::skip();

    blocks.push_back(block);
    return WalkResult::advance();
  });

  using CacheT = llvm::DenseMap<Block *, DependencyTree>;

  auto func = [&](CacheT &cache, Block *block) {
    cache[block] = DependencyTree::getDependencyTree(block);
    return success();
  };

  auto consolidate = [&](CacheT &cache, MutableArrayRef<CacheT> others) {
    for (auto &other : others)
      for (auto &[key, val] : other)
        cache[key] = std::move(val);
  };

  CacheT cache;
  (void)M::failableParallelForEach(op->getContext(), blocks, func, cache,
                                   consolidate);

  for (auto &[key, value] : cache)
    dominatorTrees[key] = std::make_unique<DependencyTree>(std::move(value));
}
