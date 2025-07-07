//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Analysis/ModuleHasher.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/TransformUtils/SCCUtils.h"
#include "Support/AssertStream.h"
#include "Support/HashUtils.h"

#include "llvm/ADT/ScopeExit.h"

using namespace M;

namespace {

constexpr StringLiteral kGeneratorHashName = "generator_hash";
constexpr StringLiteral kSemanticHashName = "semantic_hash";

/// The analysis for each node is a small state machine computing the
/// different parts of the state. This is for convenience as parts of the
/// state depend on each other and the structure needs to fit into the
/// SCCGraph structure, which iteratively applies an analysis update function to
/// each node in an SCC.
enum class State {
  SetGenerator,
  SetInputs,
  SetScc,
  SetFinal,
  Done,
};

State nextState(State state) {
  return (State)std::min((int)State::Done, (int)state + 1);
}

bool isAfter(State lhs, State rhs) { return (int)lhs > (int)rhs; }

struct Node : public KGEN::SCCNode<Node, KGEN::GeneratorOp, KGEN::CallOp> {
  using Base = KGEN::SCCNode<Node, KGEN::GeneratorOp, KGEN::CallOp>;

  using Base::Base;
  Node(KGEN::GeneratorOp func)
      : Base(func), functionName(func ? func.getSymNameAttr() : nullptr) {}

  template <typename T>
  void addHash(T &&value) {
    *hasher << std::forward<T>(value);
  }

  /// Hasher for the current node.
  /// Stored in a separate allocation as the hasher state is quite large
  /// (~800 bytes) and we only need them active while computing the hash for the
  /// current SCC.
  std::unique_ptr<raw_xxhash_stream> hasher;

  /// Need to save the function name as we have to temporarily mutate it in
  /// setGeneratorHash while computing the bytecode hash.
  StringAttr functionName;

  /// The bytecode hash of the generator for the node.
  /// We need this state here since it can be observed by other members of
  /// the same SCC.
  StringAttr generatorHash;

  /// The full hash of the current node:
  ///
  ///     totalHash = (hash(generator) + hash(inputs) + hash(SCC members))
  StringAttr totalHash;

  State state{State::SetGenerator};
};

struct Graph : public KGEN::SCCGraph<Graph, Node> {
  bool doAnalysis(Node *node);
  void doRewrite(const Node *node);

  bool setGeneratorHash(Node *node);
  bool setInputsHash(Node *node);
  bool setFinalHash(Node *node);
  bool setSccHash(Node *node);
};

bool Graph::setGeneratorHash(Node *node) {
  node->hasher = std::make_unique<raw_xxhash_stream>();

  auto ctx = node->func.getContext();
  Builder builder(ctx);

  // Subtlety, the nodes in the current SCC cannot be handled by erasing their
  // name. Erasing the names works for the inputs to the SCC because we
  // incorporate each _callsite_ when computing the inputs hash (see
  // setInputsHash).
  //
  // Instead, map all reference to functions in the current SCC to a unique
  // number.
  DenseMap<Attribute, Attribute> attrMap;
  for (auto [index, sccNode] : llvm::enumerate(node->scc->nodes)) {
    auto name = sccNode->functionName;
    attrMap[name] = builder.getIndexAttr(static_cast<int64_t>(index));
  }

  auto emptyString = builder.getStringAttr("");
  auto replacer = [&](Attribute attr) -> Attribute {
    if (auto symbol = dyn_cast<FlatSymbolRefAttr>(attr))
      return attrMap.lookup_or(symbol.getAttr(), emptyString);
    return attr;
  };

  {
    // Set the function name so we do not incorporate the function name into
    // the bytecode hash.
    auto oldName = node->func.getSymName();
    auto guard = llvm::make_scope_exit([&]() { node->func.setName(oldName); });

    node->func.setSymName("");

    // Hash the bytecode of the backing generator.
    (void)M::writeBytecode(node->func, *node->hasher, replacer);
  }

  node->generatorHash = builder.getStringAttr(node->hasher->hashString());
  return true;
}

bool Graph::setInputsHash(Node *node) {
  // Incorporate the hash of each callee for calls that target functions
  // outside the current SCC.
  for (auto sccNode : node->scc->nodes) {
    for (auto edge : sccNode->callsites) {
      // Calls within the current SCC will not have their hash set yet.
      if (auto nodeHash = edge.node->totalHash)
        node->addHash(edge.node->totalHash.strref());
    }
  }

  return true;
}

bool Graph::setSccHash(Node *node) {
  // Can only proceed if all nodes in the SCC have their generator hashes set.
  if (!llvm::all_of(node->scc->nodes, [&](Node *sccNode) {
        return isAfter(sccNode->state, State::SetGenerator);
      }))
    return false;

  for (auto sccNode : node->scc->nodes)
    node->addHash(sccNode->generatorHash.strref());

  return true;
}

bool Graph::setFinalHash(Node *node) {
  Builder builder(node->func.getContext());
  node->totalHash = builder.getStringAttr(node->hasher->hashString());
  node->hasher = nullptr;
  return true;
}

bool Graph::doAnalysis(Node *node) {
  State originalState = node->state;

  while (true) {
    bool advance = false;
    switch (node->state) {
    case State::SetGenerator:
      advance = setGeneratorHash(node);
      break;
    case State::SetScc:
      advance = setSccHash(node);
      break;
    case State::SetInputs:
      advance = setInputsHash(node);
      break;
    case State::SetFinal:
      advance = setFinalHash(node);
      break;
    case State::Done:
      advance = false;
      break;
    }

    if (!advance)
      break;
    node->state = nextState(node->state);
  }

  return node->state != originalState;
}

void Graph::doRewrite(const Node *node) {
  auto func = node->func;

  Builder builder(func->getContext());

  NamedAttrList attrs(func->getDiscardableAttrDictionary());
  attrs.append(kGeneratorHashName, node->generatorHash);
  attrs.append(kSemanticHashName, node->totalHash);
  func->setDiscardableAttrs(attrs);
}

} // end namespace

void M::MOGG::hashModuleContents(mlir::ModuleOp module) {
  SymbolTable symtab(module);
  hashModuleContents(module, symtab);
}

void M::MOGG::hashModuleContents(mlir::ModuleOp module,
                                 const mlir::SymbolTable &symtab) {
  AsyncRT::Runtime &runtime =
      *loadContext(module.getContext())->get<AsyncRT::Runtime>();

  Graph graph;
  graph.build(module, symtab);
  graph.run(runtime);
}

std::string M::MOGG::getSemanticHash(KGEN::GeneratorOp generator) {
  auto hashAttr = generator->getAttrOfType<StringAttr>(kSemanticHashName);
  ASSERT_STREAM(hashAttr, "unable to find semantic hash for function");
  return hashAttr.str();
}
