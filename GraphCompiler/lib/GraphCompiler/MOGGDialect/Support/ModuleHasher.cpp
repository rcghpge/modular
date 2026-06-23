//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/ModuleHasher.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/TransformUtils/SCCUtils.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "Support/AssertStream.h"
#include "Support/HashUtils.h"

#include "mlir/Bytecode/BytecodeWriter.h"

using namespace M;

namespace {

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

struct Node
    : public KGEN::SCCNode<Node, KGEN::GeneratorOpInterface, KGEN::CallOp> {
  using Base = KGEN::SCCNode<Node, KGEN::GeneratorOpInterface, KGEN::CallOp>;

  using Base::Base;
  Node(KGEN::GeneratorOpInterface func)
      : Base(func), functionName(func ? func.getNameAttr() : nullptr) {}

  template <typename T>
  void addHash(T &&value) {
    *hasher << std::forward<T>(value);
  }

  /// Hasher for the current node.
  /// Stored in a separate allocation as the hasher state is quite large
  /// (~800 bytes) and we only need them active while computing the hash for the
  /// current SCC.
  std::unique_ptr<raw_xxhash128_stream> hasher;

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
  void build(ModuleOp module, const SymbolTable &symtab);

  bool doAnalysis(Node *node);
  void doRewrite(const Node *node);

  bool setGeneratorHash(Node *node);
  bool setInputsHash(Node *node);
  bool setFinalHash(Node *node);
  bool setSccHash(Node *node);
};

void Graph::build(ModuleOp module, const SymbolTable &symtab) {
  // This is a custom build method for the graph type. The callgraph normally
  // built by KGEN::SCCGraph uses call operations as the edges in the graph.
  //
  // Before Elaboration, KGEN IR can refer to functions through
  // FlatSymbolRefAttrs that appear in arbitrary attributes and types. We need
  // to account for those references when computing the module hash, so this
  // code overrides the CallGraphBase::build method in the parent class to get
  // the desired callgraph structure.
  //
  // The end result of this algorithm is a callgraph that is a superset of the
  // callgraph normally built by CallGraphBase::build.

  // Create a callgraph node for each GeneratorOpInterface in the module.
  // This logic is the same as in the parent class' build method.
  for (auto func : module.getOps<FuncOpT>())
    nodes.try_emplace(func, Node(func));

  // Populate the callers and callees lists for each node in the graph.
  for (auto &[caller, callerNode] : nodes) {
    // Rather than creating edges for each GeneratorOpInterface, we scan each
    // Operation for any FlatSymbolRefAttr that points to a GeneratorOpInterface
    // and create edges for those attributes.
    caller.walk([&](Operation *op) {
      // Walker to collect all FlatSymbolRefAttrs and add an edge to the graph.
      mlir::AttrTypeWalker walker;
      walker.addWalk([&](FlatSymbolRefAttr symbol) {
        auto callee =
            dyn_cast_or_null<Node::FuncOpT>(symtab.lookup(symbol.getAttr()));
        if (!callee)
          return;

        Node *calleeNode = &nodes.find(callee)->second;
        callerNode.callsites.emplace_back(dyn_cast<CallOpT>(op), calleeNode);

        calleeNode->callers.push_back(&callerNode);
      });

      // Walk operation attributes.
      walker.walk(op->getAttrDictionary());

      // Walk operation result types.
      for (auto type : op->getResultTypes())
        walker.walk(type);

      // Walk types of block arguments.
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument &arg : block.getArguments()) {
            walker.walk(arg.getType());
          }
        }
      }
    });
  }
}

bool Graph::setGeneratorHash(Node *node) {
  node->hasher = std::make_unique<raw_xxhash128_stream>();

  ModuleOp module = node->func->getParentOfType<ModuleOp>();
  ASSERT_STREAM(module, "generator must live under a module");
  SymbolTable symtab(module);

  auto ctx = node->func.getContext();
  Builder builder(ctx);

  // Subtlety, the nodes in the current SCC cannot be handled by erasing their
  // name. Erasing the names works for the inputs to the SCC because we
  // incorporate each _callsite_ when computing the inputs hash (see
  // setInputsHash).
  //
  // Instead, map all reference to functions in the current SCC to a unique
  // number.
  llvm::SmallDenseMap<SymbolRefAttr, SymbolRefAttr> attrMap;
  for (auto [index, sccNode] : llvm::enumerate(node->scc->nodes)) {
    auto name = FlatSymbolRefAttr::get(sccNode->functionName);
    attrMap[name] =
        FlatSymbolRefAttr::get(builder.getStringAttr(std::to_string(index)));
  }

  auto emptyString = FlatSymbolRefAttr::get(builder.getStringAttr(""));
  auto unknownLoc = UnknownLoc::get(ctx);

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](FlatSymbolRefAttr symbol) -> SymbolRefAttr {
    // Replace SCC member references with stable indices.
    if (auto mapped = attrMap.lookup(symbol))
      return mapped;
    // Replace external GeneratorOpInterface references with empty string —
    // their hash is incorporated via setInputsHash. Non-GeneratorOpInterface
    // refs (e.g., trait symbol refs from TypeConformsToTraitAttr) must be
    // preserved so they contribute to the bytecode hash.
    if (auto *op = symtab.lookup(symbol.getAttr()))
      if (isa<KGEN::GeneratorOpInterface>(op))
        return emptyString;
    return symbol;
  });
  replacer.addReplacement(
      [&](LocationAttr loc) -> LocationAttr { return unknownLoc; });

  // Set the function name so we do not incorporate the function name into
  // the bytecode hash.
  OwningOpRef<KGEN::GeneratorOpInterface> clone = node->func.clone();
  clone->setName("");

  replacer.recursivelyReplaceElementsIn(*clone,
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);

  // Hash the bytecode of the backing generator.
  (void)mlir::writeBytecodeToFile(*clone, *node->hasher);

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
  MLRT::CPUDevice &cpuDevice =
      *loadContext(module.getContext())->get<MLRT::CPUDevice>();

  Graph graph;
  graph.build(module, symtab);
  graph.run(cpuDevice);
}

std::string M::MOGG::getSemanticHash(KGEN::GeneratorOp generator) {
  auto hashAttr = generator->getAttrOfType<StringAttr>(kSemanticHashName);
  ASSERT_STREAM(hashAttr, "unable to find semantic hash for function");
  return hashAttr.str();
}

llvm::DenseMap<StringAttr, KGEN::GeneratorOp>
M::MOGG::applySemanticNaming(ModuleOp module,
                             ArrayRef<StringAttr> symbolsToRename,
                             const SymbolTable &symtab, StringRef namePrefix) {
  // Mapping from old (public) names to the renamed generators.  The keys are
  // the names that callers in the MGP module use (i.e. the linkage name when
  // present, otherwise the sym_name).
  llvm::DenseMap<StringAttr, KGEN::GeneratorOp> renaming;
  renaming.reserve(symbolsToRename.size());

  if (symbolsToRename.empty())
    return renaming;

  // At this stage of Mojo compilation, generator ops do not yet have their
  // 'final' linkage names set onto their symbol names; they're resolved during
  // elaboration. Thus we cannot simply look up the symbols we're interested in
  // the symbol table. We have to build up a few data structures to get us from
  // linkage name to generator. Technically we shouldn't be able to rely on
  // "literal" linakge names (they may contain parameter expressions in theory)
  // so this is a bit of a hack, assuming the symbols we're interested in are
  // simple literals.

  // Build a map from linkageName -> GeneratorOp so we can look up generators
  // by their public name even when the sym_name is mangled.
  llvm::DenseMap<StringAttr, KGEN::GeneratorOp> linkageNameToGen;
  for (auto gen : module.getOps<KGEN::GeneratorOp>()) {
    if (auto ln = gen.getLinkageNameAttr()) {
      // We're only interested in looking up literal linkage names. Our symbols
      // will not refer to parametric ones, so skip them.
      if (auto str = dyn_cast<StringAttr>(ln.getName())) {
        // Convert back from a !kgen.string-typed StringAttr to an untyped one,
        // so that symbol equality works.
        linkageNameToGen[StringAttr::get(module.getContext(), str.getValue())] =
            gen;
      }
    }
  }

  auto findGenerator = [&](StringAttr name) -> KGEN::GeneratorOp {
    if (auto gen = symtab.lookup<KGEN::GeneratorOp>(name))
      return gen;
    return linkageNameToGen.lookup(name);
  };

  for (StringAttr symbol : symbolsToRename) {
    auto generator = findGenerator(symbol);
    ASSERT_STREAM(generator, << "symbol " << symbol << " not found in module");
    // Remove it because it influences the hashing.
    generator.removeSourceNameAttr();
  }

  hashModuleContents(module, symtab);

  // The semantic hashes of the functions we have visited already. This tells
  // us if we need to rename the current generator, or if we can delete it.
  llvm::DenseMap<StringAttr, KGEN::GeneratorOp> visited;
  visited.reserve(symbolsToRename.size());

  // Track old sym_name -> new sym_name for updating references within the
  // kgen module (sym_name may differ from the public linkage name).
  llvm::DenseMap<StringAttr, StringAttr> symRenaming;

  for (StringAttr symbol : symbolsToRename) {
    auto generator = findGenerator(symbol);
    auto hash = generator->getAttrOfType<StringAttr>(kSemanticHashName);
    ASSERT_STREAM(hash, "module has not had semantic hashes computed");

    StringAttr oldSymName = generator.getSymNameAttr();

    // If the operation is already in the module, then it is semantically
    // equivalent to another operation. We can delete it and replace it with its
    // equivalent when updating all symbol uses
    if (auto remapped = visited.lookup_or(hash, nullptr)) {
      symRenaming[oldSymName] = remapped.getSymNameAttr();
      renaming[symbol] = remapped;
      generator.erase();
    } else {
      auto newName = StringAttr::get(module.getContext(),
                                     Twine(namePrefix) + hash.strref());

      symRenaming[oldSymName] = newName;
      renaming[symbol] = generator;
      generator.setSymNameAttr(newName);
      // Remove the linkage name so that it doesn't propagate through
      // elaboration and cause the linkage name resolution to undo this rename.
      generator.removeLinkageNameAttr();
      visited.insert({hash, generator});
    }
  }

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](FlatSymbolRefAttr ref) {
    if (auto newName = symRenaming.lookup_or(ref.getAttr(), {}))
      return FlatSymbolRefAttr::get(newName);
    return ref;
  });

  replacer.recursivelyReplaceElementsIn(module,
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);

  return renaming;
}
