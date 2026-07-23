//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "MLRT/include/MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/include/MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "MLRT/include/MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "Support/AssertStream.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/CommonAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace M::MGP {

#define GEN_PASS_DEF_OUTLINEMODELSTOCALLABLES
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

using namespace M;
using namespace M::MGP;

namespace {

class OutlineModelsToCallablesPass
    : public impl::OutlineModelsToCallablesBase<OutlineModelsToCallablesPass> {
  using OutlineModelsToCallablesBase::OutlineModelsToCallablesBase;

  struct GraphContext;

  /// Corresponds 1:1 with `MGP::ModelOp`s in the source IR.
  /// This structure lives longer than the original `MGP::ModelOps`,
  /// though, and is used to pass context to the "parent" graph
  /// of a subgraph during `rewriteAndSeal()`.
  struct GraphContext {
    struct SealedGraphContext {
      MEF::CallableOp initCallable;
      MEF::CallableOp execCallable;

      // pass arg structure for matching up implicit arg passing
      SealedGraphContext(MEF::CallableOp init, MEF::CallableOp exec)
          : initCallable(init), execCallable(exec) {}
    };

    /// Only exists once `sealed` is true
    std::optional<SealedGraphContext> sealedContext;

    /// Only exists while `sealed` is false
    std::optional<MGP::ModelOp> unsealedContext;

    std::optional<mlir::Value> initPrefixValue;
    std::optional<mlir::Value> execPrefixValue;

    bool hasSubgraphs;

    llvm::SmallVector<GraphContext *> referencedBy;

    GraphContext(MGP::ModelOp model)
        : sealedContext(), unsealedContext(model), hasSubgraphs(false),
          sealed(false) {}

    /// GraphContext is definitionally inter-referential,
    /// so copying is poorly defined
    GraphContext(const GraphContext &) = delete;
    GraphContext &operator=(const GraphContext &) = delete;

    /// But keep the ability to move at initial creation to properly
    /// construct unique ptrs
    GraphContext(GraphContext &&) = default;

    /// Marks this GraphContext as having been visited and rewritten,
    /// such that the original `MGP::ModelOp` no longer exists.
    /// All information that is needed after this point must be
    /// accessed through `sealedContext` or the other fields on `this`.
    void seal() {
      ASSERT_STREAM(this->sealedContext.has_value(),
                    "can't seal a context without initializing sealedContext");
      ASSERT_STREAM(!this->sealed,
                    "tried to seal a GraphContext that was already sealed");
      this->sealed = true;
    }

    /// See `seal()`
    bool isSealed() { return this->sealed; }

  private:
    /// See `seal()`
    bool sealed;
  };

  /// This is used to wrap the state to pass around for a single invocation of
  /// this pass, and builds and holds the entire interconnected call
  /// graph of `GraphContext`s within the current module.
  ///
  /// See `GraphContext` for most of the semantics of `ModuleContext`s contents
  struct ModuleContext {
    ModuleOp moduleOp;

    llvm::DenseMap<StringAttr, std::unique_ptr<GraphContext>> models;

    /// The non-subgraph models in this module
    /// Every currently-GraphRT-executable module will have at most
    /// one of these.
    llvm::SmallVector<GraphContext *, 1> entryGraphs;

    GraphContext *intern(GraphContext gc) {
      ASSERT_STREAM(!gc.isSealed(),
                    "tried to intern an already-sealed GraphContext");

      auto gcUnique = std::make_unique<GraphContext>(std::move(gc));
      auto gcPtr = gcUnique.get();

      if (!gc.unsealedContext.value().getIsSubgraph())
        this->entryGraphs.push_back(gcPtr);

      bool inserted = this->models
                          .insert({gcPtr->unsealedContext.value().getNameAttr(),
                                   std::move(gcUnique)})
                          .second;

      ASSERT_STREAM(inserted, "two models with the same symbol name existed:"
                                  << gcPtr->unsealedContext.value().getName());

      return gcPtr;
    }

    GraphContext *getModel(StringAttr name) {
      auto model = this->models.find(name);
      if (model != this->models.end())
        return model->getSecond().get();
      else
        return nullptr;
    }

    ModuleContext(ModuleOp moduleOp) : moduleOp(moduleOp), models() {
      // First, register all ModelOps so we can ref them
      for (auto model : moduleOp.getOps<MGP::ModelOp>())
        this->intern(GraphContext(model));

      // Then, link up references now that both sides must exist
      for (auto &[name, gc] : this->models) {
        gc->unsealedContext.value().getExecute().walk([&](MGP::CallOp call) {
          auto &callee = *this->models.at(call.getCalleeAttr().getAttr());
          gc->hasSubgraphs = true;
          callee.referencedBy.push_back(&*gc);
        });
      }
    }
  };

  /// NOTE: operates pre-sealing _only_
  /// Takes the model graph and adds an arg to each of init and exec
  /// that will receive a stacked prefix during the rewriteAndSeal
  /// stage of this pass. Then, prefixes that arg to the placeholder
  /// weight name at runtime using a string concat op for any
  /// BufferConstantExternalOps that are marked as placeholders.
  void localPropagatePrefixes(ModuleContext &context,
                              std::optional<mlir::Value> prefix) {
    for (auto &[name, modelPtr] : context.models) {
      ModelOp model = modelPtr.get()->unsealedContext.value();
      if (model.getIsSubgraph()) {
        auto builder = mlir::OpBuilder(model.getContext());
        auto implicitArgsCount = model.getExecuteImplicitArguments().size();
        Block *execBlock = model.getExecuteBlock();
        Block *initBlock = model.getInitBlock();

        auto mojoStringType = MGP::OpaqueType::get(
            builder.getContext(), builder.getStringAttr("mojo_string"), {});

        // for exec, put it after implicit args/before explicit arguments
        Value execPrefix = execBlock->insertArgument(
            implicitArgsCount, mojoStringType, model->getLoc());

        modelPtr.get()->execPrefixValue = execPrefix;

        // for init, put it on the end
        Value initPrefix =
            initBlock->addArgument(mojoStringType, model->getLoc());

        // fixup any ops (currently only buffer.constant.external)
        // that depend on prefix, but _not_ mgp.call as that
        // specifically is fixed up later during conversion
        llvm::SmallVector<std::pair<Block *, mlir::Value>> blocks(
            {{initBlock, initPrefix}, {execBlock, execPrefix}});

        for (auto [block, prefix] : blocks) {
          block->walk([&](MGP::BufferConstantExternalOp op) {
            if (!op.getEIsPlaceholder()) {
              // Anything that isn't marked as IsPlaceholder
              // has its final name already, and shouldn't get a prefix
              return mlir::WalkResult::skip();
            }

            builder.setInsertionPoint(op);

            Value rawName = op.getName();

            Value concatenatedName = StringConcatOp::create(
                builder, op.getLoc(), mojoStringType, prefix, rawName);

            op.getNameMutable().assign(concatenatedName);

            return mlir::WalkResult::advance();
          });
        }

        modelPtr.get()->initPrefixValue = initPrefix;

        // init should return the prefix so it will
        // pass into exec without needing to re-munge strings
        llvm::cast<MEF::OutputOp>(initBlock->getTerminator())
            .getOperandsMutable()
            .append(initPrefix);
      }
    }
  }

  /// Performs a post-order traversal of the model call graph,
  /// asserting that there are no recursive model dependencies.
  ///
  /// For any `GraphContext` that is not yet visited, consumes the
  /// `MGP::ModelOp` that `GraphContext` represents.
  ///
  /// That model is then pulled apart into two `MEF::CallableOp`s,
  /// one for init and one for exec. Then, for every `MGP::CallOp`,
  /// the target `GraphContext`'s forwarded values are plumbed from
  /// one `GRT::Invoke` in self's init to one in self's exec, such
  /// that the callee's exec receives the output of the callee's init
  /// plus any explicit args and prefix. All of these values,
  /// plus any values that otherwise need to be passed from self's
  /// init to self's exec become self's forwarded bundle for anything
  /// that calls self.
  ///
  /// If the current model is _not_ marked as a subgraph, then it
  /// is considered a possible entry point to the graph, and a
  /// `MEF::ModelOp` is created to wrap self's init and exec in
  /// a form that GraphRT can directly execute.
  ///
  /// The original GraphContext is then "sealed", with the
  /// original `MGP::ModelOp` being erased.
  FailureOr<GraphContext *>
  rewriteAndSeal(ModuleContext &context, GraphContext &graph,
                 llvm::SmallVector<ModelOp> ancestors) {

    // If already visited this graph, we've resolved everything needed
    // and have emitted callables already. Can safely return the already-sealed
    // value.
    if (graph.isSealed())
      return &graph;

    // first, check if there's a cycle in arriving at this op
    // this should be cheap compared to the rest of what's going on
    // on account of subgraph call depth rarely (if ever) exceeding 2
    for (ModelOp ancestor : ancestors) {
      if (ancestor.getNameAttr() ==
          graph.unsealedContext.value().getNameAttr()) {
        InFlightDiagnostic diag = graph.unsealedContext.value().emitOpError(
            "cycle detected in subgraph call graph");

        // Annotate only the subgraphs that form the cycle, starting from the
        // ancestor we are re-entering. Ancestors before that point are on the
        // call stack but not part of the cycle itself.
        bool inCycle = false;
        for (ModelOp diagAncestor : ancestors) {
          if (diagAncestor.getNameAttr() ==
              graph.unsealedContext.value().getNameAttr())
            inCycle = true;
          if (inCycle)
            diag.attachNote(diagAncestor.getLoc()) << "subgraph in cycle";
        }

        return failure();
      }
    }

    ancestors.push_back(graph.unsealedContext.value());

    // extract info from MGP::ModelOp before denormalizing it
    ModelOp model = graph.unsealedContext.value();
    Location modelLoc = model->getLoc();
    StringRef modelName = model.getName().getValue();
    bool isSubgraph = model.getIsSubgraph();

    // at the very least, forward the full set of parameters of _this_ graph
    // from init to execute, and from each iter of execute to the next
    auto initOutput =
        llvm::cast<MEF::OutputOp>(model.getInitBlock()->getTerminator());

    // Collect this now so iteration is cheap, since
    // need to do multiple passes over it to apply fixups
    struct CallContext {
      MGP::CallOp call;
      StringAttr prefixAttr;
      GraphContext *toGraph;

      // within the scope of the containing graph,
      // these are the results of the call to the `init` callable
      // that should be forwarded as a bundle
      llvm::SmallVector<mlir::Value> initResults;
      llvm::SmallVector<Type> initResultTypes;

      llvm::SmallVector<mlir::Value> execArgsBundle;
    };

    llvm::SmallVector<CallContext> calls;

    // CallOps only happen in the Execute region
    WalkResult result = model.getExecute().walk([&](MGP::CallOp call) {
      GraphContext *maybeCallee =
          context.getModel(call.getCalleeAttr().getAttr());
      ASSERT_STREAM(maybeCallee, << "CallOp's callee did not exist in module");

      StringAttr prefix = call.getPrefixAttr();

      FailureOr<GraphContext *> sealed =
          rewriteAndSeal(context, *maybeCallee, ancestors);

      if (failed(sealed))
        return WalkResult::interrupt();

      calls.push_back({call, prefix, sealed.value(), {}, {}, {}});
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return failure();

    // translate MGP::CallOp to InvokeOps in init and exec,
    // thread arg bundles through init output -> exec input
    {
      OpBuilder initBuilder(model.getInitBlock()->getTerminator());
      // if I have any child calls, I take each of their
      // init's outputs and stack them onto my init's outputs,
      // put them into my exec's inputs, and expect them to be forwarded
      //
      // I will also take every call I make and pass their stacked forwarded
      // args
      //
      // Init is going to receive:
      //  (...<existing ModelOp init operands>, <prefix>)
      // Init is going to output:
      //  (<existing ModelOp init results>, <prefix>, <inner callee bundles>)
      //
      // This entire thing becomes a single inner callee bundle as viewed from
      // "above"
      //
      // Exec is going to receive:
      //  (<existing ModelOp init results>, <prefix>, <inner callee bundles>,
      //  <explicit arguments>
      //
      // Exec is going to output:
      //  (<ModelOp exec results>)

      // Keep around
      SmallVector<Value> initChains;

      for (auto &[call, prefix, ctx, initResults, initResultTypes,
                  execArgsBundle] : calls) {
        GraphContext::SealedGraphContext target = ctx->sealedContext.value();

        // Create prefix here to pass down in.
        // This will also propagate through the bundle to exec,
        // and should be pulled out there
        auto mojoStringType =
            MGP::OpaqueType::get(initBuilder.getContext(),
                                 initBuilder.getStringAttr("mojo_string"), {});

        Value prefixValue = StringCreateOp::create(initBuilder, call->getLoc(),
                                                   mojoStringType, prefix)
                                .getResult();
        if (graph.execPrefixValue.has_value()) {
          auto parentPrefix = graph.execPrefixValue.value();
          prefixValue =
              StringConcatOp::create(initBuilder, call->getLoc(),
                                     mojoStringType, parentPrefix, prefixValue)
                  .getResult();
        }

        // now, build up expected init operands for the call
        auto initArgs =
            llvm::SmallVector<Value>(model.getInitBlock()->getArguments());

        // replace parent prefix with inner prefix
        initArgs.push_back(prefixValue);

        auto initInvokeOp = GRT::InvokeOp::create(
            initBuilder, call.getLoc(), target.initCallable.getResultTypes(),
            target.initCallable.getName(), initArgs);

        auto results = initInvokeOp.getResults();

        initResults.append(results.begin(), results.end());

        initResultTypes.append(results.getTypes().begin(),
                               results.getTypes().end());

        initChains.push_back(initArgs.front());
      }

      // now, we have the stacked results of calling all child inits,
      // so push them onto our `output`, and make sure the inputs to execute
      // match
      auto initOutputOperands = initOutput.getOperandsMutable();

      // this starts out containing the "original" results
      for (auto &callContext : calls) {
        size_t insertionIndex = initOutputOperands.size();
        initOutputOperands.append(
            callContext.initResults); // push all of the bundles

        for (size_t i = 0; i < callContext.initResultTypes.size(); i++) {
          auto execArg = model.getExecuteBlock()->insertArgument(
              insertionIndex + i, callContext.initResultTypes[i],
              model->getLoc());

          callContext.execArgsBundle.push_back(execArg);
        }
      }

      for (auto &callContext : calls) {
        // insert any forwarded bundles between the implicit args (including
        // prefix) and the explicit args

        // -1 to avoid dropping the prefix
        auto numOperands = callContext.call.getOperands().size();
        auto dropCount = callContext.initResults.size() - 1;
        ASSERT_STREAM(numOperands >= dropCount,
                      << "call has " << numOperands
                      << " operands but trying to drop " << dropCount);
        auto explicitArgs =
            callContext.call.getOperands().drop_front(dropCount);

        auto &args = callContext.execArgsBundle;
        args.append(explicitArgs.begin(), explicitArgs.end());

        GraphContext::SealedGraphContext target =
            callContext.toGraph->sealedContext.value();

        OpBuilder execBuilder(callContext.call);

        auto execInvokeOp =
            GRT::InvokeOp::create(execBuilder, callContext.call->getLoc(),
                                  target.execCallable.getResultTypes(),
                                  target.execCallable.getName(), args);

        callContext.call->replaceAllUsesWith(execInvokeOp->getResults());
        callContext.call->erase();
      }

      // As a fixup, ensure the outer init's chain also waits for the
      // inner init's chain. This _seems_ like it shouldn't be necessary,
      // but some situations seem to implicitly expect
      // subgraph inits to be done if their parent init is done.
      Value parentChain = initOutput.getOperand(0);
      initChains.push_back(parentChain);
      auto newInitChain = GRT::ChainCreateOp::create(
          initBuilder, initOutput->getLoc(), initChains);
      initOutput->setOperand(0, newInitChain.getResult());
    }

    mlir::OpBuilder builder(context.moduleOp);

    // init is "easy", it just takes the same inputs as it always did and
    // passes it to every callee. It then outputs what it originally output
    // plus what each callee outputs.
    auto initFunctionType = mlir::FunctionType::get(
        context.moduleOp.getContext(), model.getInitBlock()->getArgumentTypes(),
        initOutput->getOperandTypes());

    auto execFunctionType = mlir::FunctionType::get(
        context.moduleOp.getContext(),
        model.getExecuteBlock()->getArgumentTypes(),
        model.getExecuteBlock()->getTerminator()->getOperandTypes());

    // all subgraphs need to get turned into MEF::Callables

    builder.setInsertionPoint(context.moduleOp);
    auto parent = ancestors.back();
    builder.setInsertionPoint(parent);

    auto initCallable = MEF::CallableOp::create(
        builder, modelLoc,
        builder.getStringAttr(modelName + mlir::Twine("$init")), {}, {}, {});

    initCallable.setFunctionType(initFunctionType);

    auto execCallable = MEF::CallableOp::create(
        builder, modelLoc,
        builder.getStringAttr(modelName + mlir::Twine("$exec")), {}, {}, {});

    execCallable.setFunctionType(execFunctionType);

    initCallable.getCallableRegion()->takeBody(model.getInit());
    execCallable.getCallableRegion()->takeBody(model.getExecute());

    // take implicit passed context and forward it
    graph.sealedContext =
        GraphContext::SealedGraphContext(initCallable, execCallable);

    if (!isSubgraph) {
      // need to fix up a wrapper model so the runtime still has
      // the expected entry point
      auto loc = model->getLoc();

      mlir::FunctionType initFnType = initCallable.getFunctionType();
      mlir::FunctionType execFnType = execCallable.getFunctionType();

      // Extract devices and frameworkName from the original model
      auto devices = model.getDevices();
      auto frameworkName = model.getFrameworkName();

      // Extract these BEFORE erasing the model to avoid use-after-free
      auto initialModelFnType = model.getInputOutputTypes();
      auto extraAttrs = model.getTheExtraAttrsAttr();

      OpBuilder builder(model->getContext());
      builder.setInsertionPointAfter(model);

      auto resultDeviceIndicesAttr = builder.getArrayAttr({});
      auto argumentDeviceIndicesAttr = builder.getArrayAttr({});
      auto argumentNamesAttr = builder.getArrayAttr({});
      auto resultNamesAttr = builder.getArrayAttr({});
      if (extraAttrs) {
        auto resultDeviceIndices = extraAttrs.get(kMgpModelResultDeviceIndices);
        if (auto indices =
                llvm::cast_if_present<mlir::ArrayAttr>(resultDeviceIndices))
          resultDeviceIndicesAttr = indices;

        auto argumentDeviceIndices =
            extraAttrs.get(kMgpModelArgumentDeviceIndices);
        if (auto indices =
                llvm::cast_if_present<mlir::ArrayAttr>(argumentDeviceIndices))
          argumentDeviceIndicesAttr = indices;

        if (auto names = llvm::cast_if_present<mlir::ArrayAttr>(
                extraAttrs.get(kMgpModelArgumentNames)))
          argumentNamesAttr = names;

        if (auto names = llvm::cast_if_present<mlir::ArrayAttr>(
                extraAttrs.get(kMgpModelResultNames)))
          resultNamesAttr = names;
      }

      auto mefModel = MEF::ModelOp::create(
          builder, loc, devices, frameworkName, initFnType, execFnType,
          initialModelFnType, argumentDeviceIndicesAttr,
          resultDeviceIndicesAttr, argumentNamesAttr, resultNamesAttr,
          modelName);

      auto &initBlock = mefModel.getInit().emplaceBlock();
      auto &execBlock = mefModel.getExecute().emplaceBlock();

      SmallVector<std::pair<Block *, MEF::CallableOp>> blocks{
          {&initBlock, initCallable}, {&execBlock, execCallable}};

      for (auto &[block, callable] : blocks) {
        builder.setInsertionPointToEnd(block);

        SmallVector<Value> operands;

        for (auto argTy : callable.getArgumentTypes()) {
          auto arg = block->addArgument(argTy, loc);
          operands.push_back(arg);
        }

        auto invoke = GRT::InvokeOp::create(
            builder, loc, callable.getResultTypes(), operands);

        invoke.setTarget(callable.getName());

        MEF::OutputOp::create(builder, loc, invoke.getResults());
      }
    }

    graph.unsealedContext->erase();
    graph.unsealedContext = {};

    graph.seal();
    ancestors.pop_back();
    return &graph;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    ModuleContext context(moduleOp);

    // First, propagate prefixes so the main rewrite traversal
    // doesn't have to spend so much effort juggling operands.
    localPropagatePrefixes(context, {});

    // Now, go through and rewrite every model into callables
    // with forwarded init context. This will also turn
    // non-subgraph `mgp.model`s into `mef.model`s.
    for (GraphContext *entryGraph : context.entryGraphs) {
      ASSERT_STREAM(!entryGraph->unsealedContext.value().getIsSubgraph(),
                    "no entry graph should be a subgraph");

      llvm::SmallVector<ModelOp> ancestors;

      FailureOr<GraphContext *> result =
          rewriteAndSeal(context, *entryGraph, ancestors);

      if (failed(result))
        signalPassFailure();
    }
  }
};

} // namespace
} // namespace M::MGP
