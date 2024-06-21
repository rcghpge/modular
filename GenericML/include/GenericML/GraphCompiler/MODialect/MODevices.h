//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MODEVICES_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MODEVICES_H

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/PagedVector.h"
namespace M::MO {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Returns the #M.device_ref annotation for op, or the 'host' device
/// if there is no such annotation on the op itself.
DeviceRefAttr getDeviceOrDefault(MOHasDeviceInterface op);

/// As above, but for any operation.
DeviceRefAttr getDeviceOrDefaultAny(Operation *op);

/// Returns the #M.device_ref annotation for type which appears in the
/// the given MO graphOp. Returns the 'host' device if there is no such
/// annotation on the type itself.
DeviceRefAttr getDeviceOrDefault(GraphOp graphOp, TensorType type);

/// As above, but for any type.
DeviceRefAttr getDeviceOrDefaultAny(GraphOp graphOp, Type type);

// (See also getHostDevice, declared in MOOps.h so can be used in MOOps.td)

/// Returns true if op should execute on the 'host'.
bool isOnHost(Operation *op);

/// Return true if a value of type should be stored on the 'host'.
bool isOnHost(GraphOp graphOp, Type type);

/// Returns the #M.device_spec_collection annotation for graphOp, or a
/// description of just the current CPU device (ignoring any other device types)
/// if it has no such annotation.
///
/// NOTE: AddDevicesPass will fill in the device specs annotation on the graph
/// based on all available devices. This helper is to support units tests which
/// don't want to have to deal with devices.
ErrorOr<DeviceSpecCollectionAttr> getDeviceSpecsOrDefault(GraphOp graphOp);

//===----------------------------------------------------------------------===//
// DeviceAnnots
//===----------------------------------------------------------------------===//

/// Captures all device-related annotations from one or more operations, and
/// allows them to be 'stamped' onto new ops.
struct DeviceAnnots {
  /// The 'device' attribute of the original op, or null.
  DeviceRefAttr device;

  /// Returns the device annotations for operation.
  static DeviceAnnots forOp(Operation *op);

  /// Unions the device annotations of this and that into this. Returns false if
  /// this and that have conflicting annotations.
  bool add(const DeviceAnnots &that);

  /// Captures the device annotations for operation in this. Returns false if
  /// this and op have conflicting annotations. No-op if op is null.
  bool addOp(Operation *op);

  /// Stamps these device annotations into operation and returns it.
  Operation *stampAny(Operation *op) const;

  /// Stamps these device annotations into operation and returns it.
  template <typename Op>
  Op stamp(Op op) const {
    stampAny(op);
    return op;
  }

  /// Return true if these annotations are consistent with the given device,
  /// which may be null.
  bool isConsistentWith(DeviceRefAttr thatDevice) const;

  /// Returns true if this and that agree in their annotations where they
  /// are given.
  bool isConsistentWith(const DeviceAnnots &that) const;
};

//===----------------------------------------------------------------------===//
// DeviceAwareRewriter
//===----------------------------------------------------------------------===//

/// Wraps a RewriterBase to automatically collect and stamp ops with device
/// annotations.
///
/// Note that we wrap rather than derive since RewriteBase is extensible.
/// Thankfully we only need to forward a dozen or so methods to make
/// DeviceAwareRewriter a usable replacement for the underlying rewriter.
struct DeviceAwareRewriter {
  /// The accumulated device annotations for all the matched ops.
  DeviceAnnots deviceAnnots;
  /// The underlying rewriter.
  mlir::RewriterBase &rewriter;
  /// Accumulated operations which may be bound by patterns, in addition
  /// to the 'anchor' op given to the constructor.
  ///
  /// NOTE: central to the use of `pushOp()` is returning a stable memory
  /// address for a pointer (Operation**). llvm::SmallVector may resize and
  /// change these addresses as we insert more addresses to track in the Vector.
  ///
  /// llvm::PagedVector does not re-allocate memory if we do not do deletion of
  /// elements so is used to guarantee stable addressing.
  llvm::PagedVector<Operation *, sizeof(Operation *) * 16> matchedOps;

  /// Constructs device-aware rewriter using rewriter, and adds the initial
  /// operation to it, on the assumption that op will be replaced by the
  /// rewriter.
  DeviceAwareRewriter(Operation *op, mlir::RewriterBase &rewriter)
      : deviceAnnots(DeviceAnnots::forOp(op)), rewriter(rewriter) {}

  /// Pushes an operation onto matchedOps for later checking by addOps.
  void pushOp(Operation *op) {
    // Update the size of the paged vector
    matchedOps.resize(matchedOps.size() + 1);

    // Update the newly allowed last element
    matchedOps[matchedOps.size() - 1] = op;
  }

  /// Pushes a null operation onto matchedOps, and returns its address.
  /// Can be passed to a op-matching pattern to be assigned if the pattern
  /// matches, then later checked by addOps.
  Operation **pushOp() {
    matchedOps.resize(matchedOps.size() + 1);
    return &matchedOps[matchedOps.size() - 1];
  }

  /// Captures the device annotations for all additional operations, including
  /// those captured in matchedOps, and those given in the argument, on the
  /// assumption they will all be replaced by the rewriter. Returns false if
  /// any ops have conflicting annotations. Null ops are ignored.
  [[nodiscard]] bool
  addOps(const std::initializer_list<Operation *> &ops = {}) {
    return llvm::all_of(
               matchedOps.materialized(),
               [this](Operation *op) { return deviceAnnots.addOp(op); }) &&
           llvm::all_of(
               ops, [this](Operation *op) { return deviceAnnots.addOp(op); });
  }

  /// Like OpBuilder::create, but the new op is stamped.
  ///
  /// TODO: Currently we always stamp every created/replaced op. That's ok
  /// since all MO ops take and return arguments on the same device as the
  /// op itself. However, once we relax that things get more subtle, and
  /// probably the rewriting code needs to distinguish between creating new
  /// ops which must be stamped vs those (eg a mo.constant for an index
  /// operand) which can be left unstamped.
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    return deviceAnnots.stamp(
        rewriter.create<OpTy>(location, std::forward<Args>(args)...));
  }

  template <typename CallableT>
  void modifyOpInPlace(Operation *root, CallableT &&callable) {
    rewriter.modifyOpInPlace(root, std::forward<CallableT>(callable));
  }

  /// Like RewriterBase::replaceOpWithNewOp, but the new op is stamped.
  template <typename OpTy, typename... Args>
  OpTy replaceOpWithNewOp(Operation *op, Args &&...args) {
    auto newOp =
        create<OpTy, Args...>(op->getLoc(), std::forward<Args>(args)...);
    rewriter.replaceOp(op, newOp.getOperation());
    return newOp;
  }

  //===--------------------------------------------------------------------===//
  /// The following redirect to the underlying rewriter.
  //===--------------------------------------------------------------------===//

  template <typename ArgT>
  LogicalResult notifyMatchFailure(ArgT &&arg, const Twine &msg) {
    return rewriter.notifyMatchFailure(std::forward<ArgT>(arg), msg);
  }
  IntegerAttr getIndexAttr(int64_t value) {
    return rewriter.getIndexAttr(value);
  }
  IntegerType getIntegerType(unsigned width, bool isSigned) {
    return rewriter.getIntegerType(width, isSigned);
  }
  template <typename Ty, typename... Args>
  Ty getType(Args &&...args) {
    return rewriter.getType<Ty>(std::forward<Args>(args)...);
  }
  template <typename Attr, typename... Args>
  Attr getAttr(Args &&...args) {
    return rewriter.getAttr<Attr>(std::forward<Args>(args)...);
  }
  void setInsertionPoint(Operation *op) { rewriter.setInsertionPoint(op); }
  void replaceOp(Operation *op, ValueRange newValues) {
    rewriter.replaceOp(op, newValues);
  }
  void replaceOp(Operation *op, Operation *newOp) {
    rewriter.replaceOp(op, newOp);
  }
  void replaceAllUsesWith(Value from, Value to) {
    rewriter.replaceAllUsesWith(from, to);
  }
  void replaceAllUsesWith(ValueRange from, ValueRange to) {
    rewriter.replaceAllUsesWith(from, to);
  }
  void setInsertionPointAfter(Operation *op) {
    rewriter.setInsertionPointAfter(op);
  }
  void replaceAllUsesExcept(Value from, Value to, Operation *exceptedUser) {
    rewriter.replaceAllUsesExcept(from, to, exceptedUser);
  }
  Operation *clone(Operation &op) { return rewriter.clone(op); }
  void eraseOp(Operation *op) { rewriter.eraseOp(op); }
  Operation *create(const OperationState &state) {
    return rewriter.create(state);
  }
  IndexType getIndexType() { return rewriter.getIndexType(); }
  BoolAttr getBoolAttr(bool value) { return rewriter.getBoolAttr(value); }
  NamedAttribute getNamedAttr(StringRef name, Attribute val) {
    return rewriter.getNamedAttr(name, val);
  }
  DictionaryAttr getDictionaryAttr(ArrayRef<NamedAttribute> value) {
    return rewriter.getDictionaryAttr(value);
  }
  DenseIntElementsAttr getI64VectorAttr(ArrayRef<int64_t> values) {
    return rewriter.getI64VectorAttr(values);
  }
};

//===----------------------------------------------------------------------===//
// DeviceAwarePattern
//===----------------------------------------------------------------------===//

/// TODO: clean this up, ask C++ geniuses
/// A drop in replacement for the likes of `OpRewritePattern`,
/// `OpInterfaceRewritePattern` and the like. Which automatically uses
/// `DeviceAwareRewriter`.
///
/// For example `RewritePatternT` might be `OpRewritePattern<AddOp>`
/// and `OpT` would be `AddOp`.
///
/// Example of use:
///
/// ```
/// struct MyRewrite : public OpInterfaceRewritePattern<Reduction> {
///   LogicalResult
///   matchAndRewrite(Reduction op,
///                   PatternRewriter &commonRewriter) const override {
///     ...
///   }
/// ...
/// };
/// ```
///
/// Would become:
///
/// ```
/// struct MyRewrite :
///   public DeviceAwarePattern<OpInterfaceRewritePattern, Reduction>  {
///
///   LogicalResult
///   deviceAwareMatchAndRewrite(Reduction op,
///                              DeviceAwareRewriter &commonRewriter)
///                              const override {
///     ...
///   }
/// ...
/// };
/// ```
///
/// Which will automatically propagate device information correctly.
template <typename RewritePatternT, typename OpT>
struct _DeviceAwarePatternShim : public RewritePatternT {
  using RewritePatternT::RewritePatternT;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &commonRewriter) const final {
    DeviceAwareRewriter rewriter(op, commonRewriter);
    return deviceAwareMatchAndRewrite(op, rewriter);
  }

  virtual LogicalResult
  deviceAwareMatchAndRewrite(OpT op, DeviceAwareRewriter &rewriter) const = 0;
};

/// Forwards template parameters to `RewritePatternBaseT`, done to avoid
/// uglier syntax.
template <template <typename> typename RewritePatternBaseT, typename OpT>
struct DeviceAwareRewritePattern
    : _DeviceAwarePatternShim<RewritePatternBaseT<OpT>, OpT> {
  using _DeviceAwarePatternShim<RewritePatternBaseT<OpT>,
                                OpT>::_DeviceAwarePatternShim;
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MODEVICES_H
