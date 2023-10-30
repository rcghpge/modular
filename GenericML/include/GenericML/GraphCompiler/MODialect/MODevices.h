//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MODEVICES_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MODEVICES_H

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "mlir/IR/PatternMatch.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Returns the #M.device_ref annotation for op, or the 'host' device
/// if there is no such annotation on the op itself.
DeviceRefAttr getDeviceOrDefault(MOHasDeviceInterface op);

/// As above, but for any operation.
DeviceRefAttr getDeviceOrDefaultAny(Operation *op);

/// Returns true if op is annotated to execute via the host. In this case
/// the op will have both a host and device-specific implementation.
bool isViaHostAny(Operation *op);

/// Returns the #M.device_ref annotation for type which appears in the
/// the given MO graphOp. Returns the 'host' device if there is no such
/// annotation on the type itself.
DeviceRefAttr getDeviceOrDefault(GraphOp graphOp, TensorType type);

/// Returns the #M.device_ref for the 'host' device of graphOp from its device
/// specs annotation, or <"cpu", 0> if the graph has no such annotation.
DeviceRefAttr getHostDevice(GraphOp graphOp);

/// Returns true if op should execute on the 'host'.
bool isOnHost(Operation *op);

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
class DeviceAnnots {
public:
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

  /// Returns true if this and that agree in their annotations where they
  /// are given.
  bool isConsistentWith(const DeviceAnnots &that) const;

private:
  /// The 'device' attribute of the original op, or null.
  DeviceRefAttr device;
  /// True if original op had the 'is_via_cpu' attribute.
  bool isViaCpu = false;
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

  /// Constructs device-aware rewriter using rewriter, and adds the initial
  /// operation to it, on the assumption that op will be replaced by the
  /// rewriter.
  DeviceAwareRewriter(Operation *op, mlir::RewriterBase &rewriter)
      : deviceAnnots(DeviceAnnots::forOp(op)), rewriter(rewriter) {}

  /// Captures the device annotations for operation in this, on the assumption
  /// the operation will be replaced by the rewriter. Returns false if this and
  /// op have conflicting annotations. No-op if op is null.
  [[nodiscard]] bool addOp(Operation *op) { return deviceAnnots.addOp(op); }

  /// Captures the device annotations for all operations, on the assumption
  /// they will all be replaced by the rewriter. Returns false if any ops
  /// have conflicting annotations. No-op for null ops.
  [[nodiscard]] bool addOps(const std::initializer_list<Operation *> &ops) {
    return llvm::all_of(
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
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MODEVICES_H
