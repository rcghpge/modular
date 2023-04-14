//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the QuantSpec and related classes, which hold
// quantization information for tensors in a framework-independent way.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_QUANTSPEC_H
#define GENERICML_SUPPORT_QUANTSPEC_H

#include "Support/ML/TensorSpec.h"

namespace M {
class MEFType;

/// This class is a framework-independent abstraction for quantization info for
/// a single tensor.
class QuantSpec {
public:
  enum Kind : uint8_t {
    invalid,
    notQuantized,
    uniform,
    uniformPerAxis,
  };

  /// This struct represents affine quantization parameters, i.e. the parameters
  /// that map a quantized value `v_q` of the storage type into a (float) value
  /// `v` of the expressed type, following the rule
  ///   `v = (v_q - zeroPoint) * scale`.
  /// The `scale` field can be interpreted as the increment in `v` when `v_q` is
  /// incremented by 1. The `zeroPoint` value is typically (but not necessarily)
  /// within the range of the storage type, and in that case it represents
  /// `v_q = 0.0`.
  struct ScaleZeroPointPair {
    float scale;
    int16_t zeroPoint;
  };

  QuantSpec() : kind(notQuantized) {}
  QuantSpec(Kind kind, uint8_t quantDim, ArrayRef<ScaleZeroPointPair> params)
      : kind(kind), quantDim(quantDim), params(params.begin(), params.end()) {}

  /// Creates a QuantSpec of uniform kind from a single scale and zero point.
  static inline QuantSpec getUniform(float scale, int16_t zeroPoint) {
    return QuantSpec(Kind::uniform, 0, {{scale, zeroPoint}});
  }

  /// Creates a QuantSpec of uniformPerAxis kind of separate arrays of scales
  /// and zero points.
  static QuantSpec getUniformPerAxis(uint8_t quantDim, ArrayRef<float> scales,
                                     ArrayRef<int64_t> zeroPoints);

  /// Returns a string representation of the kind enum.
  static StringRef kindAsString(QuantSpec::Kind kind);

  /// Returns a QuantSpec::Kind enum given a string representation.
  static QuantSpec::Kind kindFromString(StringRef kind);

  /// Returns the paired scales and zero points.
  ArrayRef<ScaleZeroPointPair> getParams() const { return params; }

  // Returns the kind of the QuantSpec.
  Kind getKind() const { return kind; }

  /// The quantized dimension is undefined when kind is invalid or notQuantized.
  uint8_t getQuantDim() const { return quantDim; }

  /// Returns false if the QuantSpec is invalid, and true otherwise.
  bool isValid() const { return kind != Kind::invalid; }

  /// Returns false if the QuantSpec is invalid or not quantized.
  bool isQuantized() const { return isValid() && kind != notQuantized; }

  std::string getAsString() const;
  void print(raw_ostream &os) const;
  void dump() const;

private:
  Kind kind;
  uint8_t quantDim = 0;

  // We optimize for the case when kind is uniform, and there is only one scale
  // and one zero point.
  SmallVector<ScaleZeroPointPair, 1> params;
};

inline raw_ostream &operator<<(raw_ostream &os, const QuantSpec &value) {
  value.print(os);
  return os;
}

/// This class extends TensorSpec to include potential quantization information.
/// Since QuantSpec can be of notQuantized kind, QuantTensorSpec can describe
/// ordinary floating point (i.e. not quantized) tensors.
class QuantTensorSpec : public TensorSpec {
private:
  QuantTensorSpec(const QuantTensorSpec &spec, const TensorShape &newShape)
      : TensorSpec(newShape, spec.getEltType()),
        quantSpec(spec.getQuantSpec()) {}

public:
  QuantTensorSpec() : TensorSpec() { setEltType(DType::invalid); }
  QuantTensorSpec(TensorSpec &&tensorSpec) : TensorSpec(tensorSpec) {}
  QuantTensorSpec(TensorSpec &&tensorSpec, QuantSpec &&quantSpec)
      : TensorSpec(tensorSpec), quantSpec(quantSpec) {}

  /// Return the underlying QuantSpec.
  const QuantSpec &getQuantSpec() const { return quantSpec; }

  /// Returns true if QuantTensorSpec describes a truly quantized tensor.
  bool isQuantized() const { return quantSpec.isQuantized(); }

  /// Returns new QuantTensorSpec with identical attributes save for newShape.
  QuantTensorSpec cloneWithNewShape(const TensorShape &newShape) const {
    return QuantTensorSpec(*this, newShape);
  }

  /// Helper method to decode a MEFType into a TensorSpec, if possible.
  static ErrorOr<QuantTensorSpec> get(MEFType type);

  std::string getAsString() const;
  void print(raw_ostream &os) const;
  void dump() const;

private:
  QuantSpec quantSpec;
};

inline raw_ostream &operator<<(raw_ostream &os, const QuantTensorSpec &value) {
  value.print(os);
  return os;
}

} // namespace M

#endif // GENERICML_SUPPORT_QUANTSPEC_H
