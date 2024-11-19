// RUN: mo-opt --split-input-file --verify-diagnostics --assign-devices="target-device=cpu default-device-target-triple=arm64-apple-darwin23.6.0 default-device-target-arch=aarch64" --mo-to-mgp %s

// RUN: mo-opt --split-input-file --verify-diagnostics --assign-devices="target-device=cpu default-device-target-triple=x86_64-unknown-linux default-device-target-arch=x86_64" --mo-to-mgp %s

mo.graph @cast<dim>(%arg0: !mo.tensor<[dim], f32>) -> !mo.tensor<[dim], f16> attributes {argument_names = ["input0"], result_names = ["output0"]} {
  // expected-error @below {{The f16 data type is not supported on device 'cpu:0'.}}
  %0 = mo.cast(%arg0) : (!mo.tensor<[dim], f32>) -> !mo.tensor<[dim], f16>
  mo.output %0 : !mo.tensor<[dim], f16>
}

// -----

// expected-error @below {{The f16 data type is not supported on device 'cpu:0'.}}
mo.graph @cast<dim>(%arg0: !mo.tensor<[dim], f16>) -> !mo.tensor<[dim], f32> attributes {argument_names = ["input0"], result_names = ["output0"]} {
  %0 = mo.cast(%arg0) : (!mo.tensor<[dim], f16>) -> !mo.tensor<[dim], f32>
  mo.output %0 : !mo.tensor<[dim], f32>
}
