// RUN: mgp-opt %s --split-input-file --verify-diagnostics

//
// mgp.cpu.buffer.constant
//

func.func @buffer_constant_mismatched_size() {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x000102"> // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%0' expects different type than prior uses: '!mgp.cpu.buffer<4>' vs '!mgp.cpu.buffer<3>'}}
  bef.output %0 : !mgp.cpu.buffer<4>
}

// -----

func.func @buffer_constant_mismatched_align() {
  // expected-error@+1 {{'mgp.cpu.buffer.constant' op the resource size does not match the attribute's element size (4 vs 5)}}
  %0 = mgp.cpu.buffer.constant dense_resource<resource> : !mgp.cpu.buffer<5>
  bef.output %0 : !mgp.cpu.buffer<5>
}

{-#
  dialect_resources: {
    builtin: {
      resource:
        // First 4 bytes are alignment in little-endian form.
        "0x4000000000010203"
    }
  }
#-}

// -----

func.func @buffer_constant_invalid_elem_type() {
  // expected-error@+1 {{'mgp.cpu.buffer.constant' op only supports attributes of uint8_t elements (have 'i16')}}
  %0 = mgp.cpu.buffer.constant array<i16: 1, 2, 3, 4>
}

// -----

//
// mgp.cpu.buffer.alloc.spec
//

func.func @buffer_alloc_static_spec(%ctx : !mgp.context, %spec : !mgp.cpu.tensor_spec<4x8xf32>) {
  // expected-error@+1 {{'mgp.cpu.buffer.alloc.spec' op expected tensor spec to not have statically known size (have '!mgp.cpu.tensor_spec<4x8xf32>').}}
  %0 = mgp.cpu.buffer.alloc.spec %ctx %spec align 64 : !mgp.cpu.tensor_spec<4x8xf32>
  bef.output %0 : !mgp.cpu.buffer
}

// -----

//
// mgp.cpu.tensor_spec.create{.dynamic}
//

func.func @spec_create_mismatched_dims(%dim : index) {
  // expected-error@+1 {{'mgp.cpu.tensor_spec.create' op expected 2 dimension operands to match '!mgp.cpu.tensor_spec<?x64x?xf32>', but given 1.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.cpu.tensor_spec<?x64x?xf32>
  bef.output %0 : !mgp.cpu.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_bogus_type(%dim : index) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create' expected a !mgp.cpu.tensor_spec type.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.cpu.tensor<?x64xf32>
  bef.output %0 : !mgp.cpu.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_unknown_elem_type(%dim : index) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create' expected !mgp.cpu.tensor_spec type to have a known element type.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.cpu.tensor_spec<?x64xunk>
  bef.output %0 : !mgp.cpu.tensor_spec<?x64xunk>
}

// -----

func.func @spec_create_dynamic_mismatched_dims(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{'mgp.cpu.tensor_spec.create.dynamic' op expected 2 dimension operands to match '!mgp.cpu.tensor_spec<?x64x?xunk>', but given 1.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype %dim : !mgp.cpu.tensor_spec<?x64x?xunk>
  bef.output %0 : !mgp.cpu.tensor_spec<?x64x?xunk>
}

// -----

func.func @spec_create_dynamic_bogus_type(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create.dynamic' expected a !mgp.cpu.tensor_spec type.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype %dim : !mgp.cpu.tensor<?x64xf32>
  bef.output %0 : !mgp.cpu.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_dynamic_known_elem_type(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create.dynamic' expected !mgp.cpu.tensor_spec type to have an unknown element type.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype %dim : !mgp.cpu.tensor_spec<?x64xf32>
  bef.output %0 : !mgp.cpu.tensor_spec<?x64xf32>
}

// -----

//
// mgp.cpu.tensor_spec.dims.dynamic
//

func.func @spec_dims_mismatched_dims(%spec : !mgp.cpu.tensor_spec<?x64x?xf32>) {
  // expected-error@+1 {{operation defines 2 results but was provided 1 to bind}}
  %dim0 = mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.cpu.tensor_spec<?x64x?xf32>
  bef.output %dim0 : index
}

// -----

//
// mgp.cpu.tensor.create
//

func.func @tensor_create_mismatched_size(%ch : !bef.chain, %buf : !mgp.cpu.buffer<61>, %spec : !mgp.cpu.tensor_spec<3x5xf32>) { // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%buf' expects different type than prior uses: '!mgp.cpu.buffer<60>' vs '!mgp.cpu.buffer<61>'}}
  %tensor = mgp.cpu.tensor.create %ch %buf %spec : !mgp.cpu.tensor<3x5xf32>
  bef.output %tensor : !mgp.cpu.tensor<3x5xf32>
}

// -----

//
// mgp.cpu.tensor.extract
//

func.func @tensor_extract_mismatched_size(%tensor : !mgp.cpu.tensor<3x5xf32>) {
  %buf, %spec = mgp.cpu.tensor.extract %tensor : !mgp.cpu.tensor<3x5xf32>  // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%buf' expects different type than prior uses: '!mgp.cpu.buffer<61>' vs '!mgp.cpu.buffer<60>'}}
  bef.output %buf, %spec : !mgp.cpu.buffer<61>, !mgp.cpu.tensor_spec<3x5xf32>
}

// -----

//
// !mgp.cpu.tensor_spec
//

// expected-error@+1 {{shape must have rank between 1 and 5.}}
func.func @spec_too_large(%x : !mgp.cpu.tensor_spec<1x2x3x4x5x6xf32>) {}

// -----

// expected-error@+1 {{shape dimensions must be strictly positive or the distinguished '?' unknown dimension.}}
func.func @spec_zero_dim(%x : !mgp.cpu.tensor_spec<1x0x3xf32>) {}

// -----

// expected-error@+1 {{unrecognized dtype 'ui33'.}}
func.func @dtype_unsupported(%x : !mgp.cpu.tensor_spec<5x10xui33>) {}

// -----

// expected-error@+1 {{invalid dtype.}}
func.func @dtype_invalid(%x : !mgp.cpu.tensor_spec<5x10xinvalid>) {}

// -----

// expected-error@+1 {{unable to represent DType 'f8' as an MLIR type.}}
func.func @dtype_unrepresentable(%x : !mgp.cpu.tensor_spec<5x10xf8>) {}

// -----

//
// !mgp.cpu.tensor
//

// expected-error@+1 {{shape dimensions must be strictly positive or the distinguished '?' unknown dimension}}
func.func @tensor_invalid(%x : !mgp.cpu.tensor<3x0x27xf32>) {}

// -----

//
// mgp.model
//

// expected-error@+1 {{'mgp.model' op expected no arguments to the 'init' body, but got '!bef.chain'.}}
"mgp.model"() ({
  ^bb0(%ch: !bef.chain):
    %ctx = mgp.context.create 1
    bef.output %ch, %ctx : !bef.chain, !mgp.context
}, {
  ^bb0(%ch: !bef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    bef.output %ch, %x : !bef.chain, tensor<3xf32>
}) {frameworkName = "tfl", inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badSetupArgs"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the result types of the 'init' body to be '!bef.chain', '!mgp.context'.}}
"mgp.model"() ({
  %ch = grt.chain.create
  %ctx = mgp.context.create 1
  // expected-error@+1 {{'bef.output' op expected 2 operands for enclosing op}}
  bef.output %ch : !bef.chain
}, {
  ^bb0(%ch: !bef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    bef.output %ch, %x : !bef.chain, tensor<3xf32>
}) {frameworkName = "tfl", inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badSetupResults"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the arguments of the 'execute' body to be '!bef.chain', '!mgp.context', 'tensor<3xf32>', but got '!bef.chain', 'tensor<3xf32>', '!mgp.context'.}}
"mgp.model"() ({
  %ch = grt.chain.create
  %ctx = mgp.context.create 1
  bef.output %ch, %ctx : !bef.chain, !mgp.context
}, {
  ^bb0(%ch: !bef.chain, %x: tensor<3xf32>, %ctx: !mgp.context):
    bef.output %ch, %x : !bef.chain, tensor<3xf32>
}) {frameworkName = "tfl", inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badExecuteArgs"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the arguments of the 'execute' body to be '!bef.chain', '!mgp.context', 'tensor<4xf32>', but got '!bef.chain', '!mgp.context', 'tensor<3xf32>'.}}
"mgp.model"() ({
  %ch = grt.chain.create
  %ctx = mgp.context.create 1
  bef.output %ch, %ctx : !bef.chain, !mgp.context
}, {
  ^bb0(%ch: !bef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    bef.output %ch, %x : !bef.chain, tensor<3xf32>
}) {frameworkName = "tfl", inputOutputTypes = (tensor<4xf32>) -> (tensor<3xf32>), sym_name = "badExecuteArgs2"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the result types of the 'execute' body to be '!bef.chain'.}}
"mgp.model"() ({
  %ch = grt.chain.create
  %ctx = mgp.context.create 1
  bef.output %ch, %ctx : !bef.chain, !mgp.context
}, {
  ^bb0(%ch: !bef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    // expected-error@+1 {{'bef.output' op expected 1 operands for enclosing op}}
    bef.output %ch, %x : !bef.chain, tensor<3xf32>
}) {frameworkName = "tfl", inputOutputTypes = (tensor<3xf32>) -> (), sym_name = "badExecuteResults"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the result types of the 'init' body to be '!bef.chain', '!mgp.context'.}}
mgp.model @mismatchedContextType : tfl init {
  %ch = grt.chain.create
  %unk = bef.opaque "bogus" [] () : ()->!bef.unknown
  // expected-error@+1 {{'bef.output' op operand #1 has type '!bef.unknown' but should be '!mgp.context'}}
  bef.output %ch, %unk : !bef.chain, !bef.unknown
} execute (%ch: !bef.chain, %ctx: !mgp.context, %x: tensor<2xf32>) -> (!bef.chain, tensor<2xf32>) {
  bef.output %ch, %x : !bef.chain, tensor<2xf32>
}

// -----

// expected-error@+1 {{'mgp.model' op expected the 'execute' entry point to take arguments of ShapedType, but argument 0 is of type 'index'.}}
mgp.model @invalidExecuteArgType : tfl init {
  %ch = grt.chain.create
  %ctx = mgp.context.create 1
  bef.output %ch, %ctx : !bef.chain, !mgp.context
} execute (%ch: !bef.chain, %ctx: !mgp.context, %x: index) -> (!bef.chain, index) {
  bef.output %ch, %x : !bef.chain, index
}

// -----

// expected-error@+1 {{'mgp.model' op attribute 'frameworkName' failed to satisfy constraint: string attribute must be a recognized framework name.}}
mgp.model @invalidExecuteArgType : bogus init {
  %ch = grt.chain.create
  %ctx = mgp.context.create 1
  bef.output %ch, %ctx : !bef.chain, !mgp.context
} execute (%ch: !bef.chain, %ctx: !mgp.context, %x: index) -> (!bef.chain, index) {
  bef.output %ch, %x : !bef.chain, index
}
