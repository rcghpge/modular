// RUN: mgp-opt %s --split-input-file --verify-diagnostics

//
// mgp.cpu.buffer.constant
//

func.func @buffer_constant_mismatched_size() {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x000102", align 32> // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%0' expects different type than prior uses: '!mgp.cpu.buffer<4>' vs '!mgp.cpu.buffer<3>'}}
  mef.output %0 : !mgp.cpu.buffer<4>
}

// -----

func.func @buffer_constant_invalid_elem_type() {
  // expected-error@+1 {{'mgp.cpu.buffer.constant' op only supports attributes of uint8_t elements (have 'i16')}}
  %0 = mgp.cpu.buffer.constant #M.dense_array<1, 2, 3, 4> : !M.array<4xi16>
}

// -----

//
// mgp.cpu.buffer.alloc.spec
//

func.func @buffer_alloc_static_spec(%ch : !mef.chain, %spec : !mgp.cpu.tensor_spec<4x8xf32>) {
  // expected-error@+1 {{'mgp.cpu.buffer.alloc.spec' op expected tensor spec to not have statically known size (have '!mgp.cpu.tensor_spec<4x8xf32>').}}
  %0 = mgp.cpu.buffer.alloc.spec %ch, %spec, align 64 : !mgp.cpu.tensor_spec<4x8xf32>
  mef.output %0 : !mgp.cpu.buffer
}

// -----

func.func @buffer_alloc_dynamic_rank_spec(%ch : !mef.chain, %spec : !mgp.cpu.tensor_spec<*xf32>) {
  // expected-error@+1 {{'mgp.cpu.buffer.alloc.spec' op expected tensor spec to have statically known rank.}}
  %0 = mgp.cpu.buffer.alloc.spec %ch, %spec, align 64 : !mgp.cpu.tensor_spec<*xf32>
  mef.output %0 : !mgp.cpu.buffer
}

// -----

//
// mgp.cpu.tensor_spec.create{.dynamic}
//

func.func @spec_create_mismatched_dims(%dim : index) {
  // expected-error@+1 {{'mgp.cpu.tensor_spec.create' op expected 2 dimension operands to match '!mgp.cpu.tensor_spec<?x64x?xf32>', but given 1.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.cpu.tensor_spec<?x64x?xf32>
  mef.output %0 : !mgp.cpu.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_bogus_type(%dim : index) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create' expected a !mgp.cpu.tensor_spec type.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.cpu.tensor<?x64xf32>
  mef.output %0 : !mgp.cpu.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_unknown_elem_type(%dim : index) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create' expected !mgp.cpu.tensor_spec type to have a known element type.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.cpu.tensor_spec<?x64xunk>
  mef.output %0 : !mgp.cpu.tensor_spec<?x64xunk>
}

// -----

func.func @spec_create_dynamic_mismatched_dims(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{'mgp.cpu.tensor_spec.create.dynamic' op expected 2 dimension operands to match '!mgp.cpu.tensor_spec<?x64x?xunk>', but given 1.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype, %dim : !mgp.cpu.tensor_spec<?x64x?xunk>
  mef.output %0 : !mgp.cpu.tensor_spec<?x64x?xunk>
}

// -----

func.func @spec_create_dynamic_bogus_type(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create.dynamic' expected a !mgp.cpu.tensor_spec type.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype, %dim : !mgp.cpu.tensor<?x64xf32>
  mef.output %0 : !mgp.cpu.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_dynamic_known_elem_type(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create.dynamic' expected !mgp.cpu.tensor_spec type to have an unknown element type.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype, %dim : !mgp.cpu.tensor_spec<?x64xf32>
  mef.output %0 : !mgp.cpu.tensor_spec<?x64xf32>
}

// -----

func.func @spec_create_dynamic_rank(%dim : index) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create' expected !mgp.cpu.tensor_spec type to have a statically known rank.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.cpu.tensor_spec<*xf32>
  mef.output %0 : !mgp.cpu.tensor_spec<*xf32>
}

// -----

func.func @spec_create_dynamic_dynamic_rank(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create.dynamic' expected !mgp.cpu.tensor_spec type to have a statically known rank.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype, %dim : !mgp.cpu.tensor_spec<*xunk>
  mef.output %0 : !mgp.cpu.tensor_spec<*xunk>
}

// -----

//
// mgp.cpu.tensor_spec.dims.dynamic
//

func.func @spec_dims_mismatched_dims(%spec : !mgp.cpu.tensor_spec<?x64x?xf32>) {
  // expected-error@+1 {{operation defines 2 results but was provided 1 to bind}}
  %dim0 = mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.cpu.tensor_spec<?x64x?xf32>
  mef.output %dim0 : index
}

// -----

func.func @spec_dims_dynamic_rank(%spec : !mgp.cpu.tensor_spec<*xf32>) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.dims.dynamic' expected !mgp.cpu.tensor_spec type to have a statically known rank.}}
  %dim0 = mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.cpu.tensor_spec<*xf32>
  mef.output %dim0 : index
}

// -----

//
// mgp.cpu.tensor.create
//

func.func @tensor_create_mismatched_size(%ch : !mef.chain, %buf : !mgp.cpu.buffer<61>, %spec : !mgp.cpu.tensor_spec<3x5xf32>) { // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%buf' expects different type than prior uses: '!mgp.cpu.buffer<60>' vs '!mgp.cpu.buffer<61>'}}
  %tensor = mgp.cpu.tensor.create %ch, %buf, %spec : !mgp.cpu.tensor<3x5xf32>
  mef.output %tensor : !mgp.cpu.tensor<3x5xf32>
}

// -----

func.func @tensor_create_mismatched_size_with_buf_annot(%ch : !mef.chain, %buf : !mgp.cpu.buffer<61>, %spec : !mgp.cpu.tensor_spec<3x5xf32>) { // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%buf' expects different type than prior uses: '!mgp.cpu.buffer<60>' vs '!mgp.cpu.buffer<61>'}}
  %tensor = mgp.cpu.tensor.create %ch, %buf, %spec !mgp.cpu.buffer<60> : !mgp.cpu.tensor<3x5xf32>
  mef.output %tensor : !mgp.cpu.tensor<3x5xf32>
}

// -----

func.func @tensor_create_mismatched_buf_and_spec_with_buf_annot(%ch : !mef.chain, %buf : !mgp.cpu.buffer<61>, %spec : !mgp.cpu.tensor_spec<3x5xf32>) {
  // expected-error@+1 {{'mgp.cpu.tensor.create' op expected buffer type to match type implied by tensor spec ('!mgp.cpu.buffer<61>' vs '!mgp.cpu.buffer<60>')}}
  %tensor = mgp.cpu.tensor.create %ch, %buf, %spec !mgp.cpu.buffer<61> : !mgp.cpu.tensor<3x5xf32>
  mef.output %tensor : !mgp.cpu.tensor<3x5xf32>
}

// -----

//
// mgp.cpu.tensor.extract
//

func.func @tensor_extract_mismatched_size(%tensor : !mgp.cpu.tensor<3x5xf32>) {
  %buf, %spec = mgp.cpu.tensor.extract %tensor : !mgp.cpu.tensor<3x5xf32>  // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%buf' expects different type than prior uses: '!mgp.cpu.buffer<61>' vs '!mgp.cpu.buffer<60>'}}
  mef.output %buf, %spec : !mgp.cpu.buffer<61>, !mgp.cpu.tensor_spec<3x5xf32>
}

// -----

//
// !mgp.cpu.tensor_spec
//

// expected-error@+1 {{tensor spec must have a rank between 0 and 8.}}
func.func @spec_too_large(%x : !mgp.cpu.tensor_spec<1x2x3x4x5x6x7x8x9xf32>) {}

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
// mgp.cpu.buffer.concat
//

func.func @buffer_concat_sizes(%ch0 : !mef.chain, %out_buf : !mgp.cpu.buffer<192>, %buf0 : !mgp.cpu.buffer<64>, %buf1 : !mgp.cpu.buffer<256>) {
// expected-error@+1 {{'mgp.cpu.buffer.concat' op expected output buffer type does not match actual output buffer type ('!mgp.cpu.buffer<320>' vs '!mgp.cpu.buffer<192>').}}
  %ch1 = mgp.cpu.buffer.concat %ch0, %out_buf, %buf0, %buf1 : (!mgp.cpu.buffer<64>, !mgp.cpu.buffer<256>) -> !mgp.cpu.buffer<192>
  mef.output %ch1 : !mef.chain
}

// -----

//
// mgp.cpu.buffer.section.borrow.static
//

func.func @buffer_section_borrow_static_beyond_end(%ch0 : !mef.chain, %in_buf : !mgp.cpu.buffer<192>) {
// expected-error@+1 {{'mgp.cpu.buffer.section.borrow.static' op requesting section end 256 which is beyond end of input buffer of size 192.}}
  %buf = mgp.cpu.buffer.section.borrow.static %ch0, %in_buf, offset 128, size 128: !mgp.cpu.buffer<192>
  mef.output %buf : !mgp.cpu.buffer<128>
}

// -----

func.func @buffer_section_borrow_static_unsized(%ch0 : !mef.chain, %in_buf : !mgp.cpu.buffer) {
// expected-error@+1 {{'mgp.cpu.buffer.section.borrow.static' op expected input buffer of type ''!mgp.cpu.buffer'' to have a statically known size.}}
  %buf = mgp.cpu.buffer.section.borrow.static %ch0, %in_buf, offset 128, size 128: !mgp.cpu.buffer
  mef.output %buf : !mgp.cpu.buffer<128>
}

// -----

//
// mgp.cpu.buffer.section.borrow.dynamic
//

func.func @buffer_section_borrow_dynamic_sized(%ch0 : !mef.chain, %in_buf : !mgp.cpu.buffer<192>, %scale : index) {
// expected-error@+1 {{'mgp.cpu.buffer.section.borrow.dynamic' op expected input buffer of type ''!mgp.cpu.buffer<192>'' to have a statically unknown size.}}
  %buf = mgp.cpu.buffer.section.borrow.dynamic %ch0, %in_buf, %scale, offset 128, size 128: !mgp.cpu.buffer<192>
  mef.output %buf : !mgp.cpu.buffer
}

// -----

//
// mgp.cpu.buffer.split.borrow
//

func.func @buffer_split_sizes(%ch0 : !mef.chain, %in_buf : !mgp.cpu.buffer<192>) {
// expected-error@+1 {{'mgp.cpu.buffer.split.borrow' op expected input buffer type does not match actual input buffer type ('!mgp.cpu.buffer<320>' vs '!mgp.cpu.buffer<192>').}}
  %buf0, %buf1 = mgp.cpu.buffer.split.borrow %ch0, %in_buf : !mgp.cpu.buffer<192> -> (!mgp.cpu.buffer<64>, !mgp.cpu.buffer<256>)
  mef.output %buf0, %buf1 : !mgp.cpu.buffer<64>, !mgp.cpu.buffer<256>
}

// -----

func.func @buffer_split_operands(%ch0 : !mef.chain, %in_buf : !mgp.cpu.buffer) {
// expected-error@+1 {{'mgp.cpu.buffer.split.borrow' op expected a runtime size operand for every output buffer without a statically known size (given 0 but need 1).}}
  %buf0, %buf1 = mgp.cpu.buffer.split.borrow %ch0, %in_buf : !mgp.cpu.buffer -> (!mgp.cpu.buffer<64>, !mgp.cpu.buffer)
  mef.output %buf0, %buf1 : !mgp.cpu.buffer<64>, !mgp.cpu.buffer
}

// -----

//
// mgp.model
//

// expected-error@+1 {{'mgp.model' op expected no arguments to the 'init' body, but got '!mef.chain'.}}
"mgp.model"() ({
  ^bb0(%ch: !mef.chain):
    %ctx = mgp.context.create 1
    mef.output %ch, %ctx : !mef.chain, !mgp.context
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tfl", inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badSetupArgs"} : () -> ()

// -----

"mgp.model"() ({
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  // expected-error@+1 {{'mef.output' op expected 2 operands, but given 1}}
  mef.output %ch : !mef.chain
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tfl", inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badSetupResults"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the arguments of the 'execute' body to be '!mef.chain', '!mgp.context', 'tensor<3xf32>', but got '!mef.chain', 'tensor<3xf32>', '!mgp.context'.}}
"mgp.model"() ({
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx : !mef.chain, !mgp.context
}, {
  ^bb0(%ch: !mef.chain, %x: tensor<3xf32>, %ctx: !mgp.context):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tfl", inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badExecuteArgs"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the arguments of the 'execute' body to be '!mef.chain', '!mgp.context', 'tensor<4xf32>', but got '!mef.chain', '!mgp.context', 'tensor<3xf32>'.}}
"mgp.model"() ({
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx : !mef.chain, !mgp.context
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tfl", inputOutputTypes = (tensor<4xf32>) -> (tensor<3xf32>), sym_name = "badExecuteArgs2"} : () -> ()

// -----

"mgp.model"() ({
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx : !mef.chain, !mgp.context
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    // expected-error@+1 {{'mef.output' op expected 1 operands, but given 2}}
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tfl", inputOutputTypes = (tensor<3xf32>) -> (), sym_name = "badExecuteResults"} : () -> ()

// -----

mgp.model @mismatchedContextType : tfl init {
  %ch = grt.chain.create ()
  %unk = mef.opaque "bogus" [] () : ()->!mef.unknown
  // expected-error@+1 {{'mef.output' op operand #1 has type '!mef.unknown' but expected '!mgp.context'}}
  mef.output %ch, %unk : !mef.chain, !mef.unknown
} execute (%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<2xf32>) -> (!mef.chain, tensor<2xf32>) {
  mef.output %ch, %x : !mef.chain, tensor<2xf32>
}

// -----

// expected-error@+1 {{'mgp.model' op expected the 'execute' entry point to take arguments of ShapedType, but argument 0 is of type 'index'.}}
mgp.model @invalidExecuteArgType : tfl init {
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx : !mef.chain, !mgp.context
} execute (%ch: !mef.chain, %ctx: !mgp.context, %x: index) -> (!mef.chain, index) {
  mef.output %ch, %x : !mef.chain, index
}

// -----

// expected-error@+1 {{'mgp.model' op attribute 'frameworkName' failed to satisfy constraint: string attribute must be a recognized framework name.}}
mgp.model @invalidExecuteArgType : bogus init {
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx : !mef.chain, !mgp.context
} execute (%ch: !mef.chain, %ctx: !mgp.context, %x: index) -> (!mef.chain, index) {
  mef.output %ch, %x : !mef.chain, index
}

//
// mgp.to_grt_bool
//

// -----

func.func @to_grt_bool(%ch0 : !mef.chain, %buf : !mgp.cpu.buffer<2>) {
  // expected-error@+1 {{'mgp.to_grt_bool' op operand #1 must be scalar buffer, but got '!mgp.cpu.buffer<2>'}}
  %0 = mgp.to_grt_bool(%ch0, %buf) : (!mef.chain, !mgp.cpu.buffer<2>) -> i1
  mef.output %0 : i1
}
