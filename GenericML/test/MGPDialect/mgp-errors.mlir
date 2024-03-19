// RUN: mo-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

//
// mgp.cpu.buffer.constant.deprecated
//

func.func @buffer_constant_mismatched_size() {
  %0 = mgp.cpu.buffer.constant.deprecated #M.aligned_bytes<"0x000102", align 32> // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%0' expects different type than prior uses: '!mgp.cpu.buffer<4>' vs '!mgp.cpu.buffer<3>'}}
  mef.output %0 : !mgp.cpu.buffer<4>
}

// -----

func.func @buffer_constant_invalid_elem_type() {
  // expected-error@+1 {{'mgp.cpu.buffer.constant.deprecated' op only supports attributes of uint8_t elements (have 'i16')}}
  %0 = mgp.cpu.buffer.constant.deprecated #M.dense_array<1, 2, 3, 4> : !M.array<4xi16>
}

// -----

//
// mgp.cpu.tensor_spec.create{.dynamic}
//

func.func @spec_create_mismatched_dims(%dim : index) {
  // expected-error@+1 {{'mgp.cpu.tensor_spec.create' op expected 2 dimension operands to match '!mgp.tensor_spec<?x64x?xf32>', but given 1.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.tensor_spec<?x64x?xf32>
  mef.output %0 : !mgp.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_bogus_type(%dim : index) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create' expected a !mgp.cpu.tensor_spec type.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.cpu.tensor<?x64xf32>
  mef.output %0 : !mgp.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_unknown_elem_type(%dim : index) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create' expected !mgp.cpu.tensor_spec type to have a known element type.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.tensor_spec<?x64xunk>
  mef.output %0 : !mgp.tensor_spec<?x64xunk>
}

// -----

func.func @spec_create_dynamic_mismatched_dims(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{'mgp.cpu.tensor_spec.create.dynamic' op expected 2 dimension operands to match '!mgp.tensor_spec<?x64x?xunk>', but given 1.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype, %dim : !mgp.tensor_spec<?x64x?xunk>
  mef.output %0 : !mgp.tensor_spec<?x64x?xunk>
}

// -----

func.func @spec_create_dynamic_bogus_type(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create.dynamic' expected a !mgp.cpu.tensor_spec type.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype, %dim : !mgp.cpu.tensor<?x64xf32>
  mef.output %0 : !mgp.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_dynamic_known_elem_type(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create.dynamic' expected !mgp.cpu.tensor_spec type to have an unknown element type.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype, %dim : !mgp.tensor_spec<?x64xf32>
  mef.output %0 : !mgp.tensor_spec<?x64xf32>
}

// -----

func.func @spec_create_dynamic_rank(%dim : index) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create' expected !mgp.cpu.tensor_spec type to have a statically known rank.}}
  %0 = mgp.cpu.tensor_spec.create %dim : !mgp.tensor_spec<*xf32>
  mef.output %0 : !mgp.tensor_spec<*xf32>
}

// -----

func.func @spec_create_dynamic_dynamic_rank(%dim : index, %dtype : !mgp.dtype) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.create.dynamic' expected !mgp.cpu.tensor_spec type to have a statically known rank.}}
  %0 = mgp.cpu.tensor_spec.create.dynamic %dtype, %dim : !mgp.tensor_spec<*xunk>
  mef.output %0 : !mgp.tensor_spec<*xunk>
}

// -----

//
// mgp.cpu.tensor_spec.with_dtype
//

func.func @spec_with_known_dtype(%shape : !mgp.tensor_spec<?x3xf32>) { //  expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%shape' expects different type than prior uses: '!mgp.tensor_spec<?x3xunk>' vs '!mgp.tensor_spec<?x3xf32>'}}
  %0 = mgp.cpu.tensor_spec.with_dtype %shape : !mgp.tensor_spec<?x3xf32>
  mef.output %0 : !mgp.tensor_spec<?x3xf32>
}

// -----

func.func @spec_with_unknown_dtype(%shape : !mgp.tensor_spec<?x3xunk>) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.with_dtype' expected an !mgp.cpu.tensor_spec type with known dtype.}}
  %0 = mgp.cpu.tensor_spec.with_dtype %shape : !mgp.tensor_spec<?x3xunk>
  mef.output %0 : !mgp.tensor_spec<?x3xunk>
}

// -----

//
// mgp.cpu.tensor_spec.dims.dynamic
//

func.func @spec_dims_mismatched_dims(%spec : !mgp.tensor_spec<?x64x?xf32>) {
  // expected-error@+1 {{operation defines 2 results but was provided 1 to bind}}
  %dim0 = mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.tensor_spec<?x64x?xf32>
  mef.output %dim0 : index
}

// -----

func.func @spec_dims_dynamic_rank(%spec : !mgp.tensor_spec<*xf32>) {
  // expected-error@+1 {{custom op 'mgp.cpu.tensor_spec.dims.dynamic' expected !mgp.cpu.tensor_spec type to have a statically known rank.}}
  %dim0 = mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.tensor_spec<*xf32>
  mef.output %dim0 : index
}

// -----

//
// !mgp.tensor_spec
//

// expected-error@+1 {{tensor spec must have a rank between 0 and 8.}}
func.func @spec_too_large(%x : !mgp.tensor_spec<1x2x3x4x5x6x7x8x9xf32>) {}

// -----

// expected-error@+1 {{unrecognized dtype 'ui33'.}}
func.func @dtype_unsupported(%x : !mgp.tensor_spec<5x10xui33>) {}

// -----

// expected-error@+1 {{invalid dtype.}}
func.func @dtype_invalid(%x : !mgp.tensor_spec<5x10xinvalid>) {}

// -----

// expected-error@+1 {{unable to represent DType 'f8' as an MLIR type.}}
func.func @dtype_unrepresentable(%x : !mgp.tensor_spec<5x10xf8>) {}

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
}) {frameworkName = "tf", inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badSetupArgs"} : () -> ()

// -----

"mgp.model"() ({
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  // expected-error@+1 {{'mef.output' op expected 2 operands, but given 1}}
  mef.output %ch : !mef.chain
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tf", inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badSetupResults"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the arguments of the 'execute' body to be '!mef.chain', '!mgp.context', 'tensor<3xf32>', but got '!mef.chain', 'tensor<3xf32>', '!mgp.context'.}}
"mgp.model"() ({
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx : !mef.chain, !mgp.context
}, {
  ^bb0(%ch: !mef.chain, %x: tensor<3xf32>, %ctx: !mgp.context):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tf", inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badExecuteArgs"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the arguments of the 'execute' body to be '!mef.chain', '!mgp.context', 'tensor<4xf32>', but got '!mef.chain', '!mgp.context', 'tensor<3xf32>'.}}
"mgp.model"() ({
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx : !mef.chain, !mgp.context
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tf", inputOutputTypes = (tensor<4xf32>) -> (tensor<3xf32>), sym_name = "badExecuteArgs2"} : () -> ()

// -----

"mgp.model"() ({
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx : !mef.chain, !mgp.context
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    // expected-error@+1 {{'mef.output' op expected 1 operands, but given 2}}
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tf", inputOutputTypes = (tensor<3xf32>) -> (), sym_name = "badExecuteResults"} : () -> ()

// -----

mgp.model @mismatchedContextType : tf init {
  %ch = grt.chain.create ()
  %unk = mef.opaque "bogus" [] () : ()->!mef.unknown
  // expected-error@+1 {{'mef.output' op operand #1 has type '!mef.unknown' but expected '!mgp.context'}}
  mef.output %ch, %unk : !mef.chain, !mef.unknown
} execute (%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<2xf32>) -> (!mef.chain, tensor<2xf32>) {
  mef.output %ch, %x : !mef.chain, tensor<2xf32>
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
// mgp.to_grt_bool.deprecated
//

// -----

func.func @to_grt_bool.deprecated(%ch0 : !mef.chain, %buf : !mgp.cpu.buffer<2>) {
  // expected-error@+1 {{'mgp.to_grt_bool.deprecated' op operand #1 must be scalar buffer, but got '!mgp.cpu.buffer<2>'}}
  %0 = mgp.to_grt_bool.deprecated(%ch0, %buf) : (!mef.chain, !mgp.cpu.buffer<2>) -> i1
  mef.output %0 : i1
}

// -----

//
// mgp.cpu.kernel.execute.deprecated
//

// CHECK-LABEL: @kernel_execute_invalid
func.func @kernel_execute_invalid(%ch0 : !mef.chain, %ctx : !mgp.context, %buf0 : !mgp.cpu.buffer, %buf1 : !mgp.cpu.buffer<8>, %dim0 : index, %dim1 : index) {
  // expected-error@+1 {{'mgp.cpu.kernel.execute.deprecated' op expected in/out signature attribute to have arity 2 but given 3.}}
  %ch1, %dim2 = "mgp.cpu.kernel.execute.deprecated"(%ch0, %ctx, %buf0, %buf1, %dim0, %dim1)
      { aSlot = 0 : ui64,
        bKernelIndex = 3 : ui64,
        cInOutSignature = #M.inout_sig<"ioo">,
        dSummary = "kernel",
        eNumOutDimensions = 1 : ui64,
        operandSegmentSizes = array<i32: 1, 1, 2, 2> }
      : (!mef.chain, !mgp.context, !mgp.cpu.buffer, !mgp.cpu.buffer<8>, index, index) -> (!mef.chain, index)
  mef.output %ch1, %dim2 : !mef.chain, index
}

// -----

//
// mgp.list.create
//

// CHECK-LABEL: @list_create_mismatch_element_type
func.func @list_create_mismatch_element_type(
  %t0 : !mgp.tensor<3x3xf32>
) {
  // expected-error@+1 {{input type must be a subtype of list element type: '!mgp.tensor<3x3xf32>' v.s. '!mgp.tensor<3x3xsi32>'}}
  %t = mgp.list.create %t0 : (!mgp.tensor<3x3xf32>) -> !mgp.list<!mgp.tensor<3x3xsi32>>
  mef.output %t : !mgp.list<!mgp.tensor<3x3xsi32>>
}

// -----

// CHECK-LABEL: @list_create_mismatch_rank
func.func @list_create_mismatch_rank(
  %t0 : !mgp.tensor<3x3xf32>
) {
  // expected-error@+1 {{input type must be a subtype of list element type: '!mgp.tensor<3x3xf32>' v.s. '!mgp.tensor<1x3x3xf32>'}}
  %t = mgp.list.create %t0 : (!mgp.tensor<3x3xf32>) -> !mgp.list<!mgp.tensor<1x3x3xf32>>
  mef.output %t : !mgp.list<!mgp.tensor<1x3x3xf32>>
}

// -----

// CHECK-LABEL: @list_create_mismatch_dim
func.func @list_create_mismatch_dim(
  %t0 : !mgp.tensor<3x?xf32>
) {
  // expected-error@+1 {{input type must be a subtype of list element type: '!mgp.tensor<3x?xf32>' v.s. '!mgp.tensor<3x3xf32>'}}
  %t = mgp.list.create %t0 : (!mgp.tensor<3x?xf32>) -> !mgp.list<!mgp.tensor<3x3xf32>>
  mef.output %t : !mgp.list<!mgp.tensor<3x3xf32>>
}

// -----

//
// mgp.list.get
//

// CHECK-LABEL: @list_get_invalid_index_dtype
func.func @list_get_invalid_index_dtype(
  %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
  %idx : !mgp.tensor<si16>
) {
  // expected-error@+1 {{'mgp.list.get' op expect index tensor to have dtype si32 or si64, but got: 'si16'}}
  %t = mgp.list.get %lst, %idx : !mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<si16>
  mef.output %t : !mgp.tensor<3x3xf32>
}

// -----

// CHECK-LABEL: @list_get_non_scalar_index
func.func @list_get_non_scalar_index(
  %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
  %idx : !mgp.tensor<2xsi32>
) {
  // expected-error@+1 {{'mgp.list.get' op expect index tensor to have 1 element, but got: '!mgp.tensor<2xsi32>'}}
  %t = mgp.list.get %lst, %idx : !mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<2xsi32>
  mef.output %t : !mgp.tensor<3x3xf32>
}

// -----

//
// mgp.list.insert
//

// CHECK-LABEL: @list_insert_invalid_index_dtype
func.func @list_insert_invalid_index_dtype(
  %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
  %tsr : !mgp.tensor<3x3xf32>,
  %idx : !mgp.tensor<si16>
) {
  // expected-error@+1 {{'mgp.list.insert' op expect index tensor to have dtype si32 or si64, but got: 'si16'}}
  %t = mgp.list.insert %lst, %tsr, %idx : (!mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<3x3xf32>, !mgp.tensor<si16>) -> !mgp.list<!mgp.tensor<3x3xf32>>
  mef.output %t : !mgp.list<!mgp.tensor<3x3xf32>>
}

// -----

// CHECK-LABEL: @list_insert_non_scalar_index
func.func @list_insert_non_scalar_index(
  %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
  %tsr : !mgp.tensor<3x3xf32>,
  %idx : !mgp.tensor<2xsi32>
) {
  // expected-error@+1 {{'mgp.list.insert' op expect index tensor to have 1 element, but got: '!mgp.tensor<2xsi32>'}}
  %t = mgp.list.insert %lst, %tsr, %idx : (!mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<3x3xf32>, !mgp.tensor<2xsi32>) -> !mgp.list<!mgp.tensor<3x3xf32>>
  mef.output %t : !mgp.list<!mgp.tensor<3x3xf32>>
}

// -----

// CHECK-LABEL: @list_insert_mismatch_element_type
func.func @list_insert_mismatch_element_type(
  %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
  %tsr : !mgp.tensor<3x3xsi32>,
  %idx : !mgp.tensor<si32>
) {
  // expected-error@+1 {{new element type must be a subtype of output element type: '!mgp.tensor<3x3xsi32>' v.s. '!mgp.tensor<3x3xf32>'}}
  %t = mgp.list.insert %lst, %tsr, %idx : (!mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<3x3xsi32>, !mgp.tensor<si32>) -> !mgp.list<!mgp.tensor<3x3xf32>>
  mef.output %t : !mgp.list<!mgp.tensor<3x3xf32>>
}

// -----

// CHECK-LABEL: @list_insert_mismatch_rank
func.func @list_insert_mismatch_rank(
  %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
  %tsr : !mgp.tensor<1x3x3xf32>,
  %idx : !mgp.tensor<si32>
) {
  // expected-error@+1 {{new element type must be a subtype of output element type: '!mgp.tensor<1x3x3xf32>' v.s. '!mgp.tensor<3x3xf32>'}}
  %t = mgp.list.insert %lst, %tsr, %idx : (!mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<1x3x3xf32>, !mgp.tensor<si32>) -> !mgp.list<!mgp.tensor<3x3xf32>>
  mef.output %t : !mgp.list<!mgp.tensor<3x3xf32>>
}

// -----

// CHECK-LABEL: @list_insert_mismatch_dim
func.func @list_insert_mismatch_dim(
  %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
  %tsr : !mgp.tensor<?x3xf32>,
  %idx : !mgp.tensor<si32>
) {
  // expected-error@+1 {{new element type must be a subtype of output element type: '!mgp.tensor<?x3xf32>' v.s. '!mgp.tensor<3x3xf32>'}}
  %t = mgp.list.insert %lst, %tsr, %idx : (!mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<?x3xf32>, !mgp.tensor<si32>) -> !mgp.list<!mgp.tensor<3x3xf32>>
  mef.output %t : !mgp.list<!mgp.tensor<3x3xf32>>
}

// -----

//
// mgp.list.append
//

// CHECK-LABEL: @list_append_mismatch_element_type
func.func @list_append_mismatch_element_type(
  %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
  %tsr : !mgp.tensor<3x3xsi32>
) {
  // expected-error@+1 {{new element type must be a subtype of output element type: '!mgp.tensor<3x3xsi32>' v.s. '!mgp.tensor<3x3xf32>'}}
  %t = mgp.list.append %lst, %tsr : (!mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<3x3xsi32>) -> !mgp.list<!mgp.tensor<3x3xf32>>
  mef.output %t : !mgp.list<!mgp.tensor<3x3xf32>>
}

// -----

// CHECK-LABEL: @list_append_mismatch_rank
func.func @list_append_mismatch_rank(
  %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
  %tsr : !mgp.tensor<1x3x3xf32>
) {
  // expected-error@+1 {{new element type must be a subtype of output element type: '!mgp.tensor<1x3x3xf32>' v.s. '!mgp.tensor<3x3xf32>'}}
  %t = mgp.list.append %lst, %tsr : (!mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<1x3x3xf32>) -> !mgp.list<!mgp.tensor<3x3xf32>>
  mef.output %t : !mgp.list<!mgp.tensor<3x3xf32>>
}

// -----

// CHECK-LABEL: @list_append_mismatch_dim
func.func @list_append_mismatch_dim(
  %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
  %tsr : !mgp.tensor<?x3xf32>
) {
  // expected-error@+1 {{new element type must be a subtype of output element type: '!mgp.tensor<?x3xf32>' v.s. '!mgp.tensor<3x3xf32>'}}
  %t = mgp.list.append %lst, %tsr : (!mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<?x3xf32>) -> !mgp.list<!mgp.tensor<3x3xf32>>
  mef.output %t : !mgp.list<!mgp.tensor<3x3xf32>>
}
