// RUN: mo-opt %s --canonicalize --split-input-file --verify-diagnostics 

mo.graph @ranked_unk(
  // expected-error@+1 {{mo.buffer is not allowed to have shapes with '?'}}
  %b0: !mo.buffer<[?, ?], f32>,
) {
  mo.output
}

// -----

mo.graph @unranked_unk(
  // expected-error@+1 {{mo.buffer is not allowed to have unknown shape}}
  %b0: !mo.buffer<?, f32>
) {
  mo.output
}

// -----

mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f64>) {
  // expected-error@+1 {{'mo.mutable.load' op failed to verify that all of {inBuffer, outTensor} have same dtype}}
  %outChain, %tensor = mo.mutable.load(%ch, %buf) :
    (!mo.chain, !mo.buffer<[10, 20], f64>) ->
    (!mo.chain, !mo.tensor<[10, 20], f32>)
  mo.output
}

// -----

mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>) {
  // expected-error@+1 {{'mo.mutable.load' op failed to verify that all of {inBuffer, outTensor} have same shapeAttr}}
  %outChain, %tensor = mo.mutable.load(%ch, %buf) :
    (!mo.chain, !mo.buffer<[10, 20], f32>) ->
    (!mo.chain, !mo.tensor<[30, 20], f32>)
  mo.output
}

// -----

mo.graph @mutable_store(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %tensor: !mo.tensor<[10, 20], f64>) {
  // expected-error@+1 {{'mo.mutable.store' op failed to verify that all of {inBuffer, inTensor} have same dtype}}
  %outChain = mo.mutable.store(%ch, %buf, %tensor) :
    (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f64>) -> (!mo.chain)
  mo.output
}

// -----

mo.graph @mutable_store(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %tensor: !mo.tensor<[10, 30], f32>) {
  // expected-error@+1 {{'mo.mutable.store' op failed to verify that all of {inBuffer, inTensor} have same shapeAttr}}
  %outChain = mo.mutable.store(%ch, %buf, %tensor) :
    (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[10, 30], f32>) -> (!mo.chain)
  mo.output
}

// -----

mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>) {
  // expected-error@+1 {{Results from mo.mutable.load are only allowed to have at most one use -- found 1 for result #1}}
  %outChain, %tensor = mo.mutable.load(%ch, %buf) :
    (!mo.chain, !mo.buffer<[10, 20], f32>) ->
    (!mo.chain, !mo.tensor<[10, 20], f32>)

  mo.add(%tensor, %tensor) : !mo.tensor<[10, 20], f32>

  mo.output
}

// -----

mo.graph @mutable_store(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %tensor: !mo.tensor<[10, 20], f32>) {
  // expected-error@+1 {{The output chain from mo.mutable.store is only allowed to have at most one use -- found: 2}}
  %outChain = mo.mutable.store(%ch, %buf, %tensor) :
    (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> (!mo.chain)

  %ch0 = mo.chain.create(%outChain, %outChain) : !mo.chain, !mo.chain

  mo.output
}
