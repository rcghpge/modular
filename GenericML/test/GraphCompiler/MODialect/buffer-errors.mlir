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

mo.graph @mutable_store(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %tensor: !mo.tensor<[10, 20], f32>) {
  // expected-error@+1 {{The output chain from mo.mutable.store can be used at most once found: 2 uses}}
  %outChain = mo.mutable.store(%ch, %buf, %tensor) :
    (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> (!mo.chain)

  %ch0 = mo.chain.create(%outChain, %outChain) : !mo.chain, !mo.chain

  mo.output
}

// -----

mo.graph @mutable_store(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %tensor: !mo.tensor<[10, 20], f32>) {
  %ch0 = mo.chain.create(%ch, %ch) : !mo.chain, !mo.chain

  // expected-error@+1 {{mo.mutable.store must have exclusive access to its input chain found: 2 other uses.}}
  %outChain = mo.mutable.store(%ch, %buf, %tensor) :
    (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> (!mo.chain)

  mo.output
}

// -----

mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>) {
  // expected-error@+1 {{The output chain from mo.mutable.load can be used at most once found: 2 uses}}
  %outChain, %tensor = mo.mutable.load(%ch, %buf) :
    (!mo.chain, !mo.buffer<[10, 20], f32>) ->
    (!mo.chain, !mo.tensor<[10, 20], f32>)

  %ch0 = mo.chain.create(%outChain, %outChain) : !mo.chain, !mo.chain

  mo.output
}

// -----

mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>) {
  %ch0 = mo.chain.create(%ch, %ch) : !mo.chain, !mo.chain

  // expected-error@+1 {{mo.mutable.load must have exclusive access to its input chain found: 2 other uses.}}
  %outChain, %tensor = mo.mutable.load(%ch, %buf) :
    (!mo.chain, !mo.buffer<[10, 20], f32>) ->
    (!mo.chain, !mo.tensor<[10, 20], f32>)


  mo.output
}

// // -----
mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %slice: !mo.tensor<[3, 3], f32>, %sliceOther: !mo.tensor<[2], si64>) {
  // expected-error@+1 {{The output chain from mo.mutable.store.slice can be used at most once found: 2 uses}}
  %outChain = mo.mutable.store.slice(%ch0, %buf, %slice, %sliceOther, %sliceOther, %sliceOther) :
    (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[3, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) ->
    (!mo.chain)

  %ch0 = mo.chain.create(%outChain, %outChain) : !mo.chain, !mo.chain

  mo.output
}

// -----

mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %slice: !mo.tensor<[3, 3], f32>, %sliceOther: !mo.tensor<[2], si64>) {
  %ch0 = mo.chain.create(%ch, %ch) : !mo.chain, !mo.chain

  // expected-error@+1 {{mo.mutable.store.slice must have exclusive access to its input chain found: 2 other uses.}}
  %outChain = mo.mutable.store.slice(%ch, %buf, %slice, %sliceOther, %sliceOther, %sliceOther) :
    (!mo.chain, !mo.buffer<[10, 20], f32>, !mo.tensor<[3, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) ->
    (!mo.chain)


  mo.output
}
