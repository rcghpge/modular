// RUN: mo-opt %s -split-input-file -verify-diagnostics

// One case per rule of the tile container verifiers. The bodies are the
// interim scalar shape (`tensor.load` / `tensor.store`).

func.func @no_captures(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  // expected-error @+1 {{requires at least one (store-target) capture}}
  %epi = mogg.tile.create.epilogue %s (%val : f32, %idx : !map.index<2>) ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.yield %val : f32
  }
  return
}

// -----

func.func @arg_count(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{body takes a value, an index, and one arg per capture}}
  %epi = mogg.tile.create.epilogue %s (%val : f32) capturing {%cout = %out : !mogg.tensor} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.yield %val : f32
  }
  return
}

// -----

func.func @compute_lane_without_yield(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{a compute lane ends in a yield of one value}}
  %epi = mogg.tile.create.epilogue %s (%val : f32, %idx : !map.index<2>) capturing {%cout = %out : !mogg.tensor} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.tensor.store %cout[%idx] = %val : !mogg.tensor, !map.index<2>, f32
  }
  return
}

// -----

func.func @store_lane_with_yield(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{a store lane does not yield}}
  %epi = mogg.tile.create.epilogue %s (%val : f32, %idx : !map.index<2>) capturing {%cout = %out : !mogg.tensor} ptr %ptr lane = store, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.yield %val : f32
  }
  return
}

// -----

func.func @store_lane_wrong_target(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %z = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Input, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{a store lane ends in a store to the store-target capture at the loop index}}
  %epi = mogg.tile.create.epilogue %s (%val : f32, %idx : !map.index<2>) capturing {%cout = %out : !mogg.tensor, %cz = %z : !mogg.tensor} ptr %ptr lane = store, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.tensor.store %cz[%idx] = %val : !mogg.tensor, !map.index<2>, f32
  }
  return
}

// -----

func.func @load_off_index(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %z = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // The body is not isolated from above, so a tensor from the enclosing
  // scope can be named; it is not a capture.
  %epi = mogg.tile.create.epilogue %s (%val : f32, %idx : !map.index<2>) capturing {%cout = %out : !mogg.tensor, %cz = %z : !mogg.tensor} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    // expected-error @+1 {{in a tile epilogue must read a capture at the loop index}}
    %zv = mogg.tensor.load %z[%idx] : !mogg.tensor, !map.index<2> -> f32
    mogg.yield %zv : f32
  }
  return
}

// -----

func.func @index_transform(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %epi = mogg.tile.create.epilogue %s (%val : f32, %idx : !map.index<2>) capturing {%cout = %out : !mogg.tensor} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    // expected-error @+1 {{cannot transform the index of a tile epilogue}}
    %other = mogg.index.reshape(%idx) <fromShape = #mosh<ape[16, 8]> : !mosh.ape, toShape = #mosh<ape[128]> : !mosh.ape> : !map.index<2> -> !map.index<1>
    mogg.yield %val : f32
  }
  return
}

// -----

func.func @shape_captures(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{a tile epilogue takes no shape captures}}
  %epi = mogg.tile.create.epilogue %s (%val : f32, %idx : !map.index<2>) capturing {%cout = %out : !mogg.tensor} shapes {%sh = %s} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.yield %val : f32
  }
  return
}
