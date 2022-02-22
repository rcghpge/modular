// RUN: bef-executor %s | FileCheck %s

// CHECK: Initializing leak check allocator

// CHECK: test1: raw
// CHECK: hello bef-executor 1
// CHECK: bef barf: 42 x

// CHECK: test2: RCRef
// CHECK: hello bef-executor 1
// CHECK: bef barf: 42 x

// CHECK: test3: AsyncValueRef
// CHECK: hello bef-executor 1
// CHECK: bef barf: 42 x
