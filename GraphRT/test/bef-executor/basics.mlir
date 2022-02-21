// RUN: bef-executor %s | FileCheck %s

// CHECK: Initializing leak check allocator
// CHECK: hello bef-executor 1
// CHECK: bef barf: 42 x

