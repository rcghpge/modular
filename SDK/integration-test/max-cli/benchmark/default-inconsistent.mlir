// (TODO: MSDK-483)
// REQUIRES: disabled
// RUN: not max benchmark %S/Inputs/semibatchedadder 2>&1 | FileCheck %s
// CHECK: Estimating latency for query generation... Error
// CHECK-NEXT: Error estimating latency: Unable to generate sample query: Random input requires fully defined input shape
