// RUN: not max-engine benchmark %S/../Inputs/relu100x100x3 2>&1 | FileCheck %s
// Until we have batching support, this won't work out of the box.  But we get
// really far!  Check that we get at least that far.
// CHECK: Estimating latency for query generation... Error
// CHECK-NEXT: Error estimating latency: Unable to generate sample query: Random input requires fully defined input shape
