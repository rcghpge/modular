# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -debug-level full %s
# REQUIRES: manual

from time import sleep, now
from runtime.asyncrt import TaskGroup
from max.serve.batch import Batcher


fn rand_sleep():
    var ptr = UnsafePointer[Int64].alloc(1)
    random.randint[DType.int64](ptr.address, 1, 0, 6)  # random workload
    var rVal = int(Scalar.load(ptr)[0])
    var l: Float64 = rVal * 0.01
    sleep(l)


fn count_batch(input: List[Int]) capturing -> List[Int]:
    rand_sleep()
    var output = List[Int](capacity=len(input))
    for _ in input:
        output.append(len(input))
    return output


fn test_batch():
    alias BATCH_SIZE = 8
    alias NUM_REQ = BATCH_SIZE * 128 + (BATCH_SIZE - 1)

    var batcher = Batcher[Int, Int, count_batch](max=BATCH_SIZE, timeout=1)

    @parameter
    async fn submit(owned n: Int):
        _ = batcher.submit(n)

    var tg = TaskGroup[__lifetime_of()]()
    for i in range(NUM_REQ):
        tg.create_task(submit(i))
    tg.wait()
    _ = batcher^


fn main():
    test_batch()
