# Buffer Cache Overview

The buffer cache is a memory manager for host or device allocated memory. It
grabs chunks of memory from an underlying allocated (e.g. the CUDA driver) and
hands them out to clients as necessary.

![BufferCacheOverview.png](BufferCacheOverview.png)

The source of the above image is available
[here](https://excalidraw.com/#json=fvkxYZNkbOnKZGoYDFvvs,7FnGbaodiDehk0RVWHDlRQ).

## Setup

We have allocated two chunks of memory with a set of memory blocks each. A chunk
starts with a single large memory block which is subdivided into multiple blocks
as needed to satisfy allocation requests.

There are 5 blocks on the shared BufferCache free list (aka parent) and 1 block
on the BufferCache free list associated with the stream in the single (in this
example!) DeviceContext.

## Usage scenarios

### Scenario 1

The DeviceContext releases the memory block `K`. Since all allocations and frees
are streaming we cannot put this block into the shared/parent free list, but
instead it has to be sequestered with blocks used by the DeviceContext’s stream:

- Check whether the next and previous blocks in the chunk (`K→next` and
  `K→prev`) are free and if they have the same owner.
- Block `J` satisfies this requirement, so we merge blocks `J` and `K` and add
  the resulting block to the free list associated with the DeviceContext.

### Scenario 2

We want to allocate a new 3MB block on the DeviceContext’s stream:

- Check if the free list associated with the DeviceContext contains a block
  greater or equal to 3MB.
- Since we cannot satisfy this, we check whether the parent/shared free list
  satisfies this requirement.
- None of the existing free lists can satisfy the request to allocate 3MB, so we
  end up allocating a new chunk and split the first 3MB off into its own block,
  adding the remainder to the shared free list.

Note: We can only allocate a new chunk if we have not exhausted the maximum
configured size of the buffer cache.

### Scenario 3

The block E is freed on a different DeviceContext (not pictured) and added to
the associated free list. At some point in time later, this DeviceContext is
released:

- All free blocks on this DeviceContext’s free list are flushed to the parent
  free list, which will merge blocks D, E and F into a single 4.5MB block.

### Scenario 4

We want to allocate a 1.5MB block on the pictured DeviceContext:

- Block J is removed from the DeviceContext’s free list.
- The DeviceContext’s free list is now empty.

### Scenario 5

We want to allocate a 1.8MB block on the pictured DeviceContext:

- The DeviceContext’s free list cannot satisfy this request.
- Find a large enough block on the parent free list identifies block F as
  sufficiently large.
- Remove F from the free list.
- Split the block F into a block F1 (1.8MB) and F2 (0.2MB).
- Add F2 to the shared free list.
