# Hardware-independent Driver

## Overview

This sub-package is a new implementation of the Modular Driver and is currently in
prototype stage. It supports the definition of pan-HW concepts like `Device`,
`Context`, `Buffer`, `Queue` etc. on top of the platform-agnostic API specified
in the Hardware Abstraction Layer (HAL): `//SDK/.../HAL`.

It also allows for the implementation of advanced features, like
reference-counting and caching in pan-HW manner.

The library life-cycle begins with a `Driver` object: the full path to the plugin
for each kind of HW (CUDA, HIP etc.) must be provided to its CTOR. Thereafter,
we may retrieve `Device` objects, create `Context` and `Buffer` objects etc.

Officially-supported HW plugins live inside `//Target/...`. But these are not
directly consumed here. Applications or uber libraries that combine both bear
the responsibility of fetching desired plugins and making their location known
to the `Driver` classes/objects defined here.
