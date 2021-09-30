---
has_children: true
has_toc: true
nav_order: 4
---

# Contributing

CUB uses Github to manage all open-source development, including bug tracking,
pull requests, and design discussions. CUB is tightly coupled to the Thrust
project, and a compatible version of Thrust is required when working on the
development version of CUB.

To setup a CUB development branch, it is recommended to recursively clone the
Thrust repository and use the CUB submodule at `dependencies/cub` to stage
changes. CUB's tests and examples can be built by configuring Thrust with the
CMake option `THRUST_INCLUDE_CUB_CMAKE=ON`.

This process is described in more detail in
[Thrust's documentation](https://nvidia.github.io/thrust/contributing).

