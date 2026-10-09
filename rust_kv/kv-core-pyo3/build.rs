//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
fn main() {
    // Python C-API symbols are resolved at import time by the embedding
    // interpreter. No-op on Linux; `-undefined dynamic_lookup` on macOS.
    pyo3_build_config::add_extension_module_link_args();
}
