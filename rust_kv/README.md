# rust_kv

Rust KV cache components for MAX Serve.

Today this workspace holds the Rust port of MAX's Python KV cache connectors
(`LocalConnector` + `TieredConnector`), plus the start of a KV cache manager
shared by MAX Serve and Mach:

- `kv-core`: host-side KV cache bookkeeping with no GPU, async runtime, or
  Python dependency. Holds the prefix-hit rules, the Jenga geometry planner,
  and the Jenga block pool for now. Mach's `engine-kv` depends on it by path.
- `kv-core-pyo3`: the pyo3 bindings of kv-core, which ship in the `max` wheel
  as the compiled module `max._kv_core` (built by
  `//max/python/max:_kv_core.bindings`, the way `max._core` is). kv-core stays
  closed source: `rust_kv/` is not in the OSS export, whose build takes the
  module from the prebuilt `max` wheel, and its public surface is the stub
  `max/python/max/_kv_core.pyi`, which must change with the bindings.
- `kv-tier-connector`: the pure logic — disk tier, copy engine, lanes,
  connectors — on top of `kv-core`'s block pool.
- `kv-tier-connector-pyo3`: the pyo3 bindings that expose it to Python as the
  `kv_tier_connector` module (bundled into the private `kv-tier-connector`
  wheel).

The directory name is deliberately generic: more of the KV cache manager may
move from Python to Rust here over time, not just the connectors.

These crates used to live under `dkv/`, but they are not part of the dKV
service — merges here were triggering dKV image builds and staging releases,
so the workspace was split out with its own CI
(`.github/workflows/rustKv.yaml`) that builds, lints, and tests without any
deploy step.

## Building

```bash
# Cargo (fast local iteration)
cd rust_kv
cargo build --workspace
cargo test --workspace
cargo clippy --workspace

# Bazel (what the monorepo presubmit runs), from the repository root
cd ..
./bazelw build //rust_kv/...
./bazelw test //rust_kv/kv-core:kv-core-test
./bazelw test //max/tests/integration/kv_cache/internal/kv_core:test_kv_core_binding
./bazelw test //rust_kv/kv-tier-connector:kv-tier-connector-test
```

Third-party crates for the Bazel build come from the `@rust_kv_crates`
repository, generated from this workspace's `Cargo.toml`/`Cargo.lock` (see
`MODULE.bazel`). The pyo3 crates are the one exception: `pyo3_extension`
injects pyo3 via the monorepo-wide `rust_pyo3_toolchain`, which is bound to
`@crates//:pyo3` (the dkv-fed universe), so the Bazel targets must not also
depend on `@rust_kv_crates`' pyo3. Keep this workspace's pyo3 pin on the same
minor version as `dkv/`'s.
