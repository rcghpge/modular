# Validating NIXL after an upstream sync

After backporting from upstream NIXL (`ai-dynamo/nixl`) into
`MLRT/Driver/NIXL`, run the checks below before landing the sync. The
goal is to catch regressions in three layers: the in-tree libraries
(plugin shared objects, agent, transfer paths), the disaggregated
inference (DI) stack that depends on them, and the EFA / cross-node
behavior that only shows up under real network load.

The tests are listed cheapest first. Run everything in
[In-tree unit tests](#in-tree-unit-tests) on every sync; escalate to
the workflow runs when the diff touches a transfer path, plugin
loading, or telemetry.

## In-tree unit tests

These run via `bazel test` and cover the agent core, UCX plugin, and
libfabric plugin. Run on a host with a CUDA GPU (`bt-h100` or
`bt-b200` if you're on a CPU box):

```bash
bt //MLRT/unittests/Driver/NIXL/Core/...
bt //MLRT/unittests/Driver/NIXL/UCX/...
bt //MLRT/unittests/Driver/NIXL/Libfabric/...
```

What each subtree covers:

- **Core** — agent lifecycle, metadata streaming, logging, telemetry
  (event queue, exporter, plugin loader).
- **UCX** — connection/concurrency, error-handling paths,
  metadata exchange, `queryMem`, and the full transfer matrix
  (`TransferTests`). This is the most exercise of UCX-backed
  read/write/notif.
- **Libfabric** — plugin loading, topology discovery, binary
  notifications.

### LibfabricVramSmoke (EFA-aware, separate target)

`//MLRT/unittests/Driver/NIXL/Libfabric:LibfabricVramSmoke` is a
standalone binary (not a `bazel test` target) that exercises DRAM→VRAM
reads through the libfabric plugin. It requires:

- `/dev/infiniband` exposed to the host (EFA NIC visible).
- An EFA-capable peer or loopback path. In practice, run it on an
  AWS B200 Coder workspace.

```bash
./bazelw run //MLRT/unittests/Driver/NIXL/Libfabric:LibfabricVramSmoke
```

## Disaggregated inference quick-tests

Trigger the
[`benchmarkMammoth`](../../../../.github/workflows/benchmarkMammoth.yaml)
workflow with the three `quick-test-*` combos below. Each runs at `mc=1,2` and
takes about 30 minutes, so a full sweep is ~90 minutes of wall-clock if you fan
them out in parallel. These exercise the full MAX Serve stack end-to-end and are
the most reliable signal that an upstream NIXL bump didn't break disaggregated
inference.

| Cluster config                                         | Benchmark config                               | Runtime |
|--------------------------------------------------------|------------------------------------------------|---------|
| `1p1d_tp_prefill_dp_decode_kimi_k25_nvfp4_b200_aws`    | `quick-test-synthetic75k300-kimi-k25-nvfp4`    | ~30 min |
| `1p1d_tp_prefill_dp_decode_kimi_k25_nvfp4_b200_aws`    | `quick-test-multiturn-kimi-k25-nvfp4`          | ~30 min |
| `1p1d_dp_prefill_dp_decode_deepseekv31_nvfp4_b200_aws` | `quick-test-synthetic75k300-deepseekv31-nvfp4` | ~30 min |

Trigger from the GitHub UI ("Run workflow" on `benchmarkMammoth`), or
via `gh`:

```bash
gh workflow run benchmarkMammoth.yaml \
  -f cluster_config=1p1d_tp_prefill_dp_decode_kimi_k25_nvfp4_b200_aws \
  -f benchmark_config=quick-test-synthetic75k300-kimi-k25-nvfp4
```

Nightly CI runs the corresponding `synthetic-75k300-*` and
`multiturn-*` variants (without the `quick-test-` prefix) for deeper
coverage. If a `quick-test-*` is green but the diff is risky, also
trigger the matching full-size variant.

## Local manual DI smoke

For changes that touch cross-node KV transfer specifically, run
through the manual two-workspace DI walkthrough:

- [`run_efa_di_benchmark.md`](./run_efa_di_benchmark.md) — pairs two
  AWS B200 Coder workspaces over EFA and runs a DeepSeek V3.1 NVFP4
  1P1D benchmark. Use this when CI is unavailable or you need to
  attach a debugger / profiler to a real EFA transfer.

## Nightly monitoring

Subscribe to
[`#mammoth-benchmark-notifications`](https://modular-ai.slack.com/archives/C0B2GH9BL01)
to get the daily nightly CI feed (aggregated + disaggregated, with diagnosis
results). After an upstream sync lands, watch this channel for a few days —
regressions that the `quick-test-*` combos miss often show up in the larger
nightly variants or as accumulated flakiness.

## Coverage gaps to track

The current validation surface is mostly "big smoke tests" — there
are open items to add targeted coverage:

- [SERVOPT-1405](https://linear.app/modularml/issue/SERVOPT-1405) —
  NIXL transfer smoke test for DI block count boundary conditions.

When either of these lands, add it to the
[In-tree unit tests](#in-tree-unit-tests) section above and drop the
corresponding manual step.

## Suggested checklist for a sync PR

Paste this into the PR description:

```markdown
- [ ] `bt //MLRT/unittests/Driver/NIXL/Core/...` green
- [ ] `bt //MLRT/unittests/Driver/NIXL/UCX/...` green
- [ ] `bt //MLRT/unittests/Driver/NIXL/Libfabric/...` green
- [ ] `LibfabricVramSmoke` run on a B200 workspace (if libfabric or
      VRAM paths changed)
- [ ] `benchmarkMammoth` triggered for all three quick-test combos,
      all green
- [ ] Watched `#mammoth-benchmark-notifications` for one nightly
      cycle after merge (if not blocking the PR)
```
