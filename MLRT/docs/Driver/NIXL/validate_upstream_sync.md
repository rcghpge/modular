# Validating NIXL after bumping the upstream pin

NIXL is vendored as an unmodified http_archive snapshot of
`ai-dynamo/nixl` (see `MODULE.bazel` for the current pin). To upgrade
to a new release, bump the `version` and `sha256` in `MODULE.bazel` and
update the overlay in `bazel/third-party/nixl_upstream.BUILD` if the
upstream source layout changed. The old backport workflow (copying
upstream commits into `MLRT/Driver/NIXL`) is retired; that fork has
been deleted.

Run the checks below before landing a pin bump. The goal is to catch
regressions in three layers: the vendored library (plugin shared
objects, agent, transfer paths), the disaggregated inference (DI) stack
that depends on them, and the EFA / cross-node behavior that only shows
up under real network load.

The tests are listed cheapest first. Run everything in
[In-tree unit tests](#in-tree-unit-tests) on every bump; escalate to
the workflow runs when the diff touches a transfer path, plugin
loading, or telemetry.

The [`validateNIXLSync`](../../../../.github/workflows/validateNIXLSync.yaml)
workflow bundles all of this into one dispatchable run. See
[`testing.md`](testing.md) for the full tier structure — which lanes gate
the run's red/green and which are informational.

## In-tree unit tests

These run via `bazel test` and cover the agent core, UCX plugin, and
libfabric plugin. Run on a host with a CUDA GPU (`bt-b200` if
you're on a CPU box):

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
via `gh`. Use `--ref main` to test against `main`, or `--ref <branch>` for a
feature branch — both auto-build the MAX Serve image from the target ref:

```bash
gh workflow run benchmarkMammoth.yaml \
  --repo modularml/modular \
  --ref <branch-or-main> \
  -f cluster_config=1p1d_tp_prefill_dp_decode_kimi_k25_nvfp4_b200_aws \
  -f benchmark_config=quick-test-synthetic75k300-kimi-k25-nvfp4
```

To run the full validation (correctness gate, which also dispatches the
shared-pool tests — EFA KV smoke + all three quick-test combos — as an
informational companion run):

```bash
gh workflow run validateNIXLSync.yaml \
  --repo modularml/modular \
  --ref <branch-or-main>
```

Pass `-f run_shared_pool_tests=false` to validate correctness only
without consuming the shared B200 pod pool, or dispatch
`nixlSharedPoolTests.yaml` directly (it takes a `docker_image` input)
to run only the pod-pool tests.

Nightly CI runs the corresponding `synthetic-75k300-*` and
`multiturn-*` variants (without the `quick-test-` prefix) for deeper
coverage. If a `quick-test-*` is green but the diff is risky, also
trigger the matching full-size variant.

## Local manual DI smoke

For changes that touch cross-node KV transfer specifically:

- **KV transfer engine smoke test** — two-process test (sender + receiver, one
  per node) that exercises `KVTransferEngine` + NIXL directly over EFA with no
  model or scheduler involved. Runs as a 4-row sequential sweep in CI via
  `nixlKvSmokeTest.yaml` (~75s per row, ~5.5 min total). For manual runs:

  ```bash
  ./bazelw run \
    //max/tests/integration/kv_cache/transfer_engine:run_cross_node -- \
    --sender-host <sender-node-ip> --receiver-host <receiver-node-ip>
  ```

  The authoritative sweep parameters are in
  `.github/workflows/nixlKvSmokeTest.yaml`.
  Each row targets a specific regression boundary:

  | Row | Label                     | Failure signal                        |
  |-----|---------------------------|---------------------------------------|
  | 1   | baseline (mc=1)           | Any error — EFA transfer path broken  |
  | 2   | data pool (mc=128)        | Crash = data pool exhaustion          |
  | 3   | control pool (mc=32, 200b)| Timeout = control pool exhaustion     |
  | 4   | xfer-id boundary          | Warning absent = structural fix in    |

  Known patches that address these boundaries:
  - [#83975](https://github.com/modularml/modular/pull/83975) — TX queue
    exhaustion under concurrent transfers (Row 2)
  - [#86044](https://github.com/modularml/modular/pull/86044) — notification
    drops at mc≥16 (Rows 2-3)
  - [#87402](https://github.com/modularml/modular/pull/87402) — CM thread
    error handling + control request pool CV backpressure (Row 3)
  - [#87503](https://github.com/modularml/modular/pull/87503) — data request
    pool inline CQ drain (Row 2)

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

Remaining items are tracked under the GEX-3974 testing epic (nightly
correctness baseline on main, resurrecting the `manual`-tagged tests,
AMD coverage). The KV transfer smoke and the single-node perf smoke are
now wired into `validateNIXLSync.yaml` and run nightly.

## Suggested checklist for a pin-bump PR

Paste this into the PR description:

```markdown
- [ ] `validateNIXLSync` correctness gate green (unit tests, binding
      tests, transfer engine tests, EFA KV smoke, perf smoke)
- [ ] Shared-pool tier (`nixlSharedPoolTests`) result reviewed — green,
      or red attributed to pod-pool capacity via the diagnostics
      artifact
- [ ] Watched `#mammoth-benchmark-notifications` for one nightly
      cycle after merge (if not blocking the PR)
```
