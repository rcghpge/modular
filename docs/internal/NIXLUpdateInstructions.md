# NIXL Update Instructions

We vendor [upstream NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA's
Inference Xfer Library) as an unmodified Bazel `http_archive`
(`@nixl_upstream`) pinned to a released tag, the same way we vendor LLVM. This
document covers how to bump the pinned version and how to carry a local patch
if one is ever needed.

## How to bump the upstream NIXL version

The version is pinned in a `BEGIN_NIXL_GENERATED` / `END_NIXL_GENERATED`
block inside `MODULE.bazel` (the `nixl_upstream` `http_archive`: its tag,
URL, `strip_prefix`, and `sha256`). Use the `update-nixl` helper to update it:

```bash
git checkout -b nixl-bump main
git pull --rebase origin main
update-nixl 1.2.0           # rewrite the pin and print the new sha256
update-nixl 1.2.0 --commit  # same, plus create a git commit
```

`update-nixl` downloads the release tarball for the given tag from
`github.com/ai-dynamo/nixl`, computes its `sha256`, and rewrites the generated
block in `MODULE.bazel`. The script lives at `utils/update_nixl.py` and mirrors
`utils/update_llvm.py`.

If upstream's source layout or build inputs changed, you may also need to
update the Bazel overlay at `bazel/third-party/nixl_upstream.BUILD`. After
bumping, rebuild and re-test (see [Testing](#testing-after-an-update)).

## Dealing with NIXL patches

We depend on **unmodified** upstream NIXL. If you must carry a local change
(for example, a fix that has not landed in a tagged release yet), apply it as a
patch rather than forking:

1. Drop the patch file at `bazel/third-party/nixl-<name>.patch` (the `nixl-`
   prefix matches the LLVM convention).
2. Add it to the `patches = [...]` list of the `nixl_upstream` `http_archive` in
   `MODULE.bazel`. The archive already sets `patch_args = ["-p1"]`.

> [!NOTE]
> Every patch should include a URL to the upstream tracking issue or PR that,
> once merged, lets us delete the patch. The point of patching is to unblock
> quickly, not to maintain a fork. Remove the patch entry (and the file) when
> the fix lands upstream and the pin is bumped to a version that includes it.

## Testing after an update

A NIXL change must be validated against the in-tree libraries, the
disaggregated-inference stack, and EFA. See the
[validation doc](../../MLRT/docs/Driver/NIXL/validate_upstream_sync.md)
for the full procedure and test list.

The quickest path is the one-click GitHub Actions workflow, which runs the
NIXL unit tests (Core/UCX/Libfabric on a B200), the EFA `LibfabricVramSmoke`,
and the disaggregated `benchmarkMammoth` quick-tests:

```bash
gh workflow run validateNIXLSync.yaml \
  --repo modularml/modular \
  --ref <branch>
```

Running from a feature branch auto-builds the MAX Serve image from that
commit and passes it to all three Mammoth quick-tests — no manual image
build step needed.

As a final sanity check, also run `br //:install`.
