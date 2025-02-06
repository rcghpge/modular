# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Usage:
# 1. Check out the max repo (main)
# 2. Check out the modular repo at the HEAD of the release branch
# 3. Run `python3 /path/to/modular/SDK/public/update-max-repo.py -p /path/to/max`
# 4. Go to the max repo, run `git add .`, and commit the result.
# 5. Push to a new branch (e.g., max/v24.x.y) and open a PR against main

import argparse
import pathlib
import shutil
import subprocess


def _check_process(result, message):
    try:
        result.check_returncode()
    except subprocess.CalledProcessError:
        print(message)
        return False

    return True


def sanity_check_max_repo(repo_path):
    _invalid_repo_msg_prefix = (
        f"Invalid max repo destination path: '{repo_path}'"
    )

    cmd_check_remote = "git remote -v"
    try:
        result = subprocess.run(
            cmd_check_remote.split(), cwd=repo_path, stdout=subprocess.PIPE
        )
    except FileNotFoundError:
        print(f"{_invalid_repo_msg_prefix}. Directory not found.")
        return False

    if not _check_process(result, f"{_invalid_repo_msg_prefix}"):
        return False

    if "origin" not in result.stdout.decode(
        "utf-8"
    ) or "modular/max" not in result.stdout.decode("utf-8"):
        print(
            f"{_invalid_repo_msg_prefix}. Repo must have upstream named origin"
            " that points to modular/max."
        )
        return False

    return True


def execute_copy(src, dest):
    rsync_cmd = f"rsync -rtvvP {max_repo_src}/ {path}"

    subprocess.run("git fetch origin".split(), cwd=dest)
    subprocess.run("git checkout main".split(), cwd=dest)
    subprocess.run("git reset --hard origin/main".split(), cwd=dest)

    # Clean slate (except .git)
    subprocess.run("git reset --hard origin/main".split(), cwd=dest)

    import glob

    for f in glob.glob(str(dest) + "/*", include_hidden=True):
        if f != str(dest) + "/.git":
            try:
                shutil.rmtree(f)
            except Exception:
                pass

            try:
                pathlib.Path.unlink(f)
            except Exception:
                pass

    result = subprocess.run("ls -la".split(), cwd=dest, stdout=subprocess.PIPE)
    lines = result.stdout.decode("utf-8").split("\n")
    assert len(lines) == 5, f"Check your folder? Got {lines}"

    subprocess.run(rsync_cmd.split())


def sanity_check_src_path(src):
    _invalid_src_msg_prefix = f"Invalid max repo source path: '{src}'"

    cmd = "ls -l"
    result = subprocess.run(cmd.split(), cwd=src, stdout=subprocess.PIPE)

    if "examples" not in result.stdout.decode("utf-8"):
        print(
            f"{_invalid_src_msg_prefix}. Did not find expected subdirectory"
            " 'examples'."
        )
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Local path to the max repo")
    args = parser.parse_args()

    path = pathlib.Path(args.path).resolve()
    if not sanity_check_max_repo(path):
        exit(1)

    max_repo_src = pathlib.Path(__file__).parent.resolve() / "max-repo"
    if not sanity_check_src_path(max_repo_src):
        exit(1)

    execute_copy(max_repo_src, path)
