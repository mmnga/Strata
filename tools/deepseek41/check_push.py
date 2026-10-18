#!/usr/bin/env python3
"""Check a DeepSeek publication without uploading local experiment artifacts."""
import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ZERO = "0" * 40
DESTINATIONS = {
    "https://github.com/mmnga/Strata",
    "https://github.com/mmnga/Strata.git",
    "git@github.com:mmnga/Strata.git",
    "ssh://git@github.com/mmnga/Strata.git",
}
FILES = {
    ".gitignore", ".githooks/pre-push", "AGENTS.md", "CMakeLists.txt",
    "docs/DEEPSEEK41_NATIVE.md", "examples/deepseek41-server.json",
    "include/strata/artifact/gguf_reader.hpp",
    "include/strata/artifact/model_family.hpp",
    "include/strata/kernels/iq_kernels.hpp",
    "include/strata/kernels/native_mmvq.hpp",
    "serve/deepseek41.py", "serve/frontend.py", "serve/server.py",
    "serve/test_deepseek41.py", "serve/test_family_name.py",
    "src/kernels/cuda/iq_kernels.cu", "src/kernels/cuda/native_mmvq.cu",
    "src/program/generate.cpp", "tests/core/deepseek41_artifact_test.cpp",
    "tests/core/gguf_fixture.hpp", "tools/strata_tokenizer.py",
}
SOURCE_DIRS = {
    "include/strata/deepseek41/": {".hpp"},
    "src/deepseek41/": {".cpp", ".cu"},
    "tests/deepseek41/": {".cpp", ".cu", ".hpp"},
    "tools/deepseek41/": {".py", ".cpp"},
}
SECRETS = {
    "private key": rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----",
    "GitHub token": rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})",
    "API key literal": rb"(?i)(?:api[_-]?key|authorization)[\"']?\s*[:=]\s*[\"'](?:Bearer\s+)?[A-Za-z0-9_-]{24,}",
    "private home path": rb"/(?:home|Users)/[^\s'\"<>]+",
    "private LAN address": rb"192\.168\.[0-9]+\.[0-9]+",
}


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError("git check failed; fetch origin and inspect the local refs")
    return result.stdout


def allowed(path):
    if path in FILES:
        return True
    if any(part.startswith(".") or part == "__pycache__" for part in Path(path).parts):
        return False
    return any(path.startswith(prefix) and Path(path).suffix in suffixes
               for prefix, suffixes in SOURCE_DIRS.items())


def check_range(base, head):
    git("merge-base", "--is-ancestor", base, head)
    commits = git("rev-list", "--reverse", head, "--not", base, "origin/main").decode().splitlines()
    checked, paths = set(), set()
    for commit in commits:
        names = git("diff-tree", "--root", "-m", "--no-commit-id", "--no-renames",
                    "--name-only", "-r", "-z", commit).decode().split("\0")
        for path in set(names) - {""}:
            if not allowed(path):
                raise ValueError(f"local-only or unreviewed path: {path}")
            paths.add(path)
            entry = git("ls-tree", "-z", commit, "--", path)
            if not entry:
                continue  # A deletion; earlier commits are still checked.
            mode, kind, oid = entry.split(b"\t", 1)[0].decode().split()
            if mode not in {"100644", "100755"} or kind != "blob":
                raise ValueError(f"unexpected file type: {path}")
            if oid in checked:
                continue
            if int(git("cat-file", "-s", oid)) > 2 * 1024 * 1024:
                raise ValueError(f"file exceeds the source-only size limit: {path}")
            data = git("cat-file", "blob", oid)
            if b"\0" in data:
                raise ValueError(f"binary artifact: {path}")
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"non-text artifact: {path}") from None
            for label, pattern in SECRETS.items():
                if re.search(pattern, data):
                    raise ValueError(f"{label} detected in {path}; inspect locally")
            checked.add(oid)
    print(f"DeepSeek push check: PASS ({len(commits)} commits, {len(paths)} paths, {len(checked)} text blobs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--pre-push", action="store_true")
    parser.add_argument("remote", nargs="?")
    parser.add_argument("url", nargs="?")
    args = parser.parse_args()
    if args.pre_push:
        if args.remote != "origin" or args.url not in DESTINATIONS:
            raise ValueError("only origin at mmnga/Strata is authorized")
        for line in sys.stdin:
            local_ref, local_oid, remote_ref, remote_oid = line.split()
            if remote_ref != "refs/heads/deepseek" or local_oid == ZERO:
                raise ValueError("push only the adopted deepseek branch; no deletions or other refs")
            check_range("origin/main" if remote_oid == ZERO else remote_oid, local_oid)
    else:
        if git("branch", "--show-current").strip() != b"deepseek":
            raise ValueError("run this check from the deepseek branch")
        urls = git("remote", "get-url", "--push", "--all", "origin").decode().splitlines()
        if len(urls) != 1 or urls[0] not in DESTINATIONS:
            raise ValueError("origin must push only to mmnga/Strata")
        base = args.base
        if base is None:
            found = subprocess.run(["git", "show-ref", "--verify", "--quiet", "refs/remotes/origin/deepseek"], cwd=ROOT)
            base = "origin/main" if found.returncode else "origin/deepseek"
        check_range(base, args.head)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(f"DeepSeek push check: REFUSED: {error}", file=sys.stderr)
        sys.exit(1)
