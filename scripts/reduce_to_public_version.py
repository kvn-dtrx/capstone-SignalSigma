#!/usr/bin/env python3

# ---
# description: |
#  This script reduces the git history of a repository "horizontally"
#  and "vertically" to create a public version of the repository.
# ---


from git_filter_repo import FilterRepo
import subprocess
from pathlib import Path
import fnmatch


def get_commit_hash_of_first_commit() -> str:
    result = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_commit_hash_of_last_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_tagged_commit_hashes() -> set[str]:
    result = subprocess.run(
        ["git", "rev-list", "--tags", "--no-walk"],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.strip().splitlines())


def load_path_blacklist(filename: str = "blacklist.txt") -> set[str]:
    print(filename)
    if not Path(filename).exists():
        return []
    with open(filename) as f:
        return [line.strip() for line in f if line.strip()]


def file_matches_blacklist(filename: str, patterns: set[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False


def main() -> None:
    first_commit = get_commit_hash_of_first_commit()
    last_commit = get_commit_hash_of_last_commit()
    tagged_commits = get_tagged_commit_hashes()

    commits_to_keep = {
        first_commit.encode("ascii"),
        last_commit.encode("ascii"),
    } | {c.encode("ascii") for c in tagged_commits}

    def commit_callback(commit) -> None:
        print(type(commit))
        if commit.original_id not in commits_to_keep:
            commit.skip()

    def blob_callback(blob) -> None:
        print(type(blob))
        filename = blob.name.decode()
        if file_matches_blacklist(filename, blacklist_patterns):
            blob.skip()

    repo = FilterRepo()
    # repo.set_commit_callback(commit_callback)
    # repo.set_file_callback(blob_callback)
    # repo.run()


if __name__ == "__main__":
    main()
