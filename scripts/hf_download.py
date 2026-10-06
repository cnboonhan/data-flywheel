# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub[hf_xet]>=0.34", "questionary>=2.0"]
# ///
"""Pick Hugging Face repos from a list, see the download size and time, then download them.

Each entry is "<folder> <type> <repo_id> <include> <local_name>":
  folder      datasets | checkpoints (top-level folder under --root)
  type        model | dataset (the Hugging Face repo type)
  include     comma-separated glob(s) to download, or "*" for the whole repo
  local_name  folder created under <root>/<folder>/

Every entry is checked first (gating/access, size left to download). A checkbox list lets you
pick what to download; the estimated size and time are shown before anything starts.
A finished download writes <local_name>/.download_complete and is not offered again.
An interrupted download resumes without re-fetching files already on disk.
"""

import argparse
import json
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path

import questionary
from huggingface_hub import HfApi, auth_check, get_token, hf_hub_url, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

MARKER = ".download_complete"


@dataclass
class Item:
    repo_type: str
    repo_id: str
    include: list[str]
    target: Path
    status: str = "ready"  # ready | done | blocked
    reason: str = ""
    left: int = 0
    total: int = 0
    biggest: str = ""  # largest matching file, used for the bandwidth probe
    files: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.target.parent.name}/{self.target.name}"


def parse_entry(line: str, root: Path) -> Item:
    parts = line.split()
    assert len(parts) == 5, f"expected '<folder> <type> <repo_id> <include> <local_name>', got: {line!r}"
    folder, repo_type, repo_id, include, local_name = parts
    assert folder in ("datasets", "checkpoints"), f"folder must be datasets or checkpoints: {line!r}"
    assert repo_type in ("model", "dataset"), f"type must be model or dataset: {line!r}"
    return Item(repo_type, repo_id, include.split(","), root / folder / local_name)


def repo_url(item: Item) -> str:
    return f"https://huggingface.co/{'datasets/' if item.repo_type == 'dataset' else ''}{item.repo_id}"


def scan(api: HfApi, item: Item) -> None:
    """Fill in the item's status, access reason and remaining size."""
    if (item.target / MARKER).exists():
        item.status, item.reason = "done", "already downloaded"
        return
    try:
        info = api.repo_info(item.repo_id, repo_type=item.repo_type, files_metadata=True)
    except RepositoryNotFoundError:
        item.status, item.reason = "blocked", "not found (or private without access)"
        return
    if info.gated:
        try:
            auth_check(item.repo_id, repo_type=item.repo_type)
        except GatedRepoError:
            login = "run `hf auth login`, then " if get_token() is None else ""
            item.status, item.reason = "blocked", f"gated: {login}request access at {repo_url(item)}"
            return
    item.files = [s for s in info.siblings if any(fnmatch(s.rfilename, pat) for pat in item.include)]
    if not item.files:
        item.status, item.reason = "blocked", "no files match the include pattern"
        return
    item.total = sum(f.size or 0 for f in item.files)
    item.left = sum(
        f.size or 0
        for f in item.files
        if not ((p := item.target / f.rfilename).is_file() and p.stat().st_size == (f.size or 0))
    )
    item.biggest = max(item.files, key=lambda f: f.size or 0).rfilename
    item.reason = f"gated ({info.gated}), access granted" if info.gated else "public"


def probe_bandwidth(item: Item, seconds: float = 5.0) -> float:
    """Download the start of the item's largest file for a few seconds; return bytes/sec."""
    request = urllib.request.Request(hf_hub_url(item.repo_id, item.biggest, repo_type=item.repo_type))
    if token := get_token():
        request.add_header("Authorization", f"Bearer {token}")
    start, received = time.monotonic(), 0
    with urllib.request.urlopen(request, timeout=30) as response:
        while time.monotonic() - start < seconds:
            if not (chunk := response.read(1 << 20)):
                break
            received += len(chunk)
    return received / max(time.monotonic() - start, 1e-3)


def fmt_bytes(n: float) -> str:
    if n >= 1e8:
        return f"{n / 1e9:.2f} GB"
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.1f} KB"


def fmt_duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{max(minutes, 1)}m"


def estimate(selected: list[Item], bandwidth: float | None) -> None:
    """Print the total size left and the estimated download time for the selection."""
    left = sum(i.left for i in selected)
    print(f"\nTo download: {len(selected)} repo(s), {fmt_bytes(left)}")
    if not left:
        return
    if bandwidth:
        speed, source = bandwidth * 1e6, "given"
    else:
        print("Measuring download speed (5 s)...", flush=True)
        try:
            speed, source = probe_bandwidth(max(selected, key=lambda i: i.left)), "measured"
        except Exception as exc:
            print(f"Estimated time: unknown (speed probe failed: {exc}); pass --bandwidth MB/s")
            return
    print(f"Estimated time: {fmt_duration(left / speed)} at {speed / 1e6:.1f} MB/s ({source}; "
          "real downloads use parallel transfers and are usually faster)")


def download(api: HfApi, item: Item) -> bool:
    print(f"\n==> {item.name}: {item.repo_type}:{item.repo_id} ({fmt_bytes(item.left)}) -> {item.target}")
    try:
        snapshot_download(item.repo_id, repo_type=item.repo_type, allow_patterns=item.include, local_dir=item.target)
    except Exception as exc:  # keep going so one bad repo doesn't block the rest
        print(f"FAILED {item.name}: {exc}")
        return False
    record = {"repo_type": item.repo_type, "repo_id": item.repo_id, "include": item.include,
              "revision": api.repo_info(item.repo_id, repo_type=item.repo_type).sha,
              "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (item.target / MARKER).write_text(json.dumps(record, indent=2) + "\n")
    print(f"DONE   {item.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="repo root containing datasets/ and checkpoints/")
    parser.add_argument("--dry-run", action="store_true", help="show the plan for every entry and exit")
    parser.add_argument("--all", action="store_true", help="select every downloadable entry (no checkbox list)")
    parser.add_argument("--yes", action="store_true", help="start without asking for confirmation")
    parser.add_argument("--bandwidth", type=float, help="assumed download speed in MB/s (skips the speed probe)")
    parser.add_argument("entries", nargs="+", help='"<folder> <type> <repo_id> <include> <local_name>"')
    args = parser.parse_args()
    interactive = sys.stdin.isatty() and sys.stdout.isatty()

    api = HfApi()
    items = [parse_entry(line, args.root) for line in args.entries]
    print(f"Checking {len(items)} repo(s)...", flush=True)
    for item in items:
        scan(api, item)
    width = max(len(i.name) for i in items)
    for item in items:
        size = f"{fmt_bytes(item.left)} left of {fmt_bytes(item.total)}" if item.status == "ready" else ""
        print(f"  {item.status:<7}  {item.name:<{width}}  {size:<26}  {item.reason}")

    ready = [i for i in items if i.status == "ready"]
    if args.dry_run:
        estimate(ready, args.bandwidth)
        return 0
    if not ready:
        print("\nNothing to download.")
        return 0

    if args.all:
        selected = ready
    elif interactive:
        choices = [
            questionary.Choice(f"{i.name:<{width}}  {fmt_bytes(i.left):>9}  {i.reason}", value=i)
            if i.status == "ready"
            else questionary.Choice(f"{i.name:<{width}}", value=i, disabled=i.reason)
            for i in items
        ]
        selected = questionary.checkbox("Select what to download (space to toggle, enter to confirm):",
                                        choices=choices).ask()
        if not selected:
            print("Nothing selected.")
            return 0
    else:
        print("\nNot running in a terminal: pass --all (and --yes) to download without the checkbox list.")
        return 2

    estimate(selected, args.bandwidth)
    if not args.yes:
        if not interactive:
            print("Not running in a terminal: pass --yes to start.")
            return 2
        if not questionary.confirm("Start download?", default=True).ask():
            return 0

    failed = [i.name for i in selected if not download(api, i)]
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
