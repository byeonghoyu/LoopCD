from __future__ import annotations
import argparse, os, sys
from pathlib import Path


OLD_BLOCK = (
    "class HuginnDynamicCache(DynamicCache):\n"
    "    def __init__(self, lookup_strategy: str = \"full\") -> None:\n"
    "        super().__init__()\n"
)
NEW_BLOCK = (
    "class HuginnDynamicCache(DynamicCache):\n"
    "    # transformers>=4.54 made key_cache/value_cache read-only properties\n"
    "    # on Cache; shadow them at class level so __init__ can assign dicts.\n"
    "    key_cache = None\n"
    "    value_cache = None\n"
    "    def __init__(self, lookup_strategy: str = \"full\") -> None:\n"
    "        super().__init__()\n"
)


def _find_files(model: str, hf_home: Path) -> list[Path]:
    org, name = model.split("/")
    paths = []
    hub_dir = hf_home / "hub" / f"models--{org}--{name}"
    if hub_dir.exists():
        paths.extend(hub_dir.rglob("raven_modeling_minimal.py"))
    mod_dir = hf_home / "modules" / "transformers_modules" / org / name
    if mod_dir.exists():
        paths.extend(mod_dir.rglob("raven_modeling_minimal.py"))
    return [p.resolve() for p in paths if p.is_file()]


def _patch_one(path: Path, dry_run: bool = False) -> str:
    src = path.read_text()
    if NEW_BLOCK in src:
        return "already-patched"
    if OLD_BLOCK not in src:
        return "unknown-format"
    if dry_run:
        return "would-patch"
    bak = path.with_suffix(path.suffix + ".bak_pre_patch")
    if not bak.exists():
        bak.write_text(src)
    path.write_text(src.replace(OLD_BLOCK, NEW_BLOCK))
    return "patched"


def main():
    ap = argparse.ArgumentParser(description="Apply the key_cache shadow patch to Huginn's modeling.py.")
    ap.add_argument("--model", default="tomg-group-umd/huginn-0125",
                    help="HF repo id (default: tomg-group-umd/huginn-0125).")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
                    help="HF cache root (default: $HF_HOME or ~/.cache/huggingface).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing.")
    args = ap.parse_args()

    hf_home = Path(args.hf_home).expanduser().resolve()
    print(f"[config] model    : {args.model}")
    print(f"[config] HF_HOME  : {hf_home}")
    print(f"[config] dry-run  : {args.dry_run}")
    print()

    files = _find_files(args.model, hf_home)
    if not files:
        print(f"[error] no raven_modeling_minimal.py found under {hf_home}.")
        print(f"        Run `AutoModelForCausalLM.from_pretrained('{args.model}', trust_remote_code=True)` first to download.")
        sys.exit(2)

    print(f"[found] {len(files)} file(s) to inspect:")
    for f in files:
        print(f"  - {f}")
    print()

    n_patched = n_skipped = n_unknown = 0
    for f in files:
        status = _patch_one(f, dry_run=args.dry_run)
        print(f"  [{status:>16s}] {f}")
        if status == "patched" or status == "would-patch": n_patched += 1
        elif status == "already-patched":                   n_skipped += 1
        elif status == "unknown-format":                    n_unknown += 1

    print()
    print(f"[summary] patched={n_patched}  already-patched={n_skipped}  unknown-format={n_unknown}")
    if n_unknown:
        print(f"[warn] {n_unknown} file(s) had an unrecognized format — patch not applied.")
        print(f"       Likely the upstream model code changed. Inspect manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
