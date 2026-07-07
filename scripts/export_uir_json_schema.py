"""scripts/export_uir_json_schema.py -- Write UIR v1.0 JSON Schema to disk.

Usage:
    python scripts/export_uir_json_schema.py                # -> docs/uir.schema.json
    python scripts/export_uir_json_schema.py custom/path.json

This is the on-disk artifact used for documentation reference and for
any downstream tooling that wants to consume the UIR contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/export_uir_json_schema.py` from project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uir_pipeline.uir_schema import schema_json_dict  # noqa: E402

# Resolve default output relative to this file so the script works
# regardless of the current working directory.
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "docs" / "uir.schema.json"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(schema_json_dict(), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
