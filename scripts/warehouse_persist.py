from __future__ import annotations

import argparse
import json

from config.settings import settings
from intraday_engine.storage.warehouse import persist_warehouse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the DuckDB research database into a partitioned Parquet warehouse"
    )
    parser.add_argument("--source", default=settings.duckdb_path, help="Source DuckDB file")
    parser.add_argument("--root", default=settings.warehouse_root, help="Warehouse root directory")
    args = parser.parse_args()

    summary = persist_warehouse(args.source, args.root)
    print(json.dumps(summary, indent=2, default=str))
    print(f"persisted {len(summary['written'])} partition(s), skipped {len(summary['skipped'])} unchanged, root={args.root}")


if __name__ == "__main__":
    main()
