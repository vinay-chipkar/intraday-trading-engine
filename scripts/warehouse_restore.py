from __future__ import annotations

import argparse
import json

from config.settings import settings
from intraday_engine.storage.warehouse import restore_warehouse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore a partitioned Parquet warehouse into a fresh DuckDB"
    )
    parser.add_argument("--root", default=settings.warehouse_root, help="Warehouse root directory")
    parser.add_argument("--target", default=settings.duckdb_path, help="Target DuckDB file (overwritten)")
    args = parser.parse_args()

    restored = restore_warehouse(args.root, args.target)
    print(json.dumps(restored, indent=2, default=str))


if __name__ == "__main__":
    main()
