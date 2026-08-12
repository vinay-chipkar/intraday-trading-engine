from intraday_engine.storage.warehouse.manifest import (
    load_manifest,
    load_schema_version,
    save_manifest,
    save_schema_version,
    sha256_of_file,
)


def test_sha256_is_deterministic_and_content_sensitive(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    c = tmp_path / "c.bin"
    a.write_bytes(b"hello warehouse")
    b.write_bytes(b"hello warehouse")
    c.write_bytes(b"hello warehouse!")

    assert sha256_of_file(a) == sha256_of_file(b)
    assert sha256_of_file(a) != sha256_of_file(c)
    assert len(sha256_of_file(a)) == 64  # hex-encoded sha256


def test_manifest_round_trips(tmp_path):
    manifest = {
        "raw/candles/date=2026-08-11/part.parquet": {
            "table": "candles", "zone": "raw", "partition": "date=2026-08-11",
            "rows": 2, "sha256": "abc123",
        }
    }
    save_manifest(tmp_path, manifest)
    assert load_manifest(tmp_path) == manifest


def test_load_manifest_defaults_to_empty_dict_when_missing(tmp_path):
    assert load_manifest(tmp_path) == {}


def test_schema_version_round_trips(tmp_path):
    save_schema_version(tmp_path, 3)
    assert load_schema_version(tmp_path) == {"schema_version": 3}


def test_load_schema_version_is_none_when_missing(tmp_path):
    assert load_schema_version(tmp_path) is None
