import importlib.util
import pathlib
import sys
import tempfile
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


index = load_script("build-wheel-index")
migration = load_script("migrate-release-assets")
DIGEST = "sha256:" + "a" * 64


def asset(repo, **overrides):
    result = {
        "name": "xllamacpp-2026.4.8672.1-cp310-abi3-linux_x86_64.whl",
        "state": "uploaded", "size": 100, "digest": DIGEST,
        "browser_download_url": f"https://github.com/xorbitsai/{repo}/releases/download/v2026.4.8672.1-cu128/wheel.whl",
    }
    result.update(overrides)
    return result


def release(repo, **overrides):
    result = {"tag_name": "v2026.4.8672.1-cu128", "draft": False, "assets": [asset(repo)]}
    result.update(overrides)
    return result


class WheelIndexTests(unittest.TestCase):
    def test_active_asset_wins_and_keeps_sha256(self):
        wheels = index.collect_wheels([[release("legacy")], [release("active")]])
        self.assertEqual(len(wheels), 1)
        url = next(iter(wheels.values()))["url"]
        self.assertIn("/active/", url)
        self.assertTrue(url.endswith("#sha256=" + "a" * 64))

    def test_incomplete_upload_keeps_legacy_download(self):
        incomplete = release("active", assets=[asset("active", state="starter", size=0)])
        wheels = index.collect_wheels([[release("legacy")], [incomplete]])
        self.assertIn("/legacy/", next(iter(wheels.values()))["url"])

    def test_partially_copied_release_keeps_missing_assets(self):
        legacy = release("legacy", assets=[asset("legacy"), asset("legacy", name="other.whl")])
        wheels = index.collect_wheels([[legacy], [release("active")]])
        self.assertEqual(len(wheels), 2)
        self.assertIn("/legacy/", wheels["cu128", "other.whl"]["url"])

    def test_drafts_and_non_wheels_are_excluded(self):
        wheels = index.collect_wheels([[release("legacy", draft=True), release("legacy", assets=[asset("legacy", name="notes.txt")])]])
        self.assertEqual(wheels, {})

    def test_historical_channels_and_patch_versions(self):
        self.assertEqual(index.channel_for_tag("v2026.4.8672.1-cu128"), ("cu128", (2026, 4, 8672, 1)))
        self.assertEqual(index.channel_for_tag("v0.1.14-rocm-6.0.2")[0], "rocm-6.0.2")
        self.assertEqual(index.channel_for_tag("v2026.10.11549-vulkan-windows-11-arm")[0], "vulkan")
        self.assertIsNone(index.channel_for_tag("unrelated"))

    def test_generated_package_page_uses_working_fallback_url(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            index.build_index([[release("xllamacpp-legacy")], []], output)
            self.assertIn('href="xllamacpp/"', (output / "whl/cu128/index.html").read_text())
            page = (output / "whl/cu128/xllamacpp/index.html").read_text()
            self.assertIn("/xllamacpp-legacy/releases/", page)
            self.assertIn("#sha256=", page)

    def test_empty_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                index.build_index([[]], pathlib.Path(temporary))


class ReleaseMigrationTests(unittest.TestCase):
    def test_digest_and_size_must_match(self):
        original = asset("legacy")
        self.assertTrue(migration.asset_matches(original, asset("active")))
        self.assertFalse(migration.asset_matches(original, asset("active", size=99)))
        self.assertFalse(migration.asset_matches(original, asset("active", digest="sha256:" + "b" * 64)))
        self.assertFalse(migration.asset_matches(original, asset("active", state="starter")))

    def test_computed_digest_protects_assets_without_source_digest(self):
        original = asset("legacy", digest=None)
        self.assertTrue(migration.asset_matches(original, asset("active"), DIGEST))
        self.assertFalse(migration.asset_matches(original, asset("active", digest="sha256:" + "b" * 64), DIGEST))

    def test_partition_keeps_every_release_once(self):
        releases = [release("legacy", tag_name=str(i), assets=[asset("legacy", size=size)]) for i, size in enumerate([100, 80, 60, 40, 20, 0])]
        shards = migration.partition(releases, 3)
        self.assertEqual(sorted(r["tag_name"] for group in shards for r in group), [str(i) for i in range(6)])
        self.assertTrue(all(group for group in shards))
        self.assertEqual([sum(a["size"] for r in group for a in r["assets"]) for group in shards], [100, 100, 100])


if __name__ == "__main__":
    unittest.main()
