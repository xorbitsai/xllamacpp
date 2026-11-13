#!/usr/bin/env python3
"""Integration checks for real native API drift in temporary fixture projects."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


AUDITOR = Path(__file__).with_name("audit_bindings.py")
HEADER = """#pragma once
#include <cstdint>
#include <string>
#include <vector>
#define CURRENT_LIMIT 7
enum common_mode { MODE_OFF, MODE_ON };
enum common_mode common_mode();
struct common_params {
    int32_t count = 0;
    float share = 0;
    bool enabled = false;
    char label[CURRENT_LIMIT];
    std::vector<std::string> shards;
    enum common_mode mode;
    union { int64_t integer; double real; };
    std::vector<std::string> get_paths() const {
        std::vector<std::string> paths = shards;
        return paths;
    }
};
"""
PXD = """cdef extern from "common.h":
    cpdef enum:
        CURRENT_LIMIT
    cpdef enum common_mode:
        MODE_OFF
        MODE_ON
    ctypedef struct common_params:
        int32_t count
        float share
        bint enabled
        char label[CURRENT_LIMIT]
        std_vector[std_string] shards
        common_mode mode
        int64_t integer
        double real
"""
PYX = """cdef class CommonParams:
    cdef xllamacpp.common_params p

    @property
    def count(self) -> int:
        return self.p.count
"""


@unittest.skipUnless(shutil.which("clang++"), "requires Clang C++")
class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="binding-audit-test-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        include = self.repo / "thirdparty/llama.cpp/common"
        include.mkdir(parents=True)
        self.header = include / "common.h"
        self.header.write_text(HEADER)
        source = self.repo / "src/xllamacpp"
        source.mkdir(parents=True)
        self.pxd = source / "xllamacpp.pxd"
        self.pyx = source / "xllamacpp.pyx"
        self.pxd.write_text(PXD)
        self.pyx.write_text(PYX)

    def run_audit(self, *args):
        return subprocess.run(
            [sys.executable, str(AUDITOR), "--repo", str(self.repo), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_exact_types_arrays_templates_union_and_enum_name_collision(self):
        report = self.repo / "report.json"
        result = self.run_audit("--json", str(report))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        data = json.loads(report.read_text())
        self.assertEqual(data["fields"], 8)
        self.assertEqual(data["findings"], [])
        self.assertEqual(data["review_notes"], [])  # get_paths' local is not a field.

    def test_unused_removed_field_is_detected(self):
        self.pxd.write_text(PXD + "        bint at_start\n")
        result = self.run_audit()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("REMOVED FIELD: common_params.at_start", result.stdout)

    def test_type_change_is_detected(self):
        self.pxd.write_text(PXD.replace("float share", "double share"))
        result = self.run_audit()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("TYPE MISMATCH: common_params.share", result.stdout)

    def test_array_extent_change_is_detected(self):
        self.pxd.write_text(PXD.replace("label[CURRENT_LIMIT]", "label[8]"))
        result = self.run_audit()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("TYPE MISMATCH: common_params.label", result.stdout)

    def test_missing_and_removed_enum_members_are_detected(self):
        self.pxd.write_text(PXD.replace("        MODE_ON", "        MODE_REMOVED"))
        result = self.run_audit()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("UNBOUND ENUM MEMBER: common_mode.MODE_ON", result.stdout)
        self.assertIn("REMOVED ENUM MEMBER: common_mode.MODE_REMOVED", result.stdout)

    def test_stale_python_property_access_is_detected(self):
        self.pyx.write_text(PYX.replace("self.p.count", "self.p.defrag_thold"))
        result = self.run_audit()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(
            "UNKNOWN PYTHON ACCESS: CommonParams -> common_params.defrag_thold",
            result.stdout,
        )

    def test_new_native_fields_are_review_notes(self):
        self.header.write_text(
            HEADER.replace("float share = 0;", "float share = 0;\n    int new_field;")
        )
        result = self.run_audit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "UNBOUND NATIVE FIELDS (review): common_params: new_field", result.stdout
        )

    def test_missing_compiler_is_incomplete(self):
        result = self.run_audit("--clang", "missing-clang-for-audit-test")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Audit incomplete", result.stderr)

    def test_missing_header_is_incomplete(self):
        self.header.unlink()
        result = self.run_audit()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Missing vendored headers", result.stderr)

    def test_unsupported_declaration_is_incomplete(self):
        self.pxd.write_text(
            PXD + "    ctypedef union unsupported:\n        int value\n"
        )
        result = self.run_audit()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported type declaration", result.stderr)

    def test_removed_constant_is_detected(self):
        self.pxd.write_text(
            PXD.replace("        CURRENT_LIMIT", "        REMOVED_LIMIT")
        )
        result = self.run_audit()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("REMOVED_LIMIT", result.stdout)


if __name__ == "__main__":
    unittest.main()
