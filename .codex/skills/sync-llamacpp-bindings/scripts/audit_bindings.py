#!/usr/bin/env python3
"""Read-only audit of xllamacpp Cython fields/enums against native Clang headers."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


class AuditError(Exception):
    """The audit could not be completed."""


@dataclass
class Record:
    fields: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class Bindings:
    headers: set[str] = field(default_factory=set)
    records: dict[str, Record] = field(default_factory=dict)
    enums: dict[str, set[str]] = field(default_factory=dict)
    constants: set[str] = field(default_factory=set)


def parse_pxd(path: Path) -> Bindings:
    bindings = Bindings()
    owner = None
    kind = None
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        header = re.match(r'^cdef extern from "([^"]+)"', line)
        if header:
            bindings.headers.add(header[1])
        declaration = re.match(
            r"^    (ctypedef struct|cdef cppclass|cpdef enum|cdef enum)(?:\s+(\w+))?\s*:",
            line,
        )
        if declaration:
            kind, owner = declaration.groups()
            if "enum" in kind:
                if owner:
                    bindings.enums[owner] = set()
                else:
                    owner = "<anonymous>"
            elif owner:
                bindings.records[owner] = Record()
            continue
        if re.match(
            r"^    (?:ctypedef (?:struct|union)|cdef cppclass|cpdef enum|cdef enum)\b",
            line,
        ):
            raise AuditError(
                f"Unsupported type declaration at {path}:{number}: {line.strip()}"
            )
        if line.strip() and not line.startswith("        "):
            owner = None
        if not owner or not line.strip() or line.strip() == "pass":
            continue
        if "enum" in kind:
            member = line.strip().split("=", 1)[0].strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", member):
                raise AuditError(
                    f"Unsupported enum syntax at {path}:{number}: {line.strip()}"
                )
            if owner == "<anonymous>":
                bindings.constants.add(member)
            else:
                bindings.enums[owner].add(member)
        elif "(" not in line:
            member = re.fullmatch(r"        (.+?)\s*\b(\w+)((?:\[[^]]+\])*)", line)
            if not member or not member[1].strip():
                raise AuditError(
                    f"Unsupported field syntax at {path}:{number}: {line.strip()}"
                )
            ctype, name, array = member.groups()
            bindings.records[owner].fields[name] = (ctype.strip(), array)
    if not bindings.headers or not any(
        record.fields for record in bindings.records.values()
    ):
        raise AuditError(f"No auditable extern fields parsed from {path}")
    return bindings


def json_documents(text: str):
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        node, offset = decoder.raw_decode(text, offset)
        yield node


def cpp_type(ctype: str, array: str, enum_names: set[str]) -> str:
    for alias, native in {
        "std_string": "std::string",
        "std_vector": "std::vector",
        "std_map": "std::map",
        "std_set": "std::set",
        "bint": "bool",
    }.items():
        ctype = re.sub(rf"\b{alias}\b", native, ctype)
    # C APIs may have a function and enum with the same spelling.
    for name in enum_names:
        ctype = re.sub(rf"(?<!enum )\b{re.escape(name)}\b", f"enum {name}", ctype)
    return ctype.replace("[", "<").replace("]", ">") + array


def native_schema(command: list[str], source: Path, names: set[str]):
    records = {}
    methods = {}
    enums = {}
    filters = {
        (
            re.match(r"(?:common|ggml|llama)_", name)[0]
            if re.match(r"(?:common|ggml|llama)_", name)
            else name
        )
        for name in names
    }
    for substring in sorted(filters):
        result = subprocess.run(
            [
                *command,
                "-Xclang",
                "-ast-dump=json",
                "-Xclang",
                f"-ast-dump-filter={substring}",
                str(source),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise AuditError(f"Clang could not parse native headers:\n{result.stderr}")
        for node in json_documents(result.stdout):
            name = node.get("name")
            if name not in names:
                continue
            if node.get("kind") in {"RecordDecl", "CXXRecordDecl"} and node.get(
                "completeDefinition"
            ):
                fields = {}

                def collect(value):
                    for child in value.get("inner", []):
                        if child.get("kind") == "FieldDecl" and child.get("name"):
                            fields[child["name"]] = child["type"]["qualType"]
                        elif child.get("kind") in {
                            "RecordDecl",
                            "CXXRecordDecl",
                        } and not child.get("name"):
                            collect(child)

                collect(node)
                records[name] = fields
                methods[name] = {
                    child["name"]
                    for child in node.get("inner", [])
                    if child.get("kind") == "CXXMethodDecl" and child.get("name")
                }
            elif node.get("kind") == "EnumDecl" and node.get("inner"):
                enums[name] = {
                    child["name"]
                    for child in node["inner"]
                    if child.get("kind") == "EnumConstantDecl"
                }
    return records, methods, enums


def wrapper_accesses(path: Path, records: dict, methods: dict):
    findings = []
    count = 0
    classes = 0
    text = path.read_text()
    for declaration in re.findall(r"^cdef class .+$", text, re.M):
        if not re.fullmatch(r"cdef class \w+:", declaration):
            raise AuditError(
                f"Unsupported wrapper declaration in {path}: {declaration}"
            )
    for match in re.finditer(
        r"^cdef class (\w+):\n(.*?)(?=^cdef class |\Z)", text, re.M | re.S
    ):
        name, body = match.groups()
        owner = re.search(r"^    cdef xllamacpp\.(\w+)\s*\*?p\b", body, re.M)
        if not owner:
            if re.search(r"\bself\.p\.", body):
                raise AuditError(f"Cannot resolve self.p owner in {path}: {name}")
            continue
        if owner[1] not in records:
            raise AuditError(
                f"No complete native definition for wrapper {name}: {owner[1]}"
            )
        code = "\n".join(
            line for line in body.splitlines() if not line.lstrip().startswith("#")
        )
        names = set(re.findall(r"\bself\.p\.(\w+)", code))
        for member in sorted(names - records[owner[1]].keys() - methods[owner[1]]):
            findings.append(f"UNKNOWN PYTHON ACCESS: {name} -> {owner[1]}.{member}")
        count += len(names)
        classes += 1
    if not classes:
        raise AuditError(f"No supported Python wrapper classes parsed from {path}")
    return findings, count, classes


def audit(repo: Path, pxd: Path, pyx: Path, clang: str):
    bindings = parse_pxd(pxd)
    vendor = repo / "thirdparty" / "llama.cpp"
    include_dirs = [
        vendor / "common",
        vendor / "include",
        vendor / "ggml" / "include",
        vendor,
        vendor / "build" / "common",
    ]
    if not (vendor / "common" / "common.h").is_file():
        raise AuditError(
            f"Missing vendored headers under {vendor}; initialize the checkout first"
        )
    command = [clang, "-std=c++17", "-fsyntax-only", "-ferror-limit=0"]
    command.extend(f"-I{path}" for path in include_dirs)
    includes = "".join(f'#include "{header}"\n' for header in sorted(bindings.headers))
    findings = []
    notes = []
    field_count = 0
    enum_count = 0
    with tempfile.TemporaryDirectory(prefix="xllamacpp-binding-audit-") as temp:
        source = Path(temp) / "audit.cpp"
        source.write_text(includes)
        names = {name for name, record in bindings.records.items() if record.fields}
        names.update(name for name, members in bindings.enums.items() if members)
        records, methods, enums = native_schema(command, source, names)
        checks = [includes, "#include <type_traits>\n#include <utility>\n"]
        for owner, binding in sorted(bindings.records.items()):
            if not binding.fields:
                continue
            if owner not in records:
                findings.append(f"MISSING NATIVE TYPE: {owner}")
                continue
            for member, (ctype, array) in sorted(binding.fields.items()):
                field_count += 1
                if member not in records[owner]:
                    findings.append(f"REMOVED FIELD: {owner}.{member}")
                    continue
                checks.append(
                    f"static_assert(std::is_same<decltype(std::declval<{owner}>().{member}), "
                    f'{cpp_type(ctype, array, set(bindings.enums))}>::value, "TYPE MISMATCH: {owner}.{member}");\n'
                )
            extra = records[owner].keys() - binding.fields.keys()
            if extra:
                notes.append(
                    f"UNBOUND NATIVE FIELDS (review): {owner}: {', '.join(sorted(extra))}"
                )
        for owner, members in sorted(bindings.enums.items()):
            if not members:
                continue
            enum_count += 1
            if owner not in enums:
                findings.append(f"MISSING NATIVE ENUM: {owner}")
                continue
            for member in sorted(members - enums[owner]):
                findings.append(f"REMOVED ENUM MEMBER: {owner}.{member}")
            for member in sorted(enums[owner] - members):
                findings.append(f"UNBOUND ENUM MEMBER: {owner}.{member}")
        for constant in sorted(bindings.constants):
            checks.append(
                f'static_assert(sizeof({constant}) > 0, "CONSTANT: {constant}");\n'
            )
        source.write_text("".join(checks))
        result = subprocess.run([*command, str(source)], capture_output=True, text=True)
        if result.returncode:
            findings.append(
                "NATIVE TYPE/CONSTANT CHECK FAILED:\n" + result.stderr.strip()
            )
        wrapper_findings, accesses, classes = wrapper_accesses(pyx, records, methods)
        findings.extend(wrapper_findings)
    return {
        "repo": str(repo),
        "pxd": str(pxd),
        "pyx": str(pyx),
        "fields": field_count,
        "named_enums": enum_count,
        "constants": len(bindings.constants),
        "wrapper_accesses": accesses,
        "wrapper_classes": classes,
        "findings": findings,
        "review_notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--pxd", type=Path, help="Override the Cython declaration file")
    parser.add_argument("--pyx", type=Path, help="Override the Python wrapper file")
    parser.add_argument("--clang", default=os.environ.get("CLANGXX", "clang++"))
    parser.add_argument("--json", type=Path, help="Write the audit report as JSON")
    args = parser.parse_args()
    try:
        repo = args.repo.resolve()
        pxd = (args.pxd or repo / "src/xllamacpp/xllamacpp.pxd").resolve()
        pyx = (args.pyx or repo / "src/xllamacpp/xllamacpp.pyx").resolve()
        clang = shutil.which(args.clang)
        if not clang:
            raise AuditError(
                f"Clang C++ compiler not found: {args.clang}; select one with --clang"
            )
        report = audit(repo, pxd, pyx, clang)
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"Audited {report['fields']} fields, {report['named_enums']} named enums, "
            f"{report['constants']} constants, and {report['wrapper_accesses']} wrapper accesses "
            f"across {report['wrapper_classes']} classes."
        )
        for finding in report["findings"]:
            print(finding)
        for note in report["review_notes"]:
            print(note)
        print(
            f"{len(report['findings'])} findings; {len(report['review_notes'])} native field groups need review."
        )
        return 1 if report["findings"] else 0
    except (AuditError, OSError, ValueError) as error:
        print(f"Audit incomplete: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
