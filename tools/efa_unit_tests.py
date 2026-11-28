#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify a run of libfabric's efa unit tests against a list of expected failures.

    tools/efa_unit_tests.py check RESULTS.xml EXPECTED
    tools/efa_unit_tests.py no-device RESULTS.xml
    tools/efa_unit_tests.py fails RESULTS.xml TEST

RESULTS.xml is what efa_unit_test writes (cmocka's JUnit output). EXPECTED lists, one per line, a test
name and after it the reason it is expected to fail on this stand-in; '#' starts a comment.

check fails when a test fails that EXPECTED does not list, and when a listed test passes: a list that
names a test which now passes no longer says why the rest fail. no-device is the negative control: it
fails unless the run found no device, so that a later green run is known to come from the stand-in.
fails is the mutation control: it fails unless TEST failed, on a stand-in built wrong on purpose.
"""
import re
import sys


def results(path):
    text = open(path, errors="replace").read()
    out = {}
    for name, body in re.findall(r'<testcase name="([^"]+)"[^>]*>(.*?)</testcase>', text, re.S):
        if "<skipped" in body:
            out[name] = "skipped"
        elif "<failure" in body or "<error" in body:
            out[name] = "failed"
        else:
            out[name] = "passed"
    if not out:
        sys.exit(f"{path}: no test cases")
    return out


def expected(path):
    out = {}
    for n, line in enumerate(open(path), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, reason = line.partition(" ")
        if not reason.strip():
            sys.exit(f"{path}:{n}: {name} has no reason")
        out[name] = reason.strip()
    return out


def counts(res):
    c = {"passed": 0, "failed": 0, "skipped": 0}
    for v in res.values():
        c[v] += 1
    return c


def main(argv):
    if len(argv) == 4 and argv[1] == "check":
        res, exp = results(argv[2]), expected(argv[3])
        c = counts(res)
        print(f"{len(res)} tests: {c['passed']} passed, {c['failed']} failed, {c['skipped']} skipped")
        bad = 0
        for name, state in sorted(res.items()):
            if state == "failed" and name not in exp:
                print(f"UNEXPECTED FAILURE {name}")
                bad = 1
            elif state == "passed" and name in exp:
                print(f"NOW PASSES {name} (remove it from {argv[3]})")
                bad = 1
        for name in sorted(set(exp) - set(res)):
            print(f"NOT RUN {name} (listed in {argv[3]})")
            bad = 1
        for name, reason in sorted(exp.items()):
            if res.get(name) == "failed":
                print(f"expected failure {name}: {reason}")
        return bad
    if len(argv) == 3 and argv[1] == "no-device":
        res = results(argv[2])
        c = counts(res)
        print(f"without the stand-in: {len(res)} tests, {c['passed']} passed, {c['failed']} failed")
        if c["passed"] > c["failed"]:
            print("more tests passed than failed without a device: this host has one, or the control is broken")
            return 1
        return 0
    if len(argv) == 4 and argv[1] == "fails":
        state = results(argv[2]).get(argv[3])
        print(f"{argv[3]}: {state}")
        return 0 if state == "failed" else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
