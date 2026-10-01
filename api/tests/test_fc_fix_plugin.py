"""
Tests for finding #9 (face-clusters review):

    plugin/stash-sense-face-groups.js:188 merge target is first-clicked not
    largest (comment wrong); clicking open then assigned group deletes the
    assigned group and its performer link.

These run the plugin's pure `chooseMergeTarget` selection logic inside a
Node.js `vm` sandbox that stubs just enough of `window.StashSense` for the
module to attach itself, then exercises `window.StashSenseFaceGroups.chooseMergeTarget`
via a small inline JS harness. Skipped if `node` is not on PATH.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PLUGIN_JS = Path(__file__).resolve().parents[2] / "plugin" / "stash-sense-face-groups.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not available")


HARNESS_TEMPLATE = """
const fs = require('fs');
const vm = require('vm');

const src = fs.readFileSync(process.argv[2], 'utf8');

const window = {
  StashSense: {
    getRoute: () => ({ type: 'other' }),
    onNavigate() {},
    onLeavePlugin() {},
    PLUGIN_NAME: 'Stash Sense',
    escapeHtml: s => s,
    getSettings: async () => ({}),
    runPluginOperation: async () => ({}),
  },
};

const sandbox = {
  window,
  console,
  setTimeout,
  clearTimeout,
  clearInterval,
};
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'stash-sense-face-groups.js' });

const cases = JSON.parse(process.argv[3]);
const results = cases.map(clusters => window.StashSenseFaceGroups.chooseMergeTarget(clusters));
console.log(JSON.stringify(results));
"""


def _run_choose_merge_target(cases):
    """Run chooseMergeTarget(clusters) for each entry in `cases`, return list of results."""
    # private temp dir: no fixed path in the repo, safe under parallel runs
    with tempfile.TemporaryDirectory() as tmp:
        harness_path = Path(tmp) / "harness.js"
        harness_path.write_text(HARNESS_TEMPLATE, encoding="utf-8")
        proc = subprocess.run(
            ["node", str(harness_path), str(PLUGIN_JS), json.dumps(cases)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert proc.returncode == 0, f"node harness failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_picks_largest_regardless_of_click_order():
    small = {"id": 1, "status": "open", "face_count": 3, "performer_id": None, "performer_name": None, "name": None}
    large = {"id": 2, "status": "open", "face_count": 30, "performer_id": None, "performer_name": None, "name": None}
    [result] = _run_choose_merge_target([[small, large]])
    assert result["targetId"] == 2
    assert result["sourceIds"] == [1]
    assert result["error"] is None


def test_picks_assigned_even_if_smaller():
    open_large = {"id": 1, "status": "open", "face_count": 50, "performer_id": None, "performer_name": None, "name": None}
    assigned_small = {"id": 2, "status": "assigned", "face_count": 4, "performer_id": "p1", "performer_name": "Alice", "name": None}
    [result] = _run_choose_merge_target([[open_large, assigned_small]])
    assert result["targetId"] == 2
    assert result["sourceIds"] == [1]
    assert result["error"] is None


def test_rejects_two_assigned_different_performers():
    a = {"id": 1, "status": "assigned", "face_count": 10, "performer_id": "p1", "performer_name": "Alice", "name": None}
    b = {"id": 2, "status": "assigned", "face_count": 20, "performer_id": "p2", "performer_name": "Bob", "name": None}
    [result] = _run_choose_merge_target([[a, b]])
    assert result["targetId"] is None
    assert "Alice" in result["error"]
    assert "Bob" in result["error"]


def test_two_assigned_same_performer_picks_larger():
    a = {"id": 1, "status": "assigned", "face_count": 10, "performer_id": "p1", "performer_name": "Alice", "name": None}
    b = {"id": 2, "status": "assigned", "face_count": 20, "performer_id": "p1", "performer_name": "Alice", "name": None}
    [result] = _run_choose_merge_target([[a, b]])
    assert result["targetId"] == 2
    assert result["sourceIds"] == [1]
    assert result["error"] is None


def test_tie_breaks_to_lowest_id():
    a = {"id": 5, "status": "open", "face_count": 10, "performer_id": None, "performer_name": None, "name": None}
    b = {"id": 2, "status": "open", "face_count": 10, "performer_id": None, "performer_name": None, "name": None}
    [result] = _run_choose_merge_target([[a, b]])
    assert result["targetId"] == 2
    assert result["sourceIds"] == [5]
    assert result["error"] is None


def test_requires_two():
    a = {"id": 1, "status": "open", "face_count": 10, "performer_id": None, "performer_name": None, "name": None}
    [result] = _run_choose_merge_target([[a]])
    assert result["targetId"] is None
    assert result["error"] == "Select at least two groups"


def test_source_no_longer_uses_first_clicked():
    src = PLUGIN_JS.read_text(encoding="utf-8")
    assert "ids[0]" not in src
    assert "largest is first in list order" not in src
