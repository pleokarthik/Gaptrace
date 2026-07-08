"""Guardrail for gaptrace-core's zero-dependency contract: importing every
gaptrace_core module must pull in nothing outside the standard library."""

import json
import subprocess
import sys


def test_import_gaptrace_core_pulls_only_stdlib():
    code = (
        "import sys, json\n"
        "before = set(sys.modules)\n"
        "import gaptrace_core, gaptrace_core.schema, gaptrace_core.store, gaptrace_core.targets\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "print(json.dumps(sorted(new)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    imported = set(json.loads(result.stdout))
    allowed = set(sys.stdlib_module_names) | {"gaptrace_core"}
    # Editable-install path hooks (e.g. "__editable___gaptrace_core_...") are
    # packaging machinery, not runtime dependencies.
    extras = {m for m in imported if m not in allowed and not m.startswith("__editable__")}
    assert not extras, f"gaptrace_core imported non-stdlib modules: {sorted(extras)}"
