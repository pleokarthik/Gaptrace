"""Guardrail for ragradar-core's zero-dependency contract: importing every
ragradar_core module must pull in nothing outside the standard library."""

import json
import subprocess
import sys


def test_import_ragradar_core_pulls_only_stdlib():
    code = (
        "import sys, json\n"
        "before = set(sys.modules)\n"
        "import ragradar_core, ragradar_core.schema, ragradar_core.store, ragradar_core.targets\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "print(json.dumps(sorted(new)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    imported = set(json.loads(result.stdout))
    allowed = set(sys.stdlib_module_names) | {"ragradar_core"}
    # Editable-install path hooks (e.g. "__editable___ragradar_core_...") are
    # packaging machinery, not runtime dependencies.
    extras = {m for m in imported if m not in allowed and not m.startswith("__editable__")}
    assert not extras, f"ragradar_core imported non-stdlib modules: {sorted(extras)}"
