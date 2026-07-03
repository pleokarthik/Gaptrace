"""Guardrail: `import ragradar_evaluate` must stay cheap. The benchmark
builder pulls in scipy and the output-quality layer pulls in ragas —
both must load lazily, only when the functions needing them run."""

import json
import subprocess
import sys


def test_import_ragradar_evaluate_does_not_import_scipy_or_ragas():
    code = (
        "import sys, json\n"
        "import ragradar_evaluate\n"
        "print(json.dumps(sorted(m.split('.')[0] for m in sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    modules = set(json.loads(result.stdout))
    assert "scipy" not in modules, "import ragradar_evaluate pulled in scipy"
    assert "ragas" not in modules, "import ragradar_evaluate pulled in ragas"
