"""
Catalog smell-coverage tests.

These assert — through the AGENT wrappers ``detect_python_smells`` /
``detect_ml_smells`` — that each major catalog smell is actually reachable and
emitted by name. They are offline and deterministic: small crafted ``.py``
files are written to a temp dir and analysed directly (no LLM involved).

Special focus: the five smells that were previously *dead* through the agent
because of upstream ``code_quality_analyzer`` bugs/omissions and are now wired
or supplemented in-repo:

  * Lazy Class, Dead Code, Data Class  (defined upstream, never dispatched)
  * Switch Statements                  (upstream branch counter never recursed)
  * Deep Inheritance Tree (DIT)        (upstream graph used bare vs qualified names)
"""

from __future__ import annotations

import pytest

from code_review_agent.tools import detect_ml_smells, detect_python_smells


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _names(result: dict, key: str) -> set[str]:
    """Collect smell names from a list-valued field of a wrapper result."""
    value = result.get(key)
    if not isinstance(value, list):
        return set()
    return {s.get("name", "") for s in value if isinstance(s, dict)}


def _ml_text(result: dict) -> str:
    """Flatten all ML smell names + descriptions into one searchable blob."""
    parts: list[str] = []
    for group in ("framework_smells", "huggingface_smells", "general_ml_smells"):
        for entry in result.get(group, []):
            for smell in entry.get("smells", []):
                parts.append(str(smell.get("name", "")))
                parts.append(str(smell.get("description", "")))
                parts.append(str(smell.get("message", "")))
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Crafted sample project (non-ML smells)
# ---------------------------------------------------------------------------

def _write_code_project(root) -> str:
    root = root / "proj"
    root.mkdir()

    # Dead Code: 3 private uncalled functions (upstream only flags private ones).
    (root / "dead.py").write_text(
        "def _a():\n    return 1\n\n\n"
        "def _b():\n    return 2\n\n\n"
        "def _c():\n    return 3\n\n\n"
        "def main():\n    return 0\n",
        encoding="utf-8",
    )

    # Data Class: 5 getter/setter pairs, no other methods.
    (root / "data_class.py").write_text(
        "class DataBag:\n"
        "    def get_a(self):\n        return self.a\n\n"
        "    def get_b(self):\n        return self.b\n\n"
        "    def get_c(self):\n        return self.c\n\n"
        "    def set_a(self, v):\n        self.a = v\n\n"
        "    def set_b(self, v):\n        self.b = v\n",
        encoding="utf-8",
    )

    # Lazy Class: a class with one tiny non-trivial method.
    (root / "lazy.py").write_text(
        "class Lazy:\n"
        "    def only(self):\n"
        "        x = 1\n"
        "        y = 2\n"
        "        return x + y\n",
        encoding="utf-8",
    )

    # Switch Statements: if/elif chain with 5 branches (> threshold 3).
    (root / "switch.py").write_text(
        "def classify(kind):\n"
        "    if kind == 1:\n        return 'one'\n"
        "    elif kind == 2:\n        return 'two'\n"
        "    elif kind == 3:\n        return 'three'\n"
        "    elif kind == 4:\n        return 'four'\n"
        "    else:\n        return 'many'\n",
        encoding="utf-8",
    )

    # Deep Inheritance Tree: A <- B <- C <- D <- E (DIT of E = 5 > threshold 3).
    (root / "dit.py").write_text(
        "class A:\n    def work(self):\n        return 1\n\n\n"
        "class B(A):\n    pass\n\n\n"
        "class C(B):\n    pass\n\n\n"
        "class D(C):\n    pass\n\n\n"
        "class E(D):\n    def more(self):\n        return 2\n",
        encoding="utf-8",
    )

    # Data Clumps: two functions sharing the same 6 parameters.
    (root / "clumps.py").write_text(
        "def fa(alpha, beta, gamma, delta, epsilon, zeta):\n"
        "    return alpha + beta + gamma + delta + epsilon + zeta\n\n\n"
        "def fb(alpha, beta, gamma, delta, epsilon, zeta):\n"
        "    return alpha * beta * gamma * delta * epsilon * zeta\n",
        encoding="utf-8",
    )

    # Duplicate Code: 15 structurally identical 5-statement functions.
    dup_fn = (
        "def f{i}(p):\n"
        "    a = p + 1\n"
        "    b = a + 2\n"
        "    c = b + 3\n"
        "    d = c + 4\n"
        "    return a + b + c + d\n"
    )
    (root / "dup.py").write_text("\n\n".join(dup_fn.format(i=i) for i in range(15)), encoding="utf-8")

    # God Object: 21 module-level functions (> threshold 20).
    (root / "god.py").write_text(
        "\n\n".join(f"def g{i}(x):\n    return x + {i}" for i in range(21)),
        encoding="utf-8",
    )

    # Unstable Dependency: a module that imports 6 others (instability 1.0).
    (root / "consumer.py").write_text(
        "import os\nimport sys\nimport json\nimport re\nimport math\nimport collections\n\n\n"
        "def use():\n    return os, sys, json, re, math, collections\n",
        encoding="utf-8",
    )

    # NOM + LCOM: class with 13 unrelated methods each touching its own field.
    big_methods = "\n".join(
        f"    def m{i}(self):\n        self.field{i} = {i}\n        return self.field{i}"
        for i in range(13)
    )
    (root / "bigclass.py").write_text("class Big:\n" + big_methods + "\n", encoding="utf-8")

    # High Cyclomatic Complexity: a method with 14 branches.
    branches = "".join(f"        if n == {i}:\n            total += {i}\n" for i in range(14))
    (root / "complex.py").write_text(
        "class Comp:\n    def tangled(self, n):\n        total = 0\n" + branches + "        return total\n",
        encoding="utf-8",
    )

    return str(root)


def _write_ml_project(root) -> str:
    root = root / "mlproj"
    root.mkdir()
    (root / "ml_sample.py").write_text(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import torch\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "from transformers import AutoModel\n\n\n"
        "def run(df):\n"
        "    arr = np.random.rand(10)\n"
        "    scaler = StandardScaler()\n"
        "    X = scaler.fit_transform(df.drop('y', axis=1))\n"
        "    y = df['y']\n"
        "    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)\n"
        "    total = 0\n"
        "    for idx, row in df.iterrows():\n"
        "        total += row.sum()\n"
        "    model = AutoModel.from_pretrained('bert-base-uncased')\n"
        "    net = torch.nn.Linear(10, 2)\n"
        "    return total, net, model\n",
        encoding="utf-8",
    )
    return str(root)


# ---------------------------------------------------------------------------
# Fixtures (built once per module; analysed once per analysis type)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def code_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("smells")
    proj = _write_code_project(root)
    return {
        "code": detect_python_smells(proj, analysis_type="code"),
        "architectural": detect_python_smells(proj, analysis_type="architectural"),
        "structural": detect_python_smells(proj, analysis_type="structural"),
    }


@pytest.fixture(scope="module")
def ml_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlsmells")
    proj = _write_ml_project(root)
    return detect_ml_smells(proj)


# ---------------------------------------------------------------------------
# Previously-dead smells (the heart of this pass)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smell_name",
    ["Dead Code", "Data Class", "Lazy Class", "Switch Statements"],
)
def test_previously_dead_code_smells_now_fire(code_results, smell_name):
    assert smell_name in _names(code_results["code"], "code_smells"), (
        f"{smell_name} should now be detected; got {_names(code_results['code'], 'code_smells')}"
    )


def test_deep_inheritance_tree_now_fires(code_results):
    names = _names(code_results["structural"], "structural_smells")
    assert "Deep Inheritance Tree (DIT)" in names, f"DIT should now fire; got {names}"


# ---------------------------------------------------------------------------
# Cross-file code smells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smell_name", ["Data Clumps", "Duplicate Code"])
def test_cross_file_code_smells(code_results, smell_name):
    assert smell_name in _names(code_results["code"], "code_smells")


# ---------------------------------------------------------------------------
# Architectural smells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smell_name", ["God Object", "Unstable Dependency"])
def test_architectural_smells(code_results, smell_name):
    assert smell_name in _names(code_results["architectural"], "architectural_smells")


# ---------------------------------------------------------------------------
# Structural smells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smell_name",
    [
        "High Number of Methods (NOM)",
        "High Lack of Cohesion of Methods (LCOM)",
        "High Cyclomatic Complexity",
    ],
)
def test_structural_smells(code_results, smell_name):
    assert smell_name in _names(code_results["structural"], "structural_smells")


# ---------------------------------------------------------------------------
# ML smells (representative sample across frameworks)
# ---------------------------------------------------------------------------

def test_ml_groups_non_empty(ml_results):
    assert "error" not in ml_results, ml_results.get("error")
    assert ml_results["framework_smells"], "expected framework-specific ML smells"
    assert ml_results["general_ml_smells"], "expected general ML smells"
    assert ml_results["huggingface_smells"], "expected HuggingFace ML smells"


@pytest.mark.parametrize(
    "needle",
    [
        "Iteration",     # pandas iterrows -> Unnecessary Iteration
        "leakage",       # sklearn preprocessing-before-split
        "Randomness",    # numpy/pytorch missing seed control
        "docstring",     # general ML smell
    ],
)
def test_ml_expected_smell_substrings(ml_results, needle):
    blob = _ml_text(ml_results)
    assert needle.lower() in blob.lower(), f"expected '{needle}' in ML output: {blob[:400]}"


def test_ml_huggingface_smell_present(ml_results):
    blob = _ml_text(ml_results).lower()
    assert any(tok in blob for tok in ("versioning", "caching", "from_pretrained", "model")), blob[:400]
