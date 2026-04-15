# Agent Review Examples

Copy-paste command and prompt examples for Claude Code and similar AI agents.

## Setup

```bash
uv sync
code-review show-config
```

## Command Recipes

### 1) Full repository review

```bash
code-review review ./my_project --output reports/full-review.md
```

### 2) Full review with Anthropic

```bash
export ANTHROPIC_API_KEY=sk-...
code-review review ./my_project --provider anthropic --output reports/anthropic-review.md
```

### 3) GitHub URL review

```bash
code-review review https://github.com/owner/repo --output reports/repo-review.md
```

### 4) Single-file deep dive

```bash
code-review analyze-file ./src/app/service.py --output reports/service-file-review.md
```

### 5) Structural hotspot triage

```bash
code-review run-tool code-intel ./src --top-n 20
code-review run-tool python-smells ./src --type structural
```

### 6) ML smell triage

```bash
code-review run-tool ml-smells ./src
```

### 7) Technical debt classification

```bash
code-review run-tool classify-td --text "TODO: remove hard-coded timeout" --text "FIXME: retry strategy is brittle"
```

## Prompt Templates for Claude Code

### Template A: Critical-first audit

Use this prompt:

```text
Run `code-review review ./my_project --output reports/review.md`.
Then summarize only critical/high-severity findings first, with file:line:col references.
Finish with a now/next/later remediation plan.
```

### Template B: Scope-limited triage

Use this prompt:

```text
Run `code-review run-tool code-intel ./src --top-n 15` and
`code-review run-tool python-smells ./src --type structural`.
Return top complexity hotspots and architecture risks only.
Do not include low-priority issues.
```

### Template C: File incident review

Use this prompt:

```text
Run `code-review analyze-file ./src/path/to/file.py`.
Focus on correctness and maintainability defects.
For each defect, include impact, evidence, and minimal fix suggestion.
```

## Recommended Output Shape

Ask the agent to produce:
1. Executive summary
2. Critical findings
3. Supporting findings
4. Recommended fixes
5. Rollout order (now/next/later)
