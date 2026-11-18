# Multi-Agent Workflows Example (`@beta.workflows.define` & `agent.beta.run_workflow`)

This example demonstrates how to author and execute deterministic multi-agent
workflows using `from google.antigravity import beta` (`beta.workflows`) and
`await agent.beta.run_workflow(...)`.

## 1. Authoring Typed Workflows with `@beta.workflows.define`

Decorate a zero-argument Python function (`async def` or `def`) with
`@beta.workflows.define` (or `@workflows.define` via
`workflows = beta.workflows`) and use the workflow primitives
(`workflows.phase`, `workflows.log`, `workflows.call_tool`, `workflows.agent`,
`workflows.parallel`, `workflows.pipeline`, `workflows.args`, and
`workflows.ToolError`). The SDK validates the function's AST at definition time
and rewrites `workflows.<primitive>` references for the sandboxed `run_workflow`
Python engine:

```python
from google.antigravity import Agent, LocalAgentConfig, beta, types

workflows = beta.workflows


@workflows.define
async def security_audit_workflow() -> None:
  """Reads source files via call_tool, audits in parallel, and synthesizes in 2 turns."""
  workflows.phase("Read & Audit Files")

  async def read_and_audit(path: str):
    content = await workflows.call_tool("read_file", Path=path)
    return await workflows.agent(
        f"Audit `{path}` in one concise sentence naming the vulnerability:\n"
        f"{content}",
        schema={"type": "string"},
        role=f"Audit {path.rsplit('/', 1)[-1]}",
        type_name="file_auditor",
    )

  findings = await workflows.parallel(workflows.args["paths"], read_and_audit)

  workflows.phase("Multi-Turn Executive Synthesis")
  combined = "\n".join(str(f).strip() for f in findings)
  lead = workflows.agent(
      "You are the lead security synthesizer.",
      role="Lead Synthesizer",
      type_name="lead_synthesizer",
  )
  summary = await lead(
      f"Summarize these security findings:\n{combined}",
      schema={
          "type": "object",
          "properties": {
              "total_issues": {"type": "integer"},
              "highest_severity": {"type": "string"},
              "summary": {"type": "string"},
          },
          "required": ["total_issues", "highest_severity", "summary"],
      },
  )
  remediation = await lead(
      "Now propose a concise 1-sentence remediation plan for the"
      " highest-severity issue from your summary.",
      schema={"type": "string"},
  )
  workflows.log(
      f"Audit complete: {summary['total_issues']} issues"
      f" (highest={summary['highest_severity']}): {summary['summary']}"
  )
  workflows.log(f"Remediation plan: {remediation}")


file_auditor = types.SubagentConfig(
    name="file_auditor",
    description="Audits a code snippet for security vulnerabilities.",
    system_instructions="Identify the vulnerability in one sentence.",
    capabilities=types.SubagentCapabilities(enabled_tools=[]),
)
lead_synthesizer = types.SubagentConfig(
    name="lead_synthesizer",
    description="Synthesizes security findings across multiple turns.",
    system_instructions="Combine findings and respond with the schema on each turn.",
    capabilities=types.SubagentCapabilities(enabled_tools=[]),
)

config = LocalAgentConfig(subagents=[file_auditor, lead_synthesizer])

async with Agent(config) as my_agent:
  result = await my_agent.beta.run_workflow(
      security_audit_workflow,
      args={"paths": ["/path/to/auth.py", "/path/to/db.py"]},
  )
  print("Output:\n", result.output)
  print("Response:", result.response_text)
```

## 2. Running a Standalone Workflow Script File

You can also pass a `.py` file path (`str` or `pathlib.Path`) via
`await agent.beta.run_workflow(script_path=..., description=...)`:

```python
import pathlib

script_path = pathlib.Path("release_checklist_workflow.py")
script_path.write_text("""\
phase("Sequential Verification")
checks = ["unit_tests", "lint_check"]
results = await pipeline(
    checks,
    lambda item: agent(
        f"Confirm check '{item}' passed.",
        schema={"type": "string"},
        role=f"Verify {item}",
    ),
)
for item_name, res in zip(checks, results):
    log(f"{item_name}: {str(res).strip()}")
""")

async with Agent(LocalAgentConfig()) as my_agent:
  result = await my_agent.beta.run_workflow(
      script_path=script_path,
      description="Run the sequential release verification checklist",
  )
  print(result.output)
```

## Workflow Execution Rules

Workflow scripts execute in a managed Python subprocess with the following
rules:

- **Top-level `await`, `async def` helpers, and loops**: `await` works at the
  top level of a standalone script (or inside an `@workflows.define` function),
  nested `async def` stage helpers can be passed to `parallel` or `pipeline`,
  and `for` / `while` loops are supported for retry or multi-turn review loops.
- **No top-level `return` or `yield` in standalone scripts**: Standalone `.py`
  workflow scripts cannot use `return`, `yield`, or `yield from` at module top
  level (`return` inside helper functions is allowed).
- **Workflow primitives and logging**: The workflow primitives (`phase`, `log`,
  `call_tool`, `agent`, `parallel`, `pipeline`, `args`, `ToolError`) are
  injected into the execution namespace (`@workflows.define` automatically
  rewrites `workflows.<primitive>` attribute access to the injected runtime
  globals). Use `log(...)` to emit structured progress output into
  `WorkflowRunResult.output`.
