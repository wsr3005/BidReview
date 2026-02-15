# Task Card

## Metadata

- Task ID: TASK-2026-02-16-timestamped-annotated-outputs
- Owner: Human + Agent
- Date: 2026-02-16
- Priority: Medium
- Risk level: Low

## Objective

为每次生成的标注副本（Word/PDF）在文件名中附带时间戳，避免多次运行覆盖输出，便于回溯具体哪一次审查产物。

## Definition of Done

1. `annotated/` 下生成的副本名称包含时间戳：`<stem>.annotated.YYYYMMDD-HHMMSS(.N).docx|pdf`
2. 写入 `annotated/annotated-copy.json` 指向最新副本路径，`--resume` 时可返回该路径。
3. `.\scripts\verify.ps1` 通过。

## Scope

- In scope:
  - `bidagent/pipeline.py`

