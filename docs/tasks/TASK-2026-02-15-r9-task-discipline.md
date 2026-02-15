# Task Card

## Metadata

- Task ID: TASK-2026-02-15-r9-task-discipline
- Owner: Human + Agent
- Date: 2026-02-15
- Priority: Medium
- Risk level: Low

## Objective

减少“任务卡已创建但未提交”的纪律问题，提升可追踪性和回溯效率。

## Definition of Done

1. 提供一个可选门禁脚本，可检测 `docs/tasks/` 下未追踪的 `TASK-*.md` 文件。
2. 默认仅提示，不影响现有开发流程；需要时可开启失败模式作为门禁。
3. 脚本不误伤本地 code review 便签（默认忽略 `code-review`）。

## Scope

- In scope:
  - `scripts/check-task-cards.ps1`
- Out of scope:
  - 将纪律门禁强制集成到 `scripts/verify.ps1` 的默认 Gates（可后续再讨论）

## Usage

```powershell
.\scripts\check-task-cards.ps1
.\scripts\check-task-cards.ps1 -FailOnUntracked
```

## Verification

- Manual:
  - 在存在未追踪 TASK 卡时应打印列表
  - `-FailOnUntracked` 时应返回非 0

