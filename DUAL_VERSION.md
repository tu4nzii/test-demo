# Dual Version Workflow

This repository is the editable test project.

| Item | Value |
| --- | --- |
| Test project | `F:\program\test-demo` |
| Release project | `F:\program\test-demo-release` |
| Release branch/worktree | `release-only-text-box` |

## Ports

| Service | URL |
| --- | --- |
| Release backend | `http://127.0.0.1:8000` |
| Release frontend | `http://127.0.0.1:5173` |
| Test backend | `http://127.0.0.1:8001` |
| Test frontend | `http://127.0.0.1:5174` |

## Scripts

- `quick_start_test.bat`: start the test version.
- `quick_start_both.bat`: start release and test together.

Routine AI edits should stay in this test folder. Only apply changes to the release folder when explicitly promoting or modifying the release version.
