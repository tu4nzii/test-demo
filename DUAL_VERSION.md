# Dual Version Workflow

This folder is the test version.

- Test project: `F:\program\test-demo`
- Release project: `F:\program\test-demo-release`
- Release branch/worktree: `release-only-text-box`
- Release baseline: `3750064` (`只绘制加密的文本和文本框`)

Ports:

- Release backend: `http://127.0.0.1:8000`
- Release frontend: `http://127.0.0.1:5173`
- Test backend: `http://127.0.0.1:8001`
- Test frontend: `http://127.0.0.1:5174`

Use `quick_start_test.bat` for the test version.
Use `quick_start_both.bat` to start release and test together.

Routine AI edits should stay in this test folder. Only apply changes to the release folder when explicitly promoting or modifying the release version.
