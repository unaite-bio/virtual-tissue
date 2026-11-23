# Contributing to virtual-tissue

## Prerequisites (one-time)
- pixi, a single binary, no admin rights needed (https://pixi.sh): `curl -fsSL https://pixi.sh/install.sh | sh`
  on macOS and Linux, `winget install prefix-dev.pixi` on Windows.
- GitHub authentication: `brew install gh && gh auth login`

## Setup
```bash
git clone https://github.com/unaite-bio/virtual-tissue.git
cd virtual-tissue
pixi install                    # the default environment, from pixi.lock
pixi run pre-commit install
```
Run anything in the environment with `pixi run <command>`, or open a shell in it with `pixi shell`.
The GPU encoders have their own environment (`pixi run -e torch ...`), on Linux with CUDA only.

## Workflow
1. Pick or open an issue.
2. `git switch main && git pull`
3. Create a branch: `feat/…`, `fix/…`, `exp/…`, or `docs/…` + short description,
   e.g. `git switch -c feat/attention-pooling`.
4. Commit, push (`git push -u origin <branch>`), and open a PR to `main` that
   links the issue (`Closes #12`). Use a **draft PR** for work in progress.
5. Get one approval and green CI, then **squash-merge**. The remote branch is
   deleted automatically; delete your local copy with `git branch -d <branch>`.

Direct pushes to `main` are blocked — all changes go through a PR.

## Day-to-day tips
- **Pre-commit modified files?** It auto-fixed formatting. Run `git add .`
  and commit again.
- **`main` moved while your PR is open?** Click **Update branch** on the PR,
  or locally: `git pull origin main` then `git push`.
- **Someone changed `pixi.lock`?** After pulling, run `pixi install`.
- **Running tests:** `pixi run test` runs the fast suite and `pixi run lint` the
  ruff checks, as CI does. GPU or slow tests are skipped by default; run them
  with `pixi run pytest -m gpu` or `pixi run pytest -m slow`.

## Rules
- Never push to someone else's branch without asking.
- Keep PRs small (aim for < 400 lines changed).
- **No data, model weights, or credentials in the repo — ever.**
- Adding a dependency: `pixi add <package>` (conda-forge) or `pixi add --pypi <package>`,
  with `--feature <group>` for one environment group only. It updates `pyproject.toml`
  and `pixi.lock`; commit both in the same PR.
- Mark GPU or long tests with `@pytest.mark.gpu` / `@pytest.mark.slow`
  and run them locally before requesting review.
- Questions about the workflow? Open an issue with the `documentation` label.
