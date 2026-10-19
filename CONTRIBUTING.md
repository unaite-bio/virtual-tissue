# Contributing to virtual-tissue

## Prerequisites (one-time)
- Conda (Miniforge recommended) with `conda-lock` in base:
  `conda install -n base -c conda-forge conda-lock`
- GitHub authentication: `brew install gh && gh auth login`

## Setup
```bash
git clone https://github.com/unaite-bio/virtual-tissue.git
cd virtual-tissue
conda-lock install -n virtual-tissue conda-lock.yml
conda activate virtual-tissue
pip install -e .
pre-commit install
```

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
- **Someone changed `conda-lock.yml`?** After pulling, resync your env:
  `conda-lock install -n virtual-tissue conda-lock.yml`
- **Running tests:** `pytest` runs the fast suite. GPU or slow tests are
  skipped by default; run them with `pytest -m gpu` or `pytest -m slow`.

## Rules
- Never push to someone else's branch without asking.
- Keep PRs small (aim for < 400 lines changed).
- **No data, model weights, or credentials in the repo — ever.**
- Adding a dependency: add it to `environment.yml`, then
```bash
  conda env update -f environment.yml
  conda-lock -f environment.yml -p osx-arm64 -p linux-64 -p linux-aarch64
```
  and commit both files in the same PR.
- Mark GPU or long tests with `@pytest.mark.gpu` / `@pytest.mark.slow`
  and run them locally before requesting review.
- Questions about the workflow? Open an issue with the `documentation` label.
