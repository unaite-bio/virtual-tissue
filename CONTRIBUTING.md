# Contributing to virtual-tissue

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
3. Create a branch: `feat/…`, `fix/…`, `exp/…`, or `docs/…` + short description.
4. Commit, push, open a PR to `main` that links the issue (`Closes #12`).
5. Get one approval and green CI, then **squash-merge**.

## Rules
- Never push to someone else's branch without asking.
- Keep PRs small (aim for < 400 lines changed).
- **No data, model weights, or credentials in the repo — ever.**
- Adding a dependency: edit `environment.yml`, re-run
  `conda-lock -f environment.yml -p osx-arm64 -p linux-64 -p linux-aarch64`,
  and commit both files in the same PR.
- Mark GPU or long tests with `@pytest.mark.gpu` / `@pytest.mark.slow`
  and run them locally before requesting review.
