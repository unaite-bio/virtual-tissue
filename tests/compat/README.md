# h5ad compatibility check

Bundles are written in the core env (newest anndata, Python 3.12) and read by method envs as old as SpatialGlue's (anndata 0.8.0, Python 3.8); outputs travel the other way.
`h5ad_compat.py write <dir>` writes the pinned input and output layouts with `usbb.io.write_h5ad`; `check <dir>` reads them back and fails with every difference.
CI (job `h5ad-compat`) writes with each stack and checks with the other. Locally, from the repo root (swap the two interpreters for the reverse):

```
D=$(mktemp -d)
PYTHONPATH=src uv run --no-project --python 3.12 --with "anndata,h5py,numpy,pandas,scipy,pyyaml" python tests/compat/h5ad_compat.py write $D
PYTHONPATH=src uv run --no-project --python 3.8 --with "anndata==0.8.0,numpy==1.22.3,pandas==1.4.2,scipy==1.8.1,h5py==3.6.0,pyyaml" python tests/compat/h5ad_compat.py check $D
```
