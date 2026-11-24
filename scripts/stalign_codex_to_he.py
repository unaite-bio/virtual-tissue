"""Refine a CODEX → H&E placement with STalign (LDDMM: affine + diffeomorphism) and warp cells.

Runs in its own conda env (STalign needs torch, kept out of the team env):

    conda create -n stalign -c conda-forge python=3.11 pip
    conda run -n stalign pip install \\
        "git+https://github.com/JEFworks-Lab/STalign.git@8a84bf1facc5173f7155400867a09c302125904e" \\
        scikit-image tifffile scipy
    conda run -n stalign python scripts/stalign_codex_to_he.py --work outputs/valis_mar21 \\
        --um-per-px 3.278 --source dapi --target rgb

Expects in --work (written by the Causer EDA notebook, VALIS section):
    he.png                RGB H&E crop of piece 2 (target)
    dapi_init204.tif      CODEX DAPI already placed in the he.png frame by the 204° outline fit
    cells_init204_px.csv  columns x, y: CODEX cell centroids in he.png pixels, same placement

The fit starts from that placement (STalign's rotation pre-fit is switched off).

Writes to --work:
    warped_cells_stalign_<source>_<target>_a<a>.csv   x, y of the cells in he.png pixels
    stalign_<source>_<target>_a<a>.json               fitted affine + settings
"""

import os

# Headless; also overrides a notebook kernel's inline backend inherited via MPLBACKEND
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import STalign
import tifffile
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2hed
from skimage.measure import block_reduce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--um-per-px", required=True, type=float, help="µm per he.png pixel")
    ap.add_argument("--source", choices=["dapi", "density"], default="dapi")
    ap.add_argument("--target", choices=["rgb", "hem"], default="rgb")
    ap.add_argument("--a", type=float, default=500.0, help="deformation smoothness scale, µm")
    ap.add_argument("--niter", type=int, default=2000)
    ap.add_argument("--down", type=int, default=2, help="downsample factor for the images")
    ap.add_argument(
        "--sigmaM",
        type=float,
        default=None,
        help="matching noise; default = std of the target (STalign's default 1.0 barely moves "
        "0-1 normalised images)",
    )
    ap.add_argument("--diffeo-start", type=int, default=200, help="affine-only iterations first")
    # STalign's automatic step sizes came out ~1e-12 (linear) / 2e-5 µm (translation) here, so
    # nothing moved; give explicit steps (affine steps are then damped per-parameter)
    ap.add_argument("--epL", type=float, default=None)
    ap.add_argument("--epT", type=float, default=None)
    ap.add_argument("--epV", type=float, default=None)
    ap.add_argument("--tag", default="", help="extra suffix for output names")
    ap.add_argument("--moving", default="dapi_init204.tif", help="pre-placed DAPI in --work")
    ap.add_argument("--cells", default="cells_init204_px.csv", help="pre-placed cells in --work")
    args = ap.parse_args()

    work = args.work.resolve()
    he = np.asarray(Image.open(work / "he.png"))[..., :3].astype(float) / 255
    cells_px = pd.read_csv(work / args.cells)[["x", "y"]].to_numpy(float)
    dx = args.um_per_px * args.down  # µm per downsampled pixel

    # Coordinates in µm, (rows, cols) as STalign expects, centred on the crop centre
    centre = np.array(he.shape[:2]) * args.um_per_px / 2

    def grid(shape):  # pixel-centre coordinates of a downsampled image
        return [
            np.arange(n) * dx + (args.down - 1) / 2 * args.um_per_px - centre[i]
            for i, n in enumerate(shape)
        ]

    # Target J: H&E (channels first)
    if args.target == "rgb":
        J = block_reduce(he, (args.down, args.down, 1), np.mean).transpose(2, 0, 1)
    else:
        hem = ndi.gaussian_filter(rgb2hed(he)[..., 0], 1)
        J = block_reduce(hem, (args.down, args.down), np.mean)[None]
    J = (J - J.min()) / (np.ptp(J) + 1e-9)
    xJ = grid(J.shape[1:])

    # Source I: DAPI image, or cell density rasterised from the centroids
    if args.source == "dapi":
        dapi = tifffile.imread(work / args.moving).astype(float)
        I = block_reduce(dapi, (args.down, args.down), np.mean)[None]
        xI = grid(I.shape[1:])
    else:
        xI_, yI_ = cells_px[:, 0] * args.um_per_px, cells_px[:, 1] * args.um_per_px
        X_, Y_, Ir = STalign.rasterize(xI_, yI_, dx=dx, blur=1.5, draw=0)
        I = Ir  # (1, rows, cols)
        xI = [Y_ - centre[0], X_ - centre[1]]
    I = (I - I.min()) / (np.ptp(I) + 1e-9)

    sigmaM = args.sigmaM if args.sigmaM is not None else float(J.std())
    steps = {k: getattr(args, k) for k in ["epL", "epT", "epV"] if getattr(args, k) is not None}
    if steps:
        steps["affine_step_control"] = True
    torch.manual_seed(0)
    out = STalign.LDDMM(
        [torch.tensor(x) for x in xI],
        torch.tensor(I),
        [torch.tensor(x) for x in xJ],
        torch.tensor(J),
        a=args.a,
        niter=args.niter,
        prefit=False,  # keep the 204° start; identity initial affine
        sigmaM=sigmaM,
        sigmaB=2 * sigmaM,
        sigmaA=5 * sigmaM,
        diffeo_start=args.diffeo_start,
        draw=False,
        **steps,
    )
    A, v, xv = out["A"], out["v"], out["xv"]

    # Cells: (x, y) px → centred (row, col) µm, warp source → target, back to (x, y) px
    pts_rc_um = np.c_[cells_px[:, 1], cells_px[:, 0]] * args.um_per_px - centre
    warped = STalign.transform_points_source_to_target(xv, v, A, pts_rc_um).numpy()
    warped_xy_px = (warped + centre)[:, ::-1] / args.um_per_px

    tag = f"stalign_{args.source}_{args.target}_a{int(args.a)}{args.tag}"
    pd.DataFrame(warped_xy_px, columns=["x", "y"]).to_csv(
        work / f"warped_cells_{tag}.csv", index=False
    )
    Lm = A[:2, :2].numpy()
    info = {
        "settings": vars(args) | {"work": str(work), "sigmaM_used": sigmaM},
        "stalign_settings": {k: float(x) for k, x in out["settings"].items()},
        "max_velocity_um": float(v.abs().max()),
        "affine_A": A.numpy().tolist(),
        "affine_det": float(np.linalg.det(Lm)),
        "affine_rotation_deg": float(np.degrees(np.arctan2(Lm[1, 0], Lm[0, 0]))),
        "median_cell_move_um": float(np.median(np.linalg.norm(warped - pts_rc_um, axis=1))),
    }
    (work / f"{tag}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "settings"}, indent=2))


if __name__ == "__main__":
    main()
