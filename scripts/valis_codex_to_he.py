"""Register a CODEX DAPI image onto an H&E image with VALIS and warp CODEX cell centroids.

Runs in its own conda env (VALIS needs libvips/pyvips/torch, kept out of the team env):

    conda create -n valis -c conda-forge python=3.11 libvips pyvips openjdk=17
    conda run -n valis pip install valis-wsi
    conda run -n valis python scripts/valis_codex_to_he.py --work outputs/valis_mar21

Expects in --work (written by the Causer EDA notebook):
    he.png         RGB H&E crop (fixed / reference image)
    dapi.tif       single-channel CODEX DAPI (moving image; --moving to override)
    cells_px.csv   columns x, y: CODEX cell centroids in moving-image pixels (--cells)

Writes to --work:
    warped_cells.csv  x_rigid, y_rigid, x_nonrigid, y_nonrigid in he.png pixel coordinates
    summary.json      VALIS summary + affine fit (rotation, scale, reflection) of the rigid map
    valis/            VALIS output (overlap thumbnails in valis/<name>/overlaps/)
"""

import os

# Headless; also overrides a notebook kernel's inline backend inherited via MPLBACKEND
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from valis import preprocessing, registration


def describe_rigid(src, dst):
    """Summarise a point mapping as rotation/scale/reflection via a least-squares affine fit.

    Reflection is read as an x-mirror applied before rotation (the notebook's rot_flip
    convention). Plain lstsq: skimage's estimate() on ~50k points was SIGKILLed in testing.
    """
    X = np.c_[src, np.ones(len(src))]
    coef = np.linalg.lstsq(X, dst, rcond=None)[0]
    A = coef[:2].T
    det = np.linalg.det(A)
    reflected = bool(det < 0)
    B = A @ np.diag([-1, 1]) if reflected else A
    resid = np.linalg.norm(X @ coef - dst, axis=1)
    return {
        "reflected": reflected,
        "rotation_deg": float(np.degrees(np.arctan2(B[1, 0], B[0, 0])) % 360),
        "scale": float(np.sqrt(abs(det))),
        "translation_px": coef[2].tolist(),
        "affine_fit_median_resid_px": float(np.median(resid)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--max-dim", type=int, default=1024, help="processed image size for VALIS")
    ap.add_argument("--no-reflections", action="store_true", help="skip the mirrored check")
    ap.add_argument("--moving", default="dapi.tif", help="moving image filename in --work")
    ap.add_argument("--cells", default="cells_px.csv", help="cell centroids (moving image px)")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _init204")
    ap.add_argument(
        "--no-rigid",
        action="store_true",
        help="skip VALIS rigid step (moving image already pre-aligned); non-rigid only",
    )
    args = ap.parse_args()

    work = args.work.resolve()
    he_f, dapi_f = str(work / "he.png"), str(work / args.moving)
    cells = pd.read_csv(work / args.cells)[["x", "y"]].to_numpy(float)

    registrar = registration.Valis(
        str(work),
        str(work / f"valis{args.tag}"),
        name=f"codex_to_he{args.tag}",
        img_list=[he_f, dapi_f],
        reference_img_f=he_f,
        align_to_reference=True,
        check_for_reflections=not args.no_reflections,
        do_rigid=not args.no_rigid,
        max_processed_image_dim_px=args.max_dim,
        max_non_rigid_registration_dim_px=args.max_dim,
    )
    # Nuclei vs nuclei: haematoxylin from the H&E, DAPI from CODEX
    processor_dict = {
        he_f: [preprocessing.HEDeconvolution, {"stain": "hem"}],
        dapi_f: [preprocessing.ChannelGetter, {"channel": "dapi", "adaptive_eq": True}],
    }
    _, _, summary = registrar.register(processor_dict=processor_dict)

    he_slide, dapi_slide = registrar.get_slide(he_f), registrar.get_slide(dapi_f)
    rigid = dapi_slide.warp_xy_from_to(cells, he_slide, non_rigid=False)
    nonrigid = dapi_slide.warp_xy_from_to(cells, he_slide, non_rigid=True)

    pd.DataFrame(
        {
            "x_rigid": rigid[:, 0],
            "y_rigid": rigid[:, 1],
            "x_nonrigid": nonrigid[:, 0],
            "y_nonrigid": nonrigid[:, 1],
        }
    ).to_csv(work / f"warped_cells{args.tag}.csv", index=False)

    out = {
        "rigid_fit": describe_rigid(cells, rigid),
        # register() returns no summary when do_rigid=False
        "valis_summary": None
        if summary is None
        else json.loads(summary.to_json(orient="records", default_handler=str)),
    }
    (work / f"summary{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["rigid_fit"], indent=2))
    # No kill_jvm(): no BioFormats JVM is started for png/tif inputs, and calling it
    # SIGKILLed the process in testing.


if __name__ == "__main__":
    main()
