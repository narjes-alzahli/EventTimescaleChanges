"""
Visualization helpers
======================

Create NIfTI maps of SL-level changes and simple diagnostic line plots for
single SLs (viewing curves across event counts).

"""

import os
from __future__ import annotations
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from typing import Mapping, Iterable
from io import load_valid_mask, load_sl_voxels, project_sl_to_vox2d, vox2d_to_volume3d

# ------------------------------ saving utils --------------------------------

def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def save_nifti(img: nib.Nifti1Image, path: str) -> None:
    """
    Save a NIfTI image to disk.

    Parameters
    ----------
    img : nib.Nifti1Image
        Image object to write.
    path : str
        Output file path (ends with .nii or .nii.gz).
    """
    nib.save(img, path)


def save_fig(fig: plt.Figure, path: str, dpi: int = 150, bbox_inches: str = "tight") -> None:
    """
    Save a matplotlib figure to disk and close it to free memory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str
        Path to .png/.pdf/.svg, etc.
    dpi : int
        Output DPI.
    bbox_inches : str
        Matplotlib bbox_inches argument.
    """
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def npvol_to_nifti(vol: np.ndarray, template_path: str) -> nib.Nifti1Image:
    """
    Convert a 3D numpy volume to NIfTI using a template for affine/header.

    Parameters
    ----------
    vol : ndarray, shape (Z, Y, X)
        Volume with NaNs outside the valid mask.
    template_path : str
        Path to a template NIfTI whose affine/header should be reused.

    Returns
    -------
    nib.Nifti1Image
        NIfTI image with template affine/header.
    """
    template = nib.load(template_path)
    return nib.Nifti1Image(vol.T, affine=template.affine, header=template.header)

# ------------------------------ plotting ------------------------------------

def plot_wbs_change_map(sl_changes: Mapping[str, float], template_path: str = "./MNI152_T1_brain_template.nii") -> nib.Nifti1Image:
    """
    Project per-SL changes into voxel space and return a NIfTI volume.

    Parameters
    ----------
    sl_changes : mapping
        SL id -> change value.
    template_path : str
        Path to a template NIfTI whose affine/header should be reused.

    Returns
    -------
    Nifti1Image
        Change map in template space.
    """
    template = nib.load(template_path)
    mask = load_valid_mask()
    sl_vox = load_sl_voxels()
    vec = project_sl_to_vox2d({int(k): float(v) for k, v in sl_changes.items()}, mask, sl_vox)
    vol = vox2d_to_volume3d(vec, mask)
    return nib.Nifti1Image(vol.T, affine=template.affine, header=template.header)


def plot_wbs_diagnostic(sl_id: str, clip: str, wbs_dict: Mapping[str, np.ndarray], n_cv: int = 5) -> plt.Figure:
    """
    Diagnostic plot of WBS vs. event count for a single SL and clip.

    Parameters
    ----------
    sl_id : str
        Searchlight id.
    clip : str
        Clip condition (for title only).
    wbs_dict : mapping
        sl_id -> WBS grid arrays.
    n_cv : int
        # CV folds to average for view-1/avg-(2..6) display.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with viewing-wise curves and average of views 2–6.
    """
    arr = wbs_dict[sl_id]
    evs = np.arange(2, 11)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    avg_rest = []
    for v in range(6):
        y = [arr[0, :n_cv, 0, v, ev - 2].mean() for ev in evs]
        ax.plot(evs, y, linewidth=(3 if v == 0 else 1), label=f"view {v+1}")
        if v > 0:
            avg_rest.append(y)
    if avg_rest:
        ax.plot(evs, np.mean(avg_rest, axis=0), linewidth=3, label="avg views 2–6")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_xlabel("Event Count"); ax.set_ylabel("WBS"); ax.set_title(f"SL {sl_id} — {clip}")
    ax.legend(); fig.tight_layout()
    return fig

# --------------------------- batch wrapper ----------------------------

def wbc_plots(
    sls: Iterable[str],
    clip: str,
    wbs_dict: Mapping[str, np.ndarray],
    out_dir: str,
    n_cv: int = 5,
    dpi: int = 150,
    prefix: str = "",
) -> None:
    """
    Save per-SL diagnostic plots in batch.

    Parameters
    ----------
    sls : iterable of str
        The SL ids to plot.
    clip : str
        Condition name for titles and filenames.
    wbs_dict : mapping
        SL -> WBS grid arrays.
    out_dir : str
        Destination directory for images.
    n_cv : int
        # of CV folds to average.
    dpi : int
        Image DPI.
    prefix : str
        Optional filename prefix.
    """
    ensure_dir(out_dir)
    for sl_id in sls:
        fig = plot_wbs_diagnostic(sl_id, clip, wbs_dict, n_cv=n_cv)
        fname = f"{prefix}sl{sl_id}.png"
        save_fig(fig, os.path.join(out_dir, fname), dpi=dpi)