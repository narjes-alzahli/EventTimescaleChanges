"""
I/O utilities
==============

Load per-searchlight HDF5 arrays, valid-voxel masks, and SL→voxel membership.
Provide helpers to project SL-level values into voxel space and back.

All paths match the original environment; change as needed.

"""

from __future__ import annotations
import os
import pickle
from typing import Mapping
import deepdish as dd
import nibabel as nib
import numpy as np

__all__ = [
    "load_sl_h5",
    "load_valid_mask",
    "load_sl_voxels",
    "project_sl_to_vox2d",
    "vox2d_to_volume3d",
]


def load_sl_h5(sl_id: str, clip: str, base_dir: str = "/data/gbh/data/SL/") -> list[np.ndarray]:
    """
    Load a searchlight's HDF5 data for a clip and optionally split halves.

    Parameters
    ----------
    sl_id : str
        Searchlight identifier (filename stem).
    clip : {"Intact", "SFix", "SRnd"}
        Movie clip condition to load.
    base_dir : str
        Base directory containing SL `.h5` files.

    Returns
    -------
    list of ndarray
        - If `clip == "Intact"`: `[D]` where `D` has shape
          (subjects, viewings, timepoints, voxels).
        - If scrambled (SFix/SRnd): `[D_v1, D_v2]` with subject halves.

    Notes
    -----
    - Each `D_orig[subj][clip]` is shaped (viewings, timepoints, voxels).
    - This function stacks subject data into the leading dimension.
    """
    fp = os.path.join(base_dir, f"{sl_id}.h5")
    D_orig = dd.io.load(fp)
    N_vox = D_orig[list(D_orig.keys())[0]][clip].shape[2]
    if N_vox == 0:
        return []
    N_subj = len(D_orig)
    D = np.zeros((N_subj, 6, 60, N_vox))
    for i, s in enumerate(D_orig.keys()):
        D[i] = D_orig[s][clip]
    return [D] if clip == "Intact" else [D[: N_subj // 2], D[N_subj // 2 :]]


def load_valid_mask(path: str = "/data/gbh/data/valid_vox.nii") -> np.ndarray:
    """
    Load the 3D boolean mask of valid voxels.

    Parameters
    ----------
    path : str
        NIfTI path for the valid voxels volume.

    Returns
    -------
    ndarray, shape (Z, Y, X), dtype=bool
        True where a voxel is valid.
    """
    return nib.load(path).get_fdata().T > 0


def load_sl_voxels(path: str = "/data/gbh/data/SL/SL_allvox.p") -> Mapping[int, np.ndarray]:
    """
    Load the SL→voxel membership mapping.

    Parameters
    ----------
    path : str
        Pickle path storing a dict[int -> np.ndarray of voxel indices].

    Returns
    -------
    Mapping[int, ndarray]
        Searchlight index -> indices within the valid-voxel mask.
    """
    return pickle.load(open(path, "rb"))


def project_sl_to_vox2d(
    sl_values: Mapping[int, float],
    mask: np.ndarray,
    sl_vox: Mapping[int, np.ndarray],
) -> np.ndarray:
    """
    Average SL-level scalars onto the valid-voxel vector.

    Parameters
    ----------
    sl_values : mapping
        SL index -> scalar (e.g., per-SL WBS change).
    mask : ndarray, shape (Z, Y, X), bool
        Valid-voxel mask.
    sl_vox : mapping
        SL index -> 1D indices of covered voxels within `mask`.

    Returns
    -------
    ndarray, shape (n_vox,)
        Vector of voxel values; NaN where no SL covered the voxel.

    Notes
    -----
    When multiple SLs cover a voxel, values are averaged.
    """
    n_vox = int(mask.sum())
    vec = np.zeros(n_vox, dtype=float)
    counts = np.zeros(n_vox, dtype=int)
    for sl_idx, val in sl_values.items():
        vox = sl_vox[int(sl_idx)]
        vec[vox] += float(val)
        counts[vox] += 1
    nz = counts > 0
    vec[~nz] = np.nan
    vec[nz] = vec[nz] / counts[nz]
    return vec


def vox2d_to_volume3d(vec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Lift a valid-voxel vector back into a 3D volume.

    Parameters
    ----------
    vec : ndarray, shape (n_vox,)
        Values for valid voxels.
    mask : ndarray, shape (Z, Y, X), bool
        Valid-voxel mask used to place values.

    Returns
    -------
    ndarray, shape (Z, Y, X)
        Volume with NaN outside the mask.
    """
    Z, Y, X = mask.shape
    assert int(mask.sum()) == len(vec), "Mismatch in voxel counts"
    vol = np.zeros((Z, Y, X), dtype=float)
    vol[mask] = vec
    vol[~mask] = np.nan
    return vol
