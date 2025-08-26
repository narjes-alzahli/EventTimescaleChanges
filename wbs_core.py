"""
Core WBS computations
======================

Implements the core WBS statistic, permutations/splits, grid computation
over clip viewings × event counts × permutations, and map generation helpers.

All functions are documented with expected shapes to make the code appendix-ready.

"""

from __future__ import annotations
import math
import random
from itertools import permutations
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import norm
from brainiak.eventseg.event import EventSegment
from tqdm import tqdm

from io import load_sl_h5, project_sl_to_vox2d, vox2d_to_volume3d
from wbs_stats import two_tailed_p_value, perform_bh_fdr
from wbs_viz import plot_wbs_change_map
import nibabel as nib


__all__ = [
    "valid_voxel_mask",
    "permute_view_order",
    "build_event_permutation_table",
    "compute_wbs_stat",
    "compute_wbs_grid",
    "compute_wbs_for_sls",
    "combine_scrambled_halves",
    "wbs_delta_at_event",
    "compute_wbs_changes",
    "generate_wbs_change_maps",
    "filter_wbs",
]


def valid_voxel_mask(data: np.ndarray) -> np.ndarray:
    """
    Identify voxels with non-zero signal across time.

    Parameters
    ----------
    data : ndarray, shape (subjects, viewings, timepoints, voxels)

    Returns
    -------
    ndarray, shape (voxels,), bool
        True for voxels that have any non-zero timecourse across viewings.
    """
    avg = data.mean(axis=0)  # [view, time, voxel]
    return np.all(np.any(avg != 0, axis=1), axis=0)


def permute_view_order(D: np.ndarray, perm_i: int) -> np.ndarray:
    """
    Shuffle viewing order per subject deterministically.

    Parameters
    ----------
    D : ndarray, shape (subjects, viewings, timepoints, voxels)
        Original data.
    perm_i : int
        Permutation index; 0 leaves data unchanged.

    Returns
    -------
    ndarray
        Permuted copy of `D`.
    """
    out = D.copy()
    if perm_i > 0:
        for subj in range(D.shape[0]):
            rng = np.random.RandomState(subj + perm_i * 1000)
            rng.shuffle(out[subj])
    return out


def build_event_permutation_table(params: Mapping[str, object], sl: str) -> list[np.ndarray]:
    """
    Build deterministic permutations of event-segment orders per event count.

    Parameters
    ----------
    params : mapping
        Must provide `nEvents` and `event_PERMS`.
    sl : str
        Searchlight ID used to seed the permutation RNG.

    Returns
    -------
    list of ndarray
        For each event count n, an array of shape (K, n) of permutations,
        with the first row being the identity order.
    """
    seed = hash(sl) % (2**32)
    rng = random.Random(seed)
    out: list[np.ndarray] = []
    for n_ev in params["nEvents"]:
        values = list(range(int(n_ev)))
        all_perms = list(permutations(values))
        non_identical = [p for p in all_perms if list(p) != values]
        rng.shuffle(non_identical)
        selected = [values] + non_identical[: int(params["event_PERMS"])]
        out.append(np.array(selected, dtype=object))
    return out


def compute_wbs_stat(hmm1: EventSegment, hmm2: EventSegment, g1: np.ndarray, g2: np.ndarray, per_subject: int) -> np.ndarray:
    """
    Compute WBS at group or subject level from two HMMs (split halves).

    Parameters
    ----------
    hmm1, hmm2 : EventSegment
        HMMs trained on group-averaged data for the two splits.
    g1, g2 : ndarray, shape (subjects, timepoints, voxels)
        Training data (per split) from which correlations are computed.
    per_subject : int
        1 => return a vector per subject; 0 => return a scalar (in a 0-D np.ndarray)

    Returns
    -------
    ndarray
        If `per_subject==1`: shape (subjects_total,).
        Else: scalar inside a 0-D ndarray (for consistency with upstream code).

    Notes
    -----
    WBS = mean(within-event corr) − mean(between-event corr) at lag=5 TRs.
    """
    step = 5
    if per_subject:
        n_subj = g1.shape[0] + g2.shape[0]
        wbs = np.empty((n_subj), dtype=float)
        mlm1 = np.argmax(hmm1.segments_[0], axis=1)
        mlm2 = np.argmax(hmm2.segments_[0], axis=1)
        for s in range(g2.shape[0]):
            corr = np.corrcoef(g2[s])
            idx = np.arange(len(mlm1) - step); off = idx + step
            within = corr[idx, off][(mlm1[idx] == mlm1[off])]
            between = corr[idx, off][(mlm1[idx] != mlm1[off])]
            wbs[s] = float(within.mean() - between.mean())
        for s in range(g1.shape[0]):
            corr = np.corrcoef(g1[s])
            idx = np.arange(len(mlm2) - step); off = idx + step
            within = corr[idx, off][(mlm2[idx] == mlm2[off])]
            between = corr[idx, off][(mlm2[idx] != mlm2[off])]
            wbs[s + g2.shape[0]] = float(within.mean() - between.mean())
        return wbs
    else:
        G1 = g1.mean(0); G2 = g2.mean(0)
        corr1 = np.corrcoef(G1); corr2 = np.corrcoef(G2)
        mlm1 = np.argmax(hmm1.segments_[0], axis=1)
        mlm2 = np.argmax(hmm2.segments_[0], axis=1)
        idx = np.arange(len(mlm1) - step); off = idx + step
        w12 = corr2[idx, off]; w21 = corr1[idx, off]
        d12 = (w12[(mlm1[idx] == mlm1[off])].mean() - w12[(mlm1[idx] != mlm1[off])].mean())
        d21 = (w21[(mlm2[idx] == mlm2[off])].mean() - w21[(mlm2[idx] != mlm2[off])].mean())
        return (d12 + d21) / 2


def compute_wbs_grid(g1: np.ndarray, g2: np.ndarray, params: Mapping[str, object], ev_perm_table: list[np.ndarray]) -> np.ndarray:
    """
    Compute WBS across (event_PERMS+1) × VIEWINGS × len(nEvents).

    Parameters
    ----------
    g1, g2 : ndarray
        Shape (subjects, viewings, timepoints, voxels) for each split.
    params : mapping
        Expect keys: `event_PERMS`, `VIEWINGS`, `nEvents`, `perSubject`.
    ev_perm_table : list of ndarray
        Permutation indices for each event count (first row is identity).

    Returns
    -------
    ndarray
        perSubject=0: shape (event_PERMS+1, VIEWINGS, len(nEvents))
        perSubject=1: shape (event_PERMS+1, VIEWINGS, len(nEvents), subjects_total)
    """
    if params.get("perSubject", 0):
        out = np.empty(
            (int(params["event_PERMS"]) + 1, int(params["VIEWINGS"]), len(params["nEvents"]), g2.shape[0] + g1.shape[0]),
            dtype=object,
        )
    else:
        out = np.empty(
            (int(params["event_PERMS"]) + 1, int(params["VIEWINGS"]), len(params["nEvents"])),
            dtype=object,
        )

    valid = valid_voxel_mask(g1) & valid_voxel_mask(g2)
    g1 = g1[:, :, :, valid]
    g2 = g2[:, :, :, valid]

    G1 = g1.mean(axis=0)
    G2 = g2.mean(axis=0)

    for v in range(int(params["VIEWINGS"])):
        for ev_i, n_ev in enumerate(params["nEvents"]):
            hmm1 = EventSegment(int(n_ev)).fit(G1[v])
            hmm2 = EventSegment(int(n_ev)).fit(G2[v])
            maxp = min(math.factorial(int(n_ev)), int(params["event_PERMS"]) + 1)
            for p in range(maxp):
                out[p, v, ev_i] = compute_wbs_stat(hmm1, hmm2, g1[:, v], g2[:, v], int(params.get("perSubject", 0)))
    return out


def compute_wbs_for_sls(sls: Iterable[str], clip: str, params: Mapping[str, object]) -> dict[str, np.ndarray]:
    """
    Compute WBS grids for a set of searchlights in a given clip.

    Parameters
    ----------
    sls : iterable of str
        Searchlight IDs (filenames without extension).
    clip : str
        Clip condition ("Intact", "SFix", "SRnd").
    params : mapping
        See `PARAMS` in `main.py`.

    Returns
    -------
    dict[str, ndarray]
        SL id -> WBS grid (see `compute_wbs_grid` return shape).
    """
    W: dict[str, np.ndarray] = {}
    for sl in tqdm(list(sls)):
        ev_table = build_event_permutation_table(params, str(sl))
        data = load_sl_h5(str(sl), clip)
        if not data:
            continue

        # Allocate container per SL
        if params.get("perSubject", 0):
            n_subj = data[0].shape[0]
            W[str(sl)] = np.empty(
                (len(data), int(params["views_PERMS"]) + 1, int(params["subj_SPLITS"]), int(params["event_PERMS"]) + 1,
                 int(params["VIEWINGS"]), len(params["nEvents"]), n_subj),
                dtype=object,
            )
        else:
            W[str(sl)] = np.empty(
                (len(data), int(params["views_PERMS"]) + 1, int(params["subj_SPLITS"]), int(params["event_PERMS"]) + 1,
                 int(params["VIEWINGS"]), len(params["nEvents"])),
                dtype=object,
            )

        # Fill container across view-order permutations and CV splits
        for v_i, D in enumerate(data):
            nS = D.shape[0]
            for vp in range(int(params["views_PERMS"]) + 1):
                perm_D = permute_view_order(D, vp)
                for split in range(int(params["subj_SPLITS"])):
                    rng = np.random.RandomState(int(params["random_split"]) + split)
                    idx = np.arange(nS); rng.shuffle(idx)
                    cut = int(np.floor(nS * (1 - float(params["test_size"])) ))
                    g1_idx, g2_idx = idx[:cut], idx[cut:]
                    grid = compute_wbs_grid(perm_D[g1_idx], perm_D[g2_idx], params, ev_table)

                    # subject reordering when perSubject=1 (exactly like your original) ---
                    if params.get("perSubject", 0):
                        # grid shape: (event_PERMS+1, VIEWINGS, len(nEvents), subjects_total)
                        n_g2 = len(g2_idx)
                        n_tot = grid.shape[-1]
                        ordered = np.empty_like(grid)
                        # first segment (0:n_g2) corresponds to g2 subjects
                        ordered[..., g2_idx] = grid[..., :n_g2]
                        # remaining segment corresponds to g1 subjects
                        ordered[..., g1_idx] = grid[..., n_g2:n_tot]
                        W[str(sl)][v_i, vp, split] = ordered
                    else:
                        W[str(sl)][v_i, vp, split] = grid
        
    return combine_scrambled_halves(W, per_subject=params["perSubject"])

def combine_scrambled_halves(W: Mapping[str, np.ndarray], per_subject: int = 0) -> dict[str, np.ndarray]:
    """
    Combine scrambled-clip halves into per-SL WBS arrays.

    Parameters
    ----------
    W : mapping
        Outputs of `compute_wbs_for_sls` for SFix/SRnd (two halves).
    per_subject : int
        If 1, concatenate subject dimension; else average halves.

    Returns
    -------
    dict[str, ndarray]
        Combined per-SL WBS arrays.
    """
    out = {}
    for sl_id, arr in W.items():

        # Intact: arr.shape[0] == 1 (one dataset)
        if arr.shape[0] == 1:
            out[sl_id] = arr[0]
            continue

        # Scrambled (SFix/SRnd): two halves in arr[0], arr[1]
        if not per_subject:
            v1, v2 = arr[0], arr[1]
            out[sl_id] = (v1 + v2) / 2.0
        else:
            # last axis is subjects – concatenate there
            out[sl_id] = np.concatenate([arr[0], arr[1]], axis=-1)

    return out


def wbs_delta_at_event(sl_arr: np.ndarray, fixed_ev: int, offset: int, n_cv: int = 5) -> float:
    """
    WBS change at a fixed event count: (avg views 2–6) − (view 1).

    Parameters
    ----------
    sl_arr : ndarray
        WBS grid for a single SL: shape (perms, viewings, events) or
        (perms, viewings, events, subjects) when perSubject=1.
    fixed_ev : int
        Absolute event count (e.g., 2, 10).
    offset : int
        Minimum event count in `params["nEvents"]` (usually 2).
    n_cv : int
        # CV folds to average (leading dimension of the first axis).

    Returns
    -------
    float
        WBS change at `fixed_ev`.
    """
    ev = fixed_ev - offset
    first_view = sl_arr[:n_cv, 0].mean(0)              # [events]
    repeats_avg = sl_arr[:n_cv, 1:].mean(0).mean(0)    # [events]
    return float(repeats_avg[ev] - first_view[ev])


def compute_wbs_changes(meaningful: Mapping[str, np.ndarray], params: Mapping[str, object], fixed_ev: int, n_cv: int = 5) -> tuple[dict[str, float], dict[str, float]]:
    """
    Estimate per-SL WBS changes and permutation p-values at `fixed_ev`.

    Parameters
    ----------
    meaningful : mapping
        SL -> WBS grids as returned by `compute_wbs_for_sls`.
    params : mapping
        Expect keys `nEvents`, `perSubject`, `views_PERMS`.
    fixed_ev : int
        Event count at which to compute change.
    n_cv : int
        # of CV folds to average.

    Returns
    -------
    (changes, p_vals) : (dict[str, float], dict[str, float])
        Observed change and p-value (NaN if nulls not computed) per SL.
    """
    changes: dict[str, float] = {}
    pvals: dict[str, float] = {}

    offset = int(params["nEvents"][0])
    nNull = int(params["views_PERMS"])     # viewing-order permutations as nulls
    perSub = int(params.get("perSubject", 0))

    for sl_id, L in meaningful.items():
        diff = wbs_delta_at_event(L[0], fixed_ev, offset, n_cv)
        if perSub or nNull == 0:
            p = float("nan")
        else:
            nulls = [wbs_delta_at_event(L[perm], fixed_ev, offset, n_cv) for perm in range(1, nNull + 1)]
            p = two_tailed_p_value(diff, np.array(nulls))
        changes[sl_id] = diff
        pvals[sl_id] = p

    return changes, pvals


def generate_wbs_change_maps(
    meaningful_WBS: Mapping[str, np.ndarray],
    params: Mapping[str, object],
    fixed_ev: int,
    fdr_threshold: float = 0.05,
    n_cv: int = 5,
    template_path: str = "./MNI152_T1_brain_template.nii"
) -> tuple[nib.Nifti1Image, nib.Nifti1Image, list[str]]:
    """
    Create unfilt/filt WBS-change maps and return FDR-surviving SL ids.

    Parameters
    ----------
    meaningful_WBS : mapping
        SL -> WBS grids restricted to meaningful SLs.
    params : mapping
        Pipeline configuration (see `PARAMS`).
    fixed_ev : int
        Event count for change computation.
    fdr_threshold : float
        BH-FDR threshold.
    n_cv : int
        # of CV folds to average.
    template_path : str
        Path to a template NIfTI whose affine/header should be reused.

    Returns
    -------
    (unfilt_img, filt_img, fdr_sls)
        WBS change maps as NIfTI images and the list of SL ids that survive FDR.
    """
    changes, p_vals = compute_wbs_changes(meaningful_WBS, params, fixed_ev, n_cv)
    
    fdr_sls = perform_bh_fdr(p_vals, alpha=fdr_threshold)


    unfilt_img = plot_wbs_change_map(changes, template_path)

    sig_changes = {int(k): float(changes[k]) for k in fdr_sls}
    filt_img = plot_wbs_change_map(sig_changes, template_path)

    return unfilt_img, filt_img, fdr_sls


def filter_wbs(data: Mapping[str, np.ndarray], n_cv: int = 5, alpha: float = 0.05) -> list[str]:
    """
    Filter searchlights (SLs) to keep only those with meaningful WBS structure.

    Conditions:
    1. Each viewing must have at least one positive WBS at some event count.
    2. For each viewing, at least one event count must be both positive and
       significantly above its permutation null (p < alpha).

    Parameters
    ----------
    data : mapping
        SL -> WBS grids.
    n_cv : int
        # of CV folds to average in the “true” (non-null) grid.
    p : float
        Ignored here (legacy signature for compatibility).

    Returns
    -------
    list[str]
        SL ids that meet the basic positivity criterion.
    """
    keep: list[str] = []

    for sl, sl_data in data.items():
        # sl_data: (nViewingPerms, n_cv, nEventPerms, VIEWINGS, nEvents)

        # ----- TRUE (event_perm = 0) -----
        # mean across CV splits
        true_wbs = sl_data[0, :n_cv, 0].mean(axis=0)   # (VIEWINGS, nEvents)

        # ---- Condition 1: in each viewing, some event has positive WBS ----
        if not np.all(np.any(true_wbs > 0, axis=1)):
            continue

        # If no null event permutations exist, accept SLs that pass Cond.1 (matches your intent)
        if sl_data.shape[2] <= 1:
            keep.append(sl)
            continue

        # ----- NULLS (event_perm >= 1) -----
        # Mean across CV splits
        null_wbs = sl_data[0, :n_cv, 1:].mean(axis=0)   # (n_null_event_perms, VIEWINGS, nEvents)    

        # Pool across *all event counts* for each viewing to get ONE null distribution per viewing.
        viewing_null = np.transpose(null_wbs, (1, 0, 2)).reshape(null_wbs.shape[1], -1) # (VIEWINGS, n_null_event_perms * nEvents)

        # Per-viewing null mean/std
        null_mean_v = viewing_null.mean(axis=1)         # (VIEWINGS,)
        null_std_v  = viewing_null.std(axis=1, ddof=1)  # (VIEWINGS,)

        # Broadcast to (VIEWINGS, nEvents) so every event in a viewing shares the same pooled null.
        null_mean = null_mean_v[:, None] # (VIEWINGS, nEvents)
        null_std  = null_std_v[:, None]  # (VIEWINGS, nEvents)

        # Avoid div-by-zero: if std==0, treat as non-sig by making z -> 0 advantage disappear
        safe_std  = np.where(null_std > 0, null_std, np.inf)

        # z & one-sided p (testing true_mean > null_mean)
        z = (true_wbs - null_mean) / safe_std                        
        p = norm.sf(z)                                       

        # ---- Condition 2: for each viewing, at least one event is positive & significant ----
        pos_and_sig = (true_wbs > 0) & (p < alpha)                
        if np.all(np.any(pos_and_sig, axis=1)):
            keep.append(sl) 

    return keep
