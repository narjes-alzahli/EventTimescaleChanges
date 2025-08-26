"""
Recall analyses
================

Prepare recall data, bootstrap the slope linking WBS change to recall,
and project significant SLs to a voxel map.

"""

from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from wbs_core import compute_wbs_for_sls, compute_wbs_changes
from wbs_viz import plot_wbs_change_map


def prepare_recall_data(recall_data: Mapping[str, Sequence[float]]):
    """
    Prepare recall scores and return a summary bar plot.

    Parameters
    ----------
    recall_data : mapping
        subject_id -> [Intact, SFix, SRnd] scores.

    Returns
    -------
    (scores, fig)
        scores : ndarray, shape (subjects, 3)
        fig : matplotlib.figure.Figure
    """
    data = {k: list(v) for k, v in recall_data.items()}

    # predtrw01 subjects watched SFix condition using original clip2, SRnd on orignal clip3. Vice versa for predtrw02 subjects.
    for subj, s in data.items():
        if "predtrw01" in subj:  # order swap for predtrw01 cohort
            s[1], s[2] = s[2], s[1]
    M = np.array(list(data.values()), dtype=float)
    fig = plt.figure(figsize=(7, 5))
    means = M.mean(0)
    ses = M.std(0) / np.sqrt(len(M))
    plt.bar(["Intact", "SFix", "SRnd"], means, yerr=ses, capsize=5, edgecolor="black", linewidth=2)
    plt.ylabel("Recall Score"); plt.tight_layout()
    return M, fig


def corr_recall_vs_wbs(timescale_changes: Mapping[str, float], recall: np.ndarray, p_thresh: float = 0.05):
    """
    Bootstrap the slope linking WBS change to recall and plot significant SLs.

    Parameters
    ----------
    timescale_changes : mapping
        SL -> scalar WBS change per subject or aggregated (see caller).
    recall : ndarray, shape (subjects, 3)
        Recall scores (columns: Intact, SFix, SRnd).
    p_thresh : float
        Significance level for bootstrap sign test.

    Returns
    -------
    (sig_dict, plots)
        sig_dict : dict[str, float]
            SL -> mean WBS change for significant associations.
        plots : dict[str, Figure]
            SL -> Scatter + bootstrap band per significant SL.
    """
    SLs = list(timescale_changes.keys())
    y = recall.mean(axis=1)
    plots = {}
    sig = {}
    n_boot = 5000

    for sl_id in SLs:
        X = np.array(timescale_changes[sl_id], dtype=float).reshape(-1, 1)
        if X.size != y.size:
            continue
        slopes = np.zeros(n_boot)
        grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        preds = np.zeros((n_boot, len(grid)))
        for b in range(n_boot):
            idx = resample(np.arange(X.shape[0]))
            m = LinearRegression().fit(X[idx], y[idx])
            slopes[b] = m.coef_[0]
            preds[b] = m.predict(grid)
        pos = (slopes > 0).sum()
        p = 2 * min(pos, n_boot - pos) / n_boot
        if p <= p_thresh:
            sig[sl_id] = float(np.mean(timescale_changes[sl_id]))
            f = plt.figure(figsize=(8, 5))
            plt.scatter(X, y, s=40)
            mean = preds.mean(0); sd = preds.std(0)
            plt.plot(grid.flatten(), mean)
            plt.fill_between(grid.flatten(), mean - 2 * sd, mean + 2 * sd, alpha=0.2)
            plt.xlabel("WBS change"); plt.ylabel("Recall score")
            plt.tight_layout()
            plots[sl_id] = f
    return sig, plots

def recall_analysis(conjunction_sls: Sequence[str], recall_data: Mapping[str, Sequence[float]], params: Mapping[str, object], fixed_ev: int, n_cv: int = 5, clips: Sequence[str] = ["Intact", "SFix", "SRnd"], template_path: str = "./MNI152_T1_brain_template.nii"):
    """
    Relate per-SL WBS changes to recall performance.

    Parameters
    ----------
    conjunction_sls : mapping
        SL id surviving across-clip conjunction (candidate set)
    recall_data : mapping
        subject_id -> [Intact, SFix, SRnd] scores.
    params : mapping
        Expect keys `nEvents`, `perSubject`, `views_PERMS`.
    fixed_ev : int
        Event count at which to compute change.
    n_cv : int
        # of CV folds (subject splits) to average.
    clips : Sequence[str]
        Names of movie clips used
    template_path : str
        Path to a template NIfTI whose affine/header should be reused.

    Returns
    -------
    (recall_map, plots)
        recall_map : ndarray (Z, Y, X)
            Voxelized map of SLs significantly associated with recall (NaN outside mask).
        recall_by_clip_plot: Figure
            Summary bar plot
        corr_plots : list[Figure]
            Per-SL scatter plots for significant SLs.
    """

    # 0) prepare recall
    recall, recall_by_clip_plot = prepare_recall_data(recall_data)

    # 1) get average per-subject recall across clips
    avg_recall_per_subject = recall.mean(axis=1)  # shape: (subjects,)

    # 2) compute per-subject WBS changes on conjunction SLs for a fixed event timescale
    indiv_wbc_changes_per_clip: Mapping[str, Mapping[str, np.ndarray]] = {}
    # per-subject, no nulls
    ps_params = dict(params); ps_params["views_PERMS"] = 0; ps_params["perSubject"] = 1; ps_params["event_PERMS"] = 0 
    for clip in clips:
        indiv_wbs = compute_wbs_for_sls(conjunction_sls, clip, ps_params)
        indiv_wbc_changes_per_clip[clip], _ = compute_wbs_changes(indiv_wbs, ps_params, fixed_ev, n_cv)

    # 3) computer average per-subject WBC changed across clips at a fixed event timescale
    common_sls = set.intersection(*(set(indiv_wbc_changes_per_clip[c].keys()) for c in clips))
    indiv_avg_change_across_clips = {
        sl_id: np.mean([np.asarray(indiv_wbc_changes_per_clip[c][sl_id]) for c in clips], axis=0)
        for sl_id in sorted(common_sls)
    }

    # 4) bootstrap correlation between average per-subject recall across clips and average per-subject WBC changed across clips for a fixed event timescale
    sig_changes, corr_plots = corr_recall_vs_wbs(indiv_avg_change_across_clips, avg_recall_per_subject, p_thresh=0.05 / max(1, len(conjunction_sls)))  
    recall_map = plot_wbs_change_map(sig_changes, template_path)

    return recall_map, recall_by_clip_plot, corr_plots