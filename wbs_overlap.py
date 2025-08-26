"""
Across-clip overlap and conjunction for WBS changes
==================================================

Find SLs with consistent WBS increases/decreases across Intact, SFix, SRnd,
and produce conjunction maps (FDR within each clip, then intersect).

"""

from __future__ import annotations
import numpy as np
from typing import Mapping, Sequence
from wbs_core import compute_wbs_changes
from wbs_stats import perform_bh_fdr
from wbs_viz import plot_wbs_change_map
import nibabel as nib


def split_by_direction(changes: dict[str, float], pvals: dict[str, float]):
    """
    Partition SLs into gain (>0) and loss (<0) sets, carrying p-values.

    Returns
    -------
    (gain_values, gain_p_values, loss_values, loss_p_values)
        Four dicts keyed by SL id.
    """
    gain_vals, loss_vals, gain_ps, loss_ps = {}, {}, {}, {}
    for sl_id, change_val in changes.items():
        if change_val > 0:
            gain_vals[sl_id] = change_val; gain_ps[sl_id] = pvals.get(sl_id, np.nan)
        elif change_val < 0:
            loss_vals[sl_id] = change_val; loss_ps[sl_id] = pvals.get(sl_id, np.nan)
    return gain_vals, gain_ps, loss_vals, loss_ps

def overlapping_regions(
    meaningful_wbs_clips: Mapping[str, Mapping[str, np.ndarray]],
    meaningful_wbs_sls: Mapping[str, list[str]],
    fixed_ev: int,
    params: Mapping[str, object],
    clips: Sequence[str] = ["Intact", "SFix", "SRnd"],
    template_path: str = "./MNI152_T1_brain_template.nii"
) -> tuple[list[nib.Nifti1Image], list[str]]:
    """
    Conjunction of significant WBS changes across Intact, SFix, and SRnd.

    Parameters
    ----------
    meaningful_wbs_clips : mapping
        clip -> (SL -> WBS grids) restricted to meaningful SLs.
    meaningful_wbs_sls : mapping
        clip -> list of meaningful SL ids (used to get overlap set).
    fixed_ev : int
        Event count at which to compute conjunctions.
    params : mapping
        Pipeline configuration.
    clips : Sequence[str]
        Names of movie clips used
    template_path : str
        Path to a template NIfTI whose affine/header should be reused.

    Returns
    -------
    (conjunction_maps, sig_overlap_sls)
        List of NIfTI maps (one per event count with overlap) and the final
        List of SL ids surviving the across-clip FDR conjunction.
    """

    overlap_all = set(meaningful_wbs_sls["Intact"]) & set(meaningful_wbs_sls["SFix"]) & set(meaningful_wbs_sls["SRnd"])
    conjunction_map: nib.Nifti1Image = None
    sig_conj_sls_avg_wbsChange: dict[str, float] = {}

    gain_vals, gain_ps, loss_vals, loss_ps = {}, {}, {}, {}

    # Compute per-clip changes and p-values on the overlap set
    for clip in clips:
        ch, pv = compute_wbs_changes({sl_id: meaningful_wbs_clips[clip][sl_id] for sl_id in overlap_all}, params, fixed_ev, n_cv=5)
        a, b, c, d = split_by_direction(ch, pv)
        gain_vals[clip], gain_ps[clip], loss_vals[clip], loss_ps[clip] = a, b, c, d

    # Directional conjunction (gains and losses separately)
    for direction_c, direction_p in [(gain_vals, gain_ps), (loss_vals, loss_ps)]:
        slsets = [set(direction_c[c].keys()) for c in clips]
        inter = set.intersection(*slsets) if slsets and slsets[0] else set()
        if not inter:
            continue

        # FDR per clip within the intersection, then intersect survivors
        kept = []
        for clip in clips:
            p_overlap = {sl_id: direction_p[clip][sl_id] for sl_id in inter}
            kept.append(set(perform_bh_fdr(p_overlap, alpha=0.05 ** (1/len(clips)))))
        sig_conj_sls = set.intersection(*kept)

        if sig_conj_sls:
            # Average effect across clips for surviving SLs
            for sl_id in sig_conj_sls:
                vals = [direction_c[c][sl_id] for c in clips]
                sig_conj_sls_avg_wbsChange[sl_id] = float(np.mean(vals))

            conjunction_map = plot_wbs_change_map(sig_conj_sls_avg_wbsChange, template_path)
            
    return conjunction_map, list(sig_conj_sls_avg_wbsChange.keys())