"""
Timescale Changes paper pipeline (main)
========================

This module orchestrates the analysis for changes in event structure across repeated viewings of three clip conditions (Intact, SFix, SRnd).

Pipeline (high-level)
---------------------
1. Gather searchlights (SLs) and load per-SL data.
2. Compute Within- vs. Between-Event Similarity (WBS) for all SLs within each clip.
3. Filter to SLs with meaningful event structure for at least one event timescale in each viewing.
4. Recompute WBS restricted to "meaningful" SLs, along with permutations to measure significant WBS changes.
5. Generate per-clip WBS-change maps at target event counts: slow timescale (2 events) and fast timescale (10 events).
6. Compute across-clip conjunction maps (consistent effects).
7. Relate WBS changes to free-recall behavior.

Reproducibility
---------------
- Deterministic random seeds are used for permutations and splits.
- Parameters are centralized in `PARAMS` for traceability.

Notes
-----
- File paths match the original environment; adjust as needed.
- WBS is positive when within-event correlations exceed between-event
  correlations at a lag (default 5 TRs).

"""

from typing import Dict, List
import os
import json
import numpy as np

from wbs_core import (
    compute_wbs_for_sls,
    generate_wbs_change_maps,
    filter_wbs as filter_WBS,
)
from wbs_overlap import overlapping_regions
from wbs_viz import (
    save_nifti,
    save_fig,
    wbc_plots,    
    ensure_dir,      
)
from recall import recall_analysis


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: If True, restrict to a small subset of SLs for quick iteration.
DRY_RUN: bool = False

#: Movie clip conditions to analyze.
CLIPS: List[str] = ["Intact", "SFix", "SRnd"]

#: Absolute event counts at which to compute WBS change (e.g., 2 or slow timescale vs. 10 or fast timescale).
CHANGE_AT_EVENTS: List[int] = [2, 10]

#: Brain MNI152 template/header path
TEMPLATE_PATH: str = "./MNI152_T1_brain_template.nii"

#: Pipeline parameters controlling data permutations, participat splits, and event grids.
PARAMS: Dict[str, object] = {
    "test_size": 0.5,           # held-out fraction for the split
    "nEvents": np.arange(2, 11),# event counts grid: 2..10
    "event_PERMS": 50,          # # of event-segment permutations per event count/timescale (nulls)
    "views_PERMS": 0,           # # of viewing-order permutations (nulls)
    "VIEWINGS": 6,              # viewing repeats per clip
    "perSubject": 0,            # 0: group level; 1: subject level outputs
    "subj_SPLITS": 5,           # # of cross-validation splits
    "random_split": 42         # RNG seed for subject splits
}

#: Free-recall scores per subject for each movie clip (Intact, SFix, SRnd). 
RECALL_FILE = "./recall_data.json"
with open(RECALL_FILE, "r") as f:
    RECALL_DATA_DICT = json.load(f)

#: Where to save all outputs (maps, plots, metadata).
OUTPUT_DIR: str = "./outputs"

#: Desired number of subject splits to average over
N_CV = 5

#: Sample searchlights
sample_sls = {"Intact":{2:['933','2743'],
                        10:['1466','3471']},
                "SFix":{2:['2272','1828']},
                "SRnd":{2:['3773','1333','1464'],
                        10:['1915']},
                "conjunction":{2:{"1715","2336"}},
                "recall":{2:{"1715","1829"}}}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full analysis pipeline.

    Side Effects
    ------------
    - Saves NIfTI maps under ``{OUTPUT_DIR}/brain_maps/...``
    - Saves per-SL diagnostic plots under ``{OUTPUT_DIR}/wbs_plots/...``
    - Saves recall figures and map under ``{OUTPUT_DIR}/recall_plots/...``

    Notes
    -----
    Set ``DRY_RUN=True`` above to analyze a small subset of searchlights
    for fast iteration while drafting the paper.
    """

    # STEP 0 - prepare results directories and variables
    ensure_dir(OUTPUT_DIR)
    brain_map_dir = os.path.join(OUTPUT_DIR, "brain_maps"); ensure_dir(brain_map_dir)
    wbs_plots_dir = os.path.join(OUTPUT_DIR, "wbs_plots"); ensure_dir(wbs_plots_dir)
    recall_plots_dir = os.path.join(OUTPUT_DIR, "recall_plots"); ensure_dir(recall_plots_dir)

    all_wbs: Dict[str, Dict[str, np.ndarray]] = {}
    meaningful_sls_by_clip: Dict[str, List[str]] = {}
    meaningful_wbs_by_clip: Dict[str, Dict[str, np.ndarray]] = {}
    conj_sls_per_ev : Dict[str, List[str]] = {}

    # STEP 1 — discover searchlights
    print("STEP 1 — total number of searchlights:", len(full_sl_list))
    full_sl_list = [f.split(".")[0] for f in os.listdir("/data/gbh/data/SL/") if f.endswith(".h5")]
    if DRY_RUN:
        full_sl_list = full_sl_list[:20]

    # STEP 2 — compute WBS for every clip for all SLs (with permutations: shuffled event-order)
    print(f"STEP 2 — computing WBS for all SLs")
    for clip in CLIPS:
        print(f"Running: clip {clip}")
        all_wbs[clip] = compute_wbs_for_sls(full_sl_list, clip, PARAMS)

    # STEP 3 — filter SLs with meaningful event structure
    print(f"STEP 3 — SLs with meaningful event structure")
    for clip in CLIPS:
        print(f"Running: clip {clip}")
        meaningful_sls_by_clip[clip] = filter_WBS(all_wbs[clip], n_cv=N_CV, p=0.05)

    # STEP 4 — recompute WBS for only meaningful SLs (with permutations: shuffled viewing-order)
    print("STEP 4 — recompute WBS for only meaningful SLs (with shuffled-viewing-order permutations)")
    for clip in CLIPS:
        print(f"Running: clip {clip}")
        PARAMS["event_PERMS"] = 0; PARAMS["views_PERMS"] = 50
        meaningful_wbs_by_clip[clip] = compute_wbs_for_sls(meaningful_sls_by_clip[clip], clip, PARAMS)

    # STEP 5 — per-clip change maps at target event counts
    print("STEP 5 — per-clip change maps at target event counts")
    for clip in CLIPS:
        for ev in CHANGE_AT_EVENTS:
            print(f"Running: clip {clip}; event timescale {ev}")
            unfilt_map, filt_map, fdr_sls = generate_wbs_change_maps(meaningful_wbs_by_clip[clip], PARAMS, fixed_ev=ev, fdr_threshold=0.05, n_cv=N_CV, template_path=TEMPLATE_PATH)

            # Save results
            save_nifti(unfilt_map, os.path.join(brain_map_dir, f"{clip}_wbsChange_unthresh_ev{ev}.nii.gz"))
            save_nifti(filt_map, os.path.join(brain_map_dir, f"{clip}_wbsChange_thresh_ev{ev}.nii.gz"))
            if fdr_sls: # per-SL diagnostic plots for sample surviving SLs
                wbc_plots(sls=sample_sls[clip][ev], clip=clip, wbs_dict=meaningful_wbs_by_clip[clip], out_dir=wbs_plots_dir, n_cv=N_CV, prefix=f"{clip}_ev{ev}_")

    # STEP 6 — across-clip conjunctions for target event count
    print("STEP 6 — across-clip conjunctions for target event count")
    for ev in CHANGE_AT_EVENTS:
        print(f"Running: event timescale {ev}")
        conj_map, conj_sls_per_ev[ev] = overlapping_regions(meaningful_wbs_by_clip, meaningful_sls_by_clip, ev, params=PARAMS, clips=CLIPS, template_path=TEMPLATE_PATH)

        # Save results
        if conj_sls_per_ev[ev]:
            save_nifti(conj_map, os.path.join(brain_map_dir, f"conj_ev{ev}.nii.gz"))  
            for clip in CLIPS:
                wbc_plots(sls=sample_sls["conjunction"][ev], clip=clip, wbs_dict=meaningful_wbs_by_clip[clip], out_dir=wbs_plots_dir, n_cv=N_CV, prefix=f"conjunction_{clip}_ev{ev}_")

    # STEP 7 — recall correlation (behavioral linkage)
    print("STEP 7 — recall correlation (behavioral linkage)")
    for ev in CHANGE_AT_EVENTS:
        print(f"Running: event timescale {ev}")

        if conj_sls_per_ev[ev]:
            recall_map, recall_data_plot, corr_recall_wbsChange_plots = recall_analysis(conj_sls_per_ev[ev], RECALL_DATA_DICT, PARAMS, ev, n_cv=N_CV, clips=CLIPS, template_path=TEMPLATE_PATH)

            # Save results
            save_nifti(recall_map, os.path.join(brain_map_dir, f"recall_ev{ev}.nii.gz"))
            save_fig(recall_data_plot, os.path.join(recall_plots_dir, f"recall_data.png"), dpi=150)
            for sl, fig in corr_recall_wbsChange_plots.items():
                save_fig(fig, os.path.join(recall_plots_dir, f"recall_corr_plot_ev{ev}_sl{sl}.png"), dpi=150)
            for clip in CLIPS:
                wbc_plots(sls=sample_sls["recall"][ev], clip=clip, wbs_dict=meaningful_wbs_by_clip[clip], out_dir=wbs_plots_dir, n_cv=N_CV, prefix=f"recall_{clip}_ev{ev}_")


if __name__ == "__main__":
    main()
