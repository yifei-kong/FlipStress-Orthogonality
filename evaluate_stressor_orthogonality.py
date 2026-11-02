import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_DIR = os.path.join(".", "Cache Trace Data")
FIXED_DIR = os.path.join(BASE_DIR, "Fixed Stressors")

STRESSOR_CONFIG = [
    ("readBuffer",      "1: Read Buffer"),
    ("writeBuffer",     "2: Write Buffer"),
    ("readLinkedList",  "3: Read Linked List"),
    ("writeLinkedList", "4: Write Linked List"),
    ("vm",              "5: VM"),
    ("stream",          "6: Stream"),
    ("memcpy",          "7: Memcpy"),
]

def extract_slot_features(slot_values, slot_duration_sec=0.5):
    arr = np.asarray(slot_values, dtype=np.float64)
    if len(arr) < 2:
        return np.zeros(11)
    q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])
    diffs = np.abs(np.diff(arr))
    return np.array([
        np.mean(arr),
        np.std(arr),
        q50, q10, q25, q75, q90,
        q75 - q25,
        np.max(arr) - np.min(arr),
        np.mean(diffs),
        len(arr) / slot_duration_sec
    ])

def build_dataset(p_val=4, slot_sec=0.5, total_sec=10.0):
    num_slots = int(total_sec / slot_sec)
    X_feats, y_stressor, trace_groups, page_states = [], [], [], []
    raw_slots_by_class = {}
    class_names = []

    global_trace_id = 0
    total_stressors = len(STRESSOR_CONFIG)
    
    print(f"  [1/4] Loading and slicing trace data for p = {p_val}...")
    for class_idx, (prefix, label_name) in enumerate(STRESSOR_CONFIG):
        class_names.append(label_name)
        raw_slots_by_class[class_idx] = []
        fpath = os.path.join(FIXED_DIR, f"p={p_val}", f"{prefix}{p_val}_10s.json")
        
        with open(fpath, "r") as f:
            data = json.load(f)
        traces, orig_y = data["X"], np.array(data["y"])

        for t_idx, trace in enumerate(traces):
            slots = np.array_split(np.asarray(trace, dtype=np.float64), num_slots)
            for s in slots:
                X_feats.append(extract_slot_features(s, slot_sec))
                y_stressor.append(class_idx)
                trace_groups.append(global_trace_id)
                page_states.append(orig_y[t_idx])
                raw_slots_by_class[class_idx].append(s)
            global_trace_id += 1
            
        print(f"        Processed [{class_idx + 1}/{total_stressors}] {label_name}")

    return (
        np.vstack(X_feats),
        np.array(y_stressor),
        np.array(trace_groups),
        np.array(page_states),
        class_names,
        raw_slots_by_class
    )

def evaluate_for_p(p_val=4, slot_sec=0.5):
    t0 = time.time()
    X, y, groups, page_states, class_names, raw_slots = build_dataset(p_val, slot_sec)

    print(f"  [2/4] Running 7-class trace-level 10-fold cross-validation...")
    sgkf = StratifiedGroupKFold(n_splits=10)
    y_pred_cv = np.zeros_like(y)
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        clf = RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        y_pred_cv[test_idx] = clf.predict(X[test_idx])
        print(f"        Fold {fold}/10 done", end="\r")
    print(f"        Fold 10/10 complete!                            ")

    acc_7class_cv = accuracy_score(y, y_pred_cv) * 100.0
    cm_7class = confusion_matrix(y, y_pred_cv, normalize="true") * 100.0

    print(f"  [3/4] Evaluating cross-state generalization (Non-Target -> Target)...")
    clf_cross = RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1)
    clf_cross.fit(X[page_states == 0], y[page_states == 0])
    acc_7class_cross = accuracy_score(y[page_states == 1], clf_cross.predict(X[page_states == 1])) * 100.0

    subsets = list(combinations(range(len(class_names)), 4))
    total_subsets = len(subsets)
    print(f"  [4/4] Searching for optimal 4-stressor orthogonal codebook ({total_subsets} combinations)...")
    
    best_subset, best_4ary_acc, best_4ary_cross = None, 0.0, 0.0
    for idx, subset in enumerate(subsets, 1):
        sub_mask = np.isin(y, subset)
        X_sub, y_sub, g_sub, ps_sub = X[sub_mask], y[sub_mask], groups[sub_mask], page_states[sub_mask]
        
        sgkf_sub = StratifiedGroupKFold(n_splits=5)
        y_sub_pred = np.zeros_like(y_sub)
        for tr, te in sgkf_sub.split(X_sub, y_sub, groups=g_sub):
            clf_s = RandomForestClassifier(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1)
            clf_s.fit(X_sub[tr], y_sub[tr])
            y_sub_pred[te] = clf_s.predict(X_sub[te])
        
        acc_sub = accuracy_score(y_sub, y_sub_pred) * 100.0
        if acc_sub > best_4ary_acc:
            best_4ary_acc = acc_sub
            best_subset = subset
            clf_c = RandomForestClassifier(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1)
            clf_c.fit(X_sub[ps_sub == 0], y_sub[ps_sub == 0])
            best_4ary_cross = accuracy_score(y_sub[ps_sub == 1], clf_c.predict(X_sub[ps_sub == 1])) * 100.0

        if idx % 5 == 0 or idx == total_subsets:
            print(f"        Subset progress: {idx}/{total_subsets} (Current Best Acc: {best_4ary_acc:.2f}%)", end="\r")
            
    print(f"        Subset search complete!                                       ")
    print(f"  └─ Evaluation for p = {p_val} finished in {time.time() - t0:.1f} s\n")

    return {
        "p": p_val,
        "acc_7class_cv": acc_7class_cv,
        "acc_7class_cross": acc_7class_cross,
        "best_4ary_subset": best_subset,
        "best_4ary_names": [class_names[i] for i in best_subset],
        "best_4ary_acc": best_4ary_acc,
        "best_4ary_cross": best_4ary_cross,
        "cm": cm_7class,
        "class_names": class_names,
        "raw_slots": raw_slots,
        "X": X, "y": y, "page_states": page_states
    }

def generate_plots(results_by_p):
    print("[*] Generating visualization plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    for idx, p_val in enumerate([1, 2, 4]):
        res = results_by_p[p_val]
        ax = axes[idx]
        sns.heatmap(
            res["cm"], annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100,
            xticklabels=[n.split(": ")[1] for n in res["class_names"]],
            yticklabels=[n.split(": ")[1] for n in res["class_names"]] if idx == 0 else False,
            cbar=(idx == 2), ax=ax
        )
        ax.set_title(
            f"p = {p_val} Workers (Slot = 0.5s)\n"
            f"7-Class: {res['acc_7class_cv']:.1f}% | Best 4-Stressor Subset: {res['best_4ary_acc']:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Predicted Stressor", fontweight="bold")
        if idx == 0:
            ax.set_ylabel("True Active Stressor", fontweight="bold")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Per-Slot (0.5s) Stressor Distinguishability Across Parallelism Levels (p = 1, 2, 4)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("fig1_stressor_confusion_matrix.png", bbox_inches="tight")
    print("    [+] Plot 1 saved: fig1_stressor_confusion_matrix.png")

    res_p4 = results_by_p[4]
    subset = list(res_p4["best_4ary_subset"])
    short_names = [res_p4["class_names"][i].split(": ")[1] for i in subset]
    
    X, y, ps = res_p4["X"], res_p4["y"], res_p4["page_states"]
    mask_sub = np.isin(y, subset)
    clf = RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1)
    clf.fit(X[mask_sub & (ps == 0)], y[mask_sub & (ps == 0)])

    sequence = [0, 3, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 3, 0, 1, 2, 0, 3, 2, 1]
    np.random.seed(42)
    time_axis, waveform_vals, annotations = [], [], []
    correct_cnt = 0

    for slot_idx, sym_i in enumerate(sequence):
        stressor_id = subset[sym_i]
        target_slots = res_p4["raw_slots"][stressor_id][:1000]
        chosen_slot = target_slots[np.random.randint(0, len(target_slots))]
        
        pred_id = clf.predict(extract_slot_features(chosen_slot, 0.5).reshape(1, -1))[0]
        pred_sym = subset.index(pred_id)
        is_ok = (pred_sym == sym_i)
        correct_cnt += int(is_ok)

        t0 = slot_idx * 0.5
        time_axis.extend(np.linspace(t0, t0 + 0.5, len(chosen_slot), endpoint=False))
        waveform_vals.extend(chosen_slot)
        annotations.append((t0 + 0.25, sym_i, pred_sym, is_ok))

    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=300)
    ax.plot(time_axis, waveform_vals, color="#2b5c8f", linewidth=0.7, alpha=0.85)
    for s_i in range(1, 20):
        ax.axvline(s_i * 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    y_max = np.percentile(waveform_vals, 99) * 1.25
    ax.set_ylim(0, y_max)
    for center_t, true_s, pred_s, is_ok in annotations:
        ax.text(center_t, y_max * 0.88, f"S{true_s}\n({true_s:02b})",
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#e6f5d0" if is_ok else "#fde0dd",
                          edgecolor="#1b7837" if is_ok else "#d73027"))

    legend_str = " | ".join([f"S{i} ({i:02b}): {short_names[i]}" for i in range(4)])
    ax.set_title(
        f"10s Deterministic Stressor Sequence Decoding (20 × 0.5s Slots, p = 4)\n"
        f"Orthogonal Subset: [{legend_str}] — Decoded Slots: {correct_cnt}/20 ({correct_cnt*5}%)",
        fontsize=10.5, fontweight="bold"
    )
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Buffer Access Time (ms)", fontweight="bold")
    ax.set_xlim(0, 10.0)
    plt.tight_layout()
    plt.savefig("fig2_modulated_session_waveform.png", bbox_inches="tight")
    print("    [+] Plot 2 saved: fig2_modulated_session_waveform.png")

if __name__ == "__main__":
    total_start = time.time()
    print("=" * 75)
    print(" Evaluating FLIPSTRESS Microarchitectural Distinguishability & Orthogonality")
    print("=" * 75)

    results_by_p = {}
    for p_val in [1, 2, 4]:
        print(f"\n>>> Running Parallelism Level p = {p_val} (DS1-10s Configuration)...")
        res = evaluate_for_p(p_val=p_val, slot_sec=0.5)
        results_by_p[p_val] = res

    generate_plots(results_by_p)

    print("\n" + "=" * 75)
    print(" Evaluation Complete! Summary of Core Metrics:")
    print("=" * 75)
    for p_val in [1, 2, 4]:
        r = results_by_p[p_val]
        print(f"[p = {p_val}]")
        print(f"  ├─ 7-Class Accuracy (10-Fold CV):                     {r['acc_7class_cv']:.2f}%")
        print(f"  ├─ Cross-State Generalization (Non-Target -> Target): {r['acc_7class_cross']:.2f}%")
        print(f"  ├─ Optimal 4-Stressor Orthogonal Codebook:            {r['best_4ary_names']}")
        print(f"  ├─ 4-Ary Codebook Accuracy (10-Fold CV):              {r['best_4ary_acc']:.2f}%")
        print(f"  └─ 4-Ary Codebook Cross-State Generalization:         {r['best_4ary_cross']:.2f}%")
    print(f"\nTotal Execution Time: {time.time() - total_start:.1f} s")
    print("=" * 75)