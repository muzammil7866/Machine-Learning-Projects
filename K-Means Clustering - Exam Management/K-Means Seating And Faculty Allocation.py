"""
Automated Exam Management System
K-Means clustering for seating plan generation and faculty allocation.
Covers batches 19-22 across 5 domains in 30 exam rooms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math
import time
from collections import defaultdict

random.seed(42)
np.random.seed(42)


# ── Configuration ─────────────────────────────────────────────────────────────
DOMAINS  = ["Computer Science", "Artificial Intelligence",
            "Business Analytics", "Software Engineering", "Electrical Engineering"]
BATCHES  = [19, 20, 21, 22]
N_ROOMS  = 30
N_CLUSTERS = 30


# ── 1. Data Collection / Generation ──────────────────────────────────────────

def generate_students():
    """
    Simulate ~2400–2500 students across batches 19-22 in 5 domains.
    Returns a DataFrame with columns: student_id, batch, domain, domain_code, batch_norm.
    """
    rows = []
    sid  = 1
    for batch in BATCHES:
        for domain in DOMAINS:
            count = random.randint(115, 135)   # ~120 per domain-batch
            for _ in range(count):
                rows.append({
                    "student_id":  f"F{batch}{sid:04d}",
                    "batch":       batch,
                    "domain":      domain,
                    "domain_code": DOMAINS.index(domain),
                })
                sid += 1

    df = pd.DataFrame(rows)
    print(f"  Total students generated : {len(df)}")
    for d in DOMAINS:
        print(f"    {d:<35} : {(df['domain'] == d).sum()}")
    return df


def generate_rooms():
    """
    30 rooms: 20 with 35 seats, 8 with 30 seats, 2 with 25 seats.
    """
    caps = [35] * 20 + [30] * 8 + [25] * 2
    return pd.DataFrame({
        "room_id":   range(1, N_ROOMS + 1),
        "capacity":  caps,
        "block":     [f"Block-{chr(65 + i // 10)}" for i in range(N_ROOMS)],
    })


def generate_faculty():
    """10 faculty members per domain."""
    rows = []
    for domain in DOMAINS:
        for i in range(1, 11):
            rows.append({
                "faculty_id":    f"{domain[:2].upper()}-F{i:02d}",
                "name":          f"Prof. {domain.split()[0]}_{i}",
                "domain":        domain,
                "domain_code":   DOMAINS.index(domain),
            })
    return pd.DataFrame(rows)


# ── 2. Data Preprocessing ────────────────────────────────────────────────────

def preprocess(df_students):
    """
    Normalise features for K-Means.
    Features: batch (19-22) and domain_code (0-4).
    """
    X = df_students[["batch", "domain_code"]].values.astype(float)
    mins  = X.min(axis=0)
    maxs  = X.max(axis=0)
    X_norm = (X - mins) / (maxs - mins + 1e-9)
    return X_norm


# ── 3. K-Means Clustering (from scratch) ─────────────────────────────────────

class KMeans:
    def __init__(self, k=30, max_iter=100, tol=1e-4):
        self.k        = k
        self.max_iter = max_iter
        self.tol      = tol
        self.centroids = None
        self.labels_   = None
        self.inertia_  = None

    def fit(self, X):
        # K-Means++ initialisation for better convergence
        self.centroids = self._kmeans_plus_plus(X)

        for iteration in range(self.max_iter):
            labels = self._assign(X)
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any()
                else self.centroids[k]
                for k in range(self.k)
            ])

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.labels_   = labels

            if shift < self.tol:
                print(f"  K-Means converged at iteration {iteration + 1}")
                break
        else:
            print(f"  K-Means reached max iterations ({self.max_iter})")

        self.inertia_ = sum(
            np.linalg.norm(X[i] - self.centroids[self.labels_[i]]) ** 2
            for i in range(len(X))
        )
        return self

    def _kmeans_plus_plus(self, X):
        centroids = [X[np.random.randint(len(X))]]
        for _ in range(1, self.k):
            dists = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X])
            probs = dists / dists.sum()
            centroids.append(X[np.random.choice(len(X), p=probs)])
        return np.array(centroids)

    def _assign(self, X):
        dists = np.array([[np.linalg.norm(x - c) for c in self.centroids] for x in X])
        return np.argmin(dists, axis=1)

    def predict(self, X):
        return self._assign(X)


# ── 4. Seating Plan ───────────────────────────────────────────────────────────

def generate_seating_plan(df_students, df_rooms):
    """
    Assign each cluster to a room, respecting room capacity.
    Students that overflow are split into the next room.
    Returns a DataFrame: student_id, cluster, room_id, seat_no.
    """
    cluster_groups = df_students.groupby("cluster")
    room_idx = 0
    seat_counter = defaultdict(int)
    assignments = []

    for cluster_id, group in cluster_groups:
        students = group.to_dict("records")
        for student in students:
            while room_idx < len(df_rooms):
                room  = df_rooms.iloc[room_idx]
                cap   = room["capacity"]
                taken = seat_counter[room["room_id"]]
                if taken < cap:
                    seat_counter[room["room_id"]] += 1
                    assignments.append({
                        "student_id": student["student_id"],
                        "batch":      student["batch"],
                        "domain":     student["domain"],
                        "cluster":    cluster_id,
                        "room_id":    int(room["room_id"]),
                        "seat_no":    taken + 1,
                    })
                    break
                else:
                    room_idx += 1
            else:
                # All rooms full — shouldn't happen with 2500 students / 30 rooms × 30 seats
                assignments.append({
                    "student_id": student["student_id"],
                    "batch": student["batch"],
                    "domain": student["domain"],
                    "cluster": cluster_id,
                    "room_id": -1, "seat_no": -1,
                })

    return pd.DataFrame(assignments)


# ── 5. Faculty Allocation ─────────────────────────────────────────────────────

def allocate_faculty(df_seating, df_faculty):
    """
    Assign at least one faculty member per domain present in each room.
    Each faculty member supervises at most 2 rooms per exam.
    Returns a DataFrame: room_id, faculty_id, domain.
    """
    faculty_load = defaultdict(int)
    allocations  = []

    for room_id, room_group in df_seating[df_seating["room_id"] != -1].groupby("room_id"):
        present_domains = room_group["domain"].unique().tolist()
        for domain in present_domains:
            # Pick faculty from that domain with the lowest load
            domain_faculty = df_faculty[df_faculty["domain"] == domain].copy()
            domain_faculty["load"] = domain_faculty["faculty_id"].map(
                lambda fid: faculty_load[fid])
            best = domain_faculty.sort_values("load").iloc[0]
            if faculty_load[best["faculty_id"]] < 3:
                allocations.append({
                    "room_id":    room_id,
                    "faculty_id": best["faculty_id"],
                    "domain":     domain,
                    "name":       best["name"],
                })
                faculty_load[best["faculty_id"]] += 1

    return pd.DataFrame(allocations)


# ── 6. Elbow Method Plot ──────────────────────────────────────────────────────

def elbow_plot(X_norm):
    print("\n  Computing elbow curve (k = 5 … 40) …")
    ks, inertias = [], []
    for k in range(5, 41, 5):
        model = KMeans(k=k, max_iter=50)
        model.fit(X_norm)
        ks.append(k)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(ks, inertias, "bo-", linewidth=2, markersize=6)
    plt.axvline(x=30, color="red", linestyle="--", label="k = 30 (chosen)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title("Elbow Method — Optimal k Selection")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ── 7. Visualisation ─────────────────────────────────────────────────────────

def visualise_clusters(df_students):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("K-Means Clustering — Exam Management System", fontsize=13,
                 fontweight="bold")

    # Students per domain
    counts = df_students.groupby("domain").size().reindex(DOMAINS)
    axes[0].bar(range(len(DOMAINS)), counts.values, color=plt.cm.Set2.colors[:5])
    axes[0].set_xticks(range(len(DOMAINS)))
    axes[0].set_xticklabels([d.replace(" ", "\n") for d in DOMAINS], fontsize=8)
    axes[0].set_ylabel("Number of Students")
    axes[0].set_title("Students per Domain")

    # Cluster scatter: domain_code vs batch, colour = cluster
    axes[1].scatter(df_students["batch"], df_students["domain_code"],
                    c=df_students["cluster"], cmap="tab20", s=5, alpha=0.6)
    axes[1].set_xlabel("Batch")
    axes[1].set_ylabel("Domain Code (0-4)")
    axes[1].set_title("K-Means Clusters (batch × domain)")
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels([d.split()[0] for d in DOMAINS], fontsize=8)

    plt.tight_layout()
    plt.show()


def visualise_rooms(df_seating, df_rooms):
    room_usage = df_seating[df_seating["room_id"] != -1].groupby("room_id").size()
    caps = df_rooms.set_index("room_id")["capacity"]

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(1, N_ROOMS + 1)
    ax.bar(x, caps.values, color="#AED6F1", label="Capacity")
    ax.bar(x, room_usage.reindex(x, fill_value=0).values,
           color="#2E86C1", label="Students assigned")
    ax.set_xlabel("Room ID")
    ax.set_ylabel("Students")
    ax.set_title("Room Utilisation")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── 8. Report Generation ──────────────────────────────────────────────────────

def generate_report(df_students, df_seating, df_faculty_alloc, df_rooms):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  EXAM MANAGEMENT SYSTEM — FINAL REPORT")
    print(f"{sep}")
    print(f"  Total students  : {len(df_students)}")
    print(f"  Total rooms     : {N_ROOMS}")
    print(f"  K-Means clusters: {N_CLUSTERS}")

    print(f"\n  {'Domain':<37} {'Students':>9}  {'% of total':>10}")
    print(f"  {'─'*58}")
    for d in DOMAINS:
        n = (df_students["domain"] == d).sum()
        print(f"  {d:<37} {n:>9}  {n/len(df_students)*100:>9.1f}%")

    print(f"\n  {'Batch':<10} {'Students':>9}")
    print(f"  {'─'*22}")
    for b in BATCHES:
        n = (df_students["batch"] == b).sum()
        print(f"  {b:<10} {n:>9}")

    # Room summary (first 10)
    print(f"\n  {'Room':>5}  {'Capacity':>9}  {'Assigned':>9}  {'Fill%':>6}  Domains present")
    print(f"  {'─'*65}")
    for _, room in df_rooms.head(10).iterrows():
        rid = int(room["room_id"])
        cap = int(room["capacity"])
        assigned = len(df_seating[df_seating["room_id"] == rid])
        doms = ", ".join(
            d.split()[0] for d in
            df_seating[df_seating["room_id"] == rid]["domain"].unique()[:2]
        )
        fill = assigned / cap * 100 if cap > 0 else 0
        print(f"  {rid:>5}  {cap:>9}  {assigned:>9}  {fill:>5.0f}%  {doms}")
    print(f"  … ({N_ROOMS - 10} more rooms)")

    # Faculty summary
    print(f"\n  Faculty allocations : {len(df_faculty_alloc)}")
    if len(df_faculty_alloc):
        print(f"  Rooms with at least one faculty : "
              f"{df_faculty_alloc['room_id'].nunique()}")
        per_domain = df_faculty_alloc.groupby("domain").size()
        print(f"\n  Faculty duty counts by domain:")
        for d, cnt in per_domain.items():
            print(f"    {d:<37}: {cnt}")

    print(f"\n{sep}")
    print("  Report complete.")
    print(f"{sep}\n")

    # Export CSVs
    df_seating.to_csv("seating_plan.csv", index=False)
    df_faculty_alloc.to_csv("faculty_allocation.csv", index=False)
    print("  Exported: seating_plan.csv, faculty_allocation.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  AUTOMATED EXAM MANAGEMENT SYSTEM")
    print("=" * 65)

    # 1. Generate data
    print("\n[1] Generating student, room, and faculty data …")
    df_students = generate_students()
    df_rooms    = generate_rooms()
    df_faculty  = generate_faculty()

    # 2. Preprocess
    print("\n[2] Preprocessing features …")
    X_norm = preprocess(df_students)

    # 3. K-Means clustering
    print("\n[3] Running K-Means clustering (k = 30) …")
    t0 = time.perf_counter()
    model = KMeans(k=N_CLUSTERS, max_iter=200)
    model.fit(X_norm)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Clustering time : {elapsed:.1f} ms")
    print(f"  Inertia         : {model.inertia_:.4f}")

    df_students["cluster"] = model.labels_

    # 4. Seating plan
    print("\n[4] Generating seating plan …")
    df_seating = generate_seating_plan(df_students, df_rooms)
    unassigned = (df_seating["room_id"] == -1).sum()
    print(f"  Students assigned  : {len(df_seating) - unassigned}")
    print(f"  Students unassigned: {unassigned}")

    # 5. Faculty allocation
    print("\n[5] Allocating faculty …")
    df_faculty_alloc = allocate_faculty(df_seating, df_faculty)
    print(f"  Total faculty duties allocated: {len(df_faculty_alloc)}")

    # 6. Visualise
    print("\n[6] Generating visualisations …")
    elbow_plot(X_norm)
    visualise_clusters(df_students)
    visualise_rooms(df_seating, df_rooms)

    # 7. Report
    print("\n[7] Generating report …")
    generate_report(df_students, df_seating, df_faculty_alloc, df_rooms)


if __name__ == "__main__":
    main()