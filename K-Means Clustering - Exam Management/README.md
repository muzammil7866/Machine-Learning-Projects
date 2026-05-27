# K-Means Clustering — Exam Management System

## Overview

An automated exam management system for a university with ~2400–2500 students across **batches 19–22** in **5 domains** (CS, AI, BA, SE, EE). Uses **K-Means clustering** (implemented from scratch) to group students, then generates an optimised **seating plan** for 30 exam rooms and allocates **faculty members** based on domain expertise.

## File

| File | Description |
|------|-------------|
| `K-Means Seating And Faculty Allocation.py` | Full pipeline: data generation, K-Means, seating, faculty allocation, report |

## How to Run

```bash
pip install numpy pandas matplotlib
python "K-Means Seating And Faculty Allocation.py"
```

Exports: `seating_plan.csv`, `faculty_allocation.csv`

## System Architecture

```
[1] Data Collection        →  Generate students, rooms, faculty
[2] Data Preprocessing     →  Normalise batch + domain_code features
[3] K-Means Clustering     →  k=30 clusters (one per exam room)
[4] Seating Plan           →  Assign cluster groups to rooms (respect capacity)
[5] Faculty Allocation     →  Assign domain-expert faculty per room
[6] Visualisation          →  Elbow plot, cluster scatter, room utilisation
[7] Report Generation      →  Console report + CSV export
```

## University Setup

| Resource | Details |
|----------|---------|
| Students | ~2400–2500 (batches 19–22, 5 domains) |
| Domains | Computer Science, Artificial Intelligence, Business Analytics, Software Engineering, Electrical Engineering |
| Rooms | 30 total — 20 × 35 seats, 8 × 30 seats, 2 × 25 seats |
| Faculty | 10 per domain (50 total) |

## K-Means Implementation (From Scratch)

```
Initialisation:  K-Means++ (distance-proportional centroid seeding)
Features:        [batch_normalised, domain_code_normalised]
Distance:        Euclidean
Convergence:     Centroid shift < 1e-4  OR  max_iter = 200
```

### K-Means++ Initialisation
Standard random initialisation can lead to poor clusters. K-Means++ picks the first centroid randomly, then selects each subsequent centroid with probability proportional to its squared distance from the nearest existing centroid — ensuring spread-out initial seeds.

### Elbow Method
The system runs K-Means for k = 5, 10, …, 40 and plots inertia to confirm k = 30 is the natural elbow point for this dataset.

## Seating Plan Logic

1. Students in each cluster are treated as a group
2. Groups are filled into rooms in order, respecting room capacity
3. When a room is full, the next room is used (overflow handling)
4. Students from different domains in the same room = mixed invigilated environment

## Faculty Allocation Strategy

- Each room receives at least one faculty member per domain represented in that room
- Faculty load is tracked; no faculty member is assigned more than 3 rooms per exam
- Faculty are selected by minimum current load (greedy fair-allocation)

## Output Files

### `seating_plan.csv`
| student_id | batch | domain | cluster | room_id | seat_no |
|-----------|-------|--------|---------|---------|---------|

### `faculty_allocation.csv`
| room_id | faculty_id | domain | name |
|---------|-----------|--------|------|

## Visualisations

1. **Elbow Curve** — inertia vs k to justify k=30
2. **Cluster Scatter** — batch × domain_code coloured by cluster
3. **Room Utilisation Bar Chart** — capacity vs students assigned

## Dependencies

```
numpy, pandas, matplotlib
```