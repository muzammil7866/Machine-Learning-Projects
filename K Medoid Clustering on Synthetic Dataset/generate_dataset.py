import csv
import random
from pathlib import Path


SEGMENTS = [
    {
        "name": "Young-High-Spenders",
        "age": (20, 6),
        "income": (35, 10),
        "score": (78, 9),
        "n": 45,
    },
    {
        "name": "MidAge-Moderate",
        "age": (35, 7),
        "income": (60, 12),
        "score": (52, 10),
        "n": 45,
    },
    {
        "name": "Senior-Low-Spenders",
        "age": (52, 6),
        "income": (45, 11),
        "score": (28, 8),
        "n": 35,
    },
    {
        "name": "Affluent-Selective",
        "age": (42, 8),
        "income": (95, 14),
        "score": (62, 12),
        "n": 35,
    },
]


def bounded_int(mu, sigma, lo, hi):
    return max(lo, min(hi, int(round(random.gauss(mu, sigma)))))


def build_rows(seed=42):
    random.seed(seed)
    rows = []

    for seg in SEGMENTS:
        for _ in range(seg["n"]):
            gender = "Female" if random.random() < 0.52 else "Male"
            age = bounded_int(seg["age"][0], seg["age"][1], 18, 70)
            income = bounded_int(seg["income"][0], seg["income"][1], 15, 150)
            score = bounded_int(seg["score"][0], seg["score"][1], 1, 100)
            rows.append([gender, age, income, score])

    random.shuffle(rows)
    return rows


def main():
    output_path = Path(__file__).resolve().parent / "customer.txt"
    rows = build_rows(seed=42)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
        )
        writer.writerows(rows)

    print(f"Wrote {output_path.name} with {len(rows)} records")


if __name__ == "__main__":
    main()
