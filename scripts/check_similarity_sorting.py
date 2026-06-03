import os
import sys
import glob
import numpy as np

sys.path.append(os.path.abspath("src"))

from su_metric import RS, SD, SU, sort_scores_desc


def find_latest_log_dir(base_dir="/content/auto_VPR/logs/amstertime_cosplace_test"):
    runs = sorted(glob.glob(os.path.join(base_dir, "*")))
    if not runs:
        raise FileNotFoundError(f"No runs found under {base_dir}")
    return runs[-1]


def find_descriptor_files(log_dir):
    npy_files = glob.glob(os.path.join(log_dir, "**", "*.npy"), recursive=True)

    query_file = None
    database_file = None

    for f in npy_files:
        name = os.path.basename(f).lower()

        if "quer" in name:
            query_file = f

        if "database" in name or "db" in name:
            database_file = f

    if query_file is None or database_file is None:
        print("Found .npy files:")
        for f in npy_files:
            print(f)
        raise FileNotFoundError(
            "Could not automatically identify query/database descriptor files."
        )

    return query_file, database_file


def main():
    log_dir = find_latest_log_dir()
    print(f"Using log directory: {log_dir}")

    query_file, database_file = find_descriptor_files(log_dir)

    print(f"Query descriptor file: {query_file}")
    print(f"Database descriptor file: {database_file}")

    queries_desc = np.load(query_file)
    database_desc = np.load(database_file)

    print("Query descriptors:", queries_desc.shape)
    print("Database descriptors:", database_desc.shape)

    sim_matrix = queries_desc @ database_desc.T

    print("Similarity matrix:", sim_matrix.shape)
    print("First row first 10 raw scores:")
    print(sim_matrix[0, :10])

    raw_rs = RS(sim_matrix[:1])
    print("RS without sorting:", raw_rs)

    sim_matrix_sorted = sort_scores_desc(sim_matrix)

    print("First row first 10 sorted scores:")
    print(sim_matrix_sorted[0, :10])

    sorted_rs = RS(sim_matrix_sorted[:1])
    print("RS after sorting:", sorted_rs)

    su_scores = SU(sim_matrix_sorted)

    print("SU shape:", su_scores.shape)
    print("First 10 SU scores:", su_scores[:10])
    print("SU mean:", su_scores.mean())
    print("SU std:", su_scores.std())

    print("\nFINAL DECISION:")
    print("Descriptor-derived similarity matrix is NOT sorted by retrieval score.")
    print("Use sort_scores_desc(sim_matrix) before RS, SD, and SU.")


if __name__ == "__main__":
    main()
