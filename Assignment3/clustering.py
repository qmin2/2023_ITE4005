import pandas as pd
import numpy as np
import argparse
import os


def get_neighbors(data, point_index, eps):
    distances = np.sqrt(np.sum((data - data[point_index]) ** 2, axis=1))
    return np.where(distances <= eps)[0]


def expand_cluster(db, labels, point_index, neighbors, cluster_id, eps, min_samples):
    labels[point_index] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_index = neighbors[i]
        if labels[neighbor_index] == -2:
            labels[neighbor_index] = cluster_id
        if labels[neighbor_index] == -1:
            labels[neighbor_index] = cluster_id
            neighbor_neighbors = get_neighbors(db, neighbor_index, eps)
            if len(neighbor_neighbors) >= min_samples:
                neighbors = np.concatenate((neighbors, neighbor_neighbors))
        i += 1


def dbscan(db, eps, min_samples, num_of_clusters):
    n = db.shape[0]
    labels = np.full(n, -1)  # -1: Undefined, -2: Noise, from 0 integers: Cluster IDs
    cluster_id = -1
    for i in range(n):  # i = id
        if labels[i] != -1:
            continue
        neighbors = get_neighbors(db, i, eps)
        if len(neighbors) < min_samples:  # check if this is noise sample or not
            labels[i] = -2  # Noise point
        else:
            cluster_id += 1
            if cluster_id >= num_of_clusters:
                break
            expand_cluster(db, labels, i, neighbors, cluster_id, eps, min_samples)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_name", default="input2.txt", type=str)
    parser.add_argument("--n", default=5, type=int)
    parser.add_argument("--Eps", default=2, type=int)
    parser.add_argument("--MinPts", default=7, type=int)

    args = parser.parse_args()

    # Read the data from the file
    data = pd.read_csv(f"{args.input_file_name}", delimiter="\t", header=None)

    data.columns = ["id", "x", "y"]
    coordinates = data[["x", "y"]].values

    num_of_clusters = args.n  # Maximum number of clusters
    eps = args.Eps
    min_samples = args.MinPts  # Minimum number of points in a cluster

    labels = dbscan(coordinates, eps, min_samples, num_of_clusters)
    data["cluster"] = labels

    base_name = os.path.splitext(args.input_file_name)[0]
    for cluster_id in np.unique(labels):
        if cluster_id not in [-1, -2]:  # Exclude noise points
            cluster_data = data[data["cluster"] == cluster_id]["id"]
            file_name = f"{base_name}_cluster_{int(cluster_id)}.txt"
            cluster_data.to_csv(file_name, sep="\t", index=False, header=False)

    print("Clustering results saved to files.")


if __name__ == "__main__":
    main()
