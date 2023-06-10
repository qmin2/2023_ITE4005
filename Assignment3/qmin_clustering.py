import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def range_query(database, point, epsilon, distance_func):
    distances = distance_func(database, point[np.newaxis, :])
    return np.where(distances <= epsilon)[0]


def dbscan(database, epsilon, min_pts, distance_func):
    num_points = len(database)
    labels = np.full(num_points, -1)  # Initialize labels as undefined

    def expand_cluster(point_index, cluster_label):
        labels[point_index] = cluster_label
        seed_set = range_query(
            database,
            database.iloc[point_index][["x", "y"]].values,
            epsilon,
            distance_func,
        )

        i = 0
        while i < len(seed_set):
            q = seed_set[i]
            if labels[q] == -1:  # Unvisited point
                labels[q] = cluster_label

                q_neighbors = range_query(
                    database,
                    database.iloc[q][["x", "y"]].values,
                    epsilon,
                    distance_func,
                )
                if len(q_neighbors) >= min_pts:
                    seed_set = np.concatenate((seed_set, q_neighbors))

            i += 1

    cluster_label = 0
    for i in range(num_points):
        if labels[i] != -1:
            continue  # Skip processed points

        neighbors = range_query(
            database, database.iloc[i][["x", "y"]].values, epsilon, distance_func
        )
        if len(neighbors) < min_pts:  # Noise point
            labels[i] = -2  # Assign noise label
            continue

        cluster_label += 1
        expand_cluster(i, cluster_label)

    return labels


# Read the database from input.txt using pandas
database = pd.read_csv("input_data/input2.txt", delimiter="\t", names=["id", "x", "y"])

# Example usage:
labels = dbscan(
    database,
    epsilon=2,
    min_pts=7,
    distance_func=lambda a, b: cdist(a, b, metric="euclidean"),
)

print(labels)
