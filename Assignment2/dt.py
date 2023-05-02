import pandas as pd
from collections import Counter
from math import log
import argparse


def entropy(y):
    freq = Counter(y)
    prob = [f / len(y) for f in freq.values()]
    return -sum(p * log(p, 2) for p in prob)


def gain_ratio(X, y, column):
    e_parent = entropy(y)
    e_children = 0
    split_info = 0
    values = X[column].unique()

    for value in values:
        subset = y[X[column] == value]
        p = len(subset) / len(y)
        e_children += p * entropy(subset)
        split_info -= p * log(p, 2) if p > 0 else 0

    gain = e_parent - e_children
    return gain / split_info if split_info > 0 else 0


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value  # 분류 기준
        self.children = children or {}

    def is_leaf(self):
        return not bool(self.children)  # {}면 true


def c45(X, y, features):
    if len(y.unique()) == 1:
        return TreeNode(y.iloc[0])

    if not features:
        return TreeNode(y.value_counts().idxmax())

    ratios = {f: gain_ratio(X, y, f) for f in features}
    best_feature = max(ratios, key=ratios.get)
    remaining_features = features - {best_feature}

    node = TreeNode(best_feature)
    for value in X[best_feature].unique():
        subset_X = X[X[best_feature] == value]
        subset_y = y[X[best_feature] == value]
        if not subset_X.empty:
            node.children[value] = c45(subset_X, subset_y, remaining_features)

    return node


def predict(tree, x, most_common_label):
    node = tree
    while not node.is_leaf():
        feature_value = x[node.value]
        node = node.children.get(feature_value)
        if node is None:
            return most_common_label
    return node.value


def train_decision_tree(train_df) -> TreeNode:
    features = set(train_df.columns[:-1].tolist())
    X = train_df.iloc[:, :-1]
    y = train_df.iloc[:, -1]
    return c45(X, y, features)


def test_decision_tree(
    tree: TreeNode, test_df, most_common_label, result_filename: str
):
    test_df["predicted_label"] = test_df.apply(
        lambda x: predict(tree, x, most_common_label), axis=1
    )
    test_df.to_csv(result_filename, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--output_file", type=str)

    args = parser.parse_args()

    with open(args.train_file, "r") as f:
        data = [line.strip().split("\t") for line in f.readlines()]
    train_df = pd.DataFrame(data[1:], columns=data[0])
    most_common_label = train_df.iloc[:, -1].value_counts().idxmax()

    with open(args.test_file, "r") as f:
        data = [line.strip().split("\t") for line in f.readlines()]
    test_df = pd.DataFrame(data[1:], columns=data[0])

    # Train the decision tree
    tree = train_decision_tree(train_df)

    # Test the decision tree
    test_decision_tree(tree, test_df, most_common_label, args.output_file)


if __name__ == "__main__":
    main()
