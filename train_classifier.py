import argparse
import pickle
from dataclasses import asdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from hand_feature_extractor import FeatureConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classifier on hand landmark features.")
    parser.add_argument("--data", default="data.pickle", help="Input dataset pickle")
    parser.add_argument("--out", default="model.p", help="Output model pickle")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--class-weight", default=None, choices=["balanced", "balanced_subsample"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dict = pickle.load(open(args.data, "rb"))

    data = data_dict.get("data", [])
    labels = data_dict.get("labels", [])
    feature_config_data = data_dict.get("feature_config")

    if feature_config_data:
        config = FeatureConfig(**feature_config_data)
        feature_dim = config.feature_dim
    else:
        feature_dim = len(data[0]) if data else 0
        include_z = feature_dim == FeatureConfig(include_z=True).feature_dim
        config = FeatureConfig(normalize=False, include_z=include_z)

    filtered_data = []
    filtered_labels = []
    for features, label in zip(data, labels):
        if len(features) == feature_dim:
            filtered_data.append(features)
            filtered_labels.append(label)

    if not filtered_data:
        raise SystemExit("No valid samples available after filtering. Rebuild the dataset.")

    X = np.asarray(filtered_data, dtype=np.float32)
    y = np.asarray(filtered_labels)

    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(args.test_size),
        shuffle=True,
        stratify=stratify,
        random_state=int(args.random_state),
    )

    model = RandomForestClassifier(
        n_estimators=int(args.n_estimators),
        max_depth=args.max_depth,
        n_jobs=int(args.n_jobs),
        random_state=int(args.random_state),
        class_weight=args.class_weight or None,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{score * 100:.2f}% of samples were classified correctly.")

    with open(args.out, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_config": asdict(config),
                "class_names": sorted(set(y)),
            },
            f,
        )


if __name__ == "__main__":
    main()
