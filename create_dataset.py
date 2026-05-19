import argparse
import os
import pickle
from dataclasses import asdict

import cv2
import mediapipe as mp

from hand_feature_extractor import FeatureConfig, extract_hand_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a hand landmark dataset from image folders.")
    parser.add_argument("--data-dir", default=".\\data", help="Dataset root with class subfolders")
    parser.add_argument("--out", default="data.pickle", help="Output pickle file")
    parser.add_argument("--include-z", action="store_true", help="Include z landmark values")
    parser.add_argument("--no-normalize", action="store_true", help="Disable landmark normalization")
    parser.add_argument("--max-hands", type=int, default=1, help="Max hands to detect per image")
    parser.add_argument("--min-detection-confidence", type=float, default=0.3)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N images (0 to disable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = FeatureConfig(normalize=not args.no_normalize, include_z=args.include_z)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=int(args.max_hands),
        min_detection_confidence=float(args.min_detection_confidence),
    )

    data = []
    labels = []
    class_names = []
    skipped_no_hand = 0
    skipped_invalid = 0

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Dataset directory not found: {data_dir}")

    samples = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_names.append(class_name)
        for img_name in os.listdir(class_dir):
            samples.append((class_name, os.path.join(class_dir, img_name)))

    total_samples = len(samples)
    processed = 0

    def print_progress() -> None:
        if args.progress_every <= 0:
            return
        if processed % args.progress_every == 0 or processed == total_samples:
            kept = len(data)
            print(
                f"\rProcessed {processed}/{total_samples} | kept {kept} "
                f"| no-hand {skipped_no_hand} | invalid {skipped_invalid}",
                end="",
                flush=True,
            )

    for class_name, img_path in samples:
        img = cv2.imread(img_path)
        processed += 1
        if img is None:
            skipped_invalid += 1
            print_progress()
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            skipped_no_hand += 1
            print_progress()
            continue

        hand_landmarks = results.multi_hand_landmarks[0]
        features = extract_hand_features(hand_landmarks, config)
        if features is None:
            skipped_invalid += 1
            print_progress()
            continue

        data.append(features)
        labels.append(class_name)
        print_progress()

    if not data:
        raise SystemExit("No valid hand samples were found. Check the dataset and detection settings.")

    if args.progress_every > 0:
        print()

    with open(args.out, "wb") as f:
        pickle.dump(
            {
                "data": data,
                "labels": labels,
                "feature_config": asdict(config),
                "class_names": class_names,
            },
            f,
        )

    print(
        f"Saved {len(data)} samples from {len(class_names)} classes "
        f"(skipped {skipped_no_hand} with no hand, {skipped_invalid} invalid)."
    )


if __name__ == "__main__":
    main()
