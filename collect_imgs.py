import argparse
import os

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hand images for each class from webcam.")
    parser.add_argument("--data-dir", default=".\\data", help="Output dataset directory")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes (used if --classes not set)")
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated class names (e.g. A,B,L). If set, overrides --num-classes.",
    )
    parser.add_argument("--dataset-size", type=int, default=100, help="Images to collect per class")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    if args.classes.strip():
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        classes = [str(i) for i in range(int(args.num_classes))]

    cap = cv2.VideoCapture(int(args.camera_index))

    for class_name in classes:
        class_dir = os.path.join(data_dir, str(class_name))
        os.makedirs(class_dir, exist_ok=True)

        print(f"Collecting data for class '{class_name}'")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Check camera connection.")
                break

            cv2.putText(
                frame,
                'Ready? Press "Q" to start',
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        # Continue numbering if the folder already has images.
        start_idx = 0
        existing = [f for f in os.listdir(class_dir) if f.lower().endswith(".jpg")]
        if existing:
            nums = []
            for f in existing:
                name = os.path.splitext(f)[0]
                if name.isdigit():
                    nums.append(int(name))
            if nums:
                start_idx = max(nums) + 1

        for i in range(int(args.dataset_size)):
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Check camera connection.")
                break

            cv2.imshow("frame", frame)
            cv2.waitKey(25)

            out_path = os.path.join(class_dir, f"{start_idx + i}.jpg")
            cv2.imwrite(out_path, frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
