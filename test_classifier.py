import argparse
import pickle

import cv2
import mediapipe as mp
import numpy as np

from hand_feature_extractor import FeatureConfig, extract_hand_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime hand sign classifier.")
    parser.add_argument("--model", default="model.p", help="Trained model pickle")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.3)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dict = pickle.load(open(args.model, "rb"))
    model = model_dict["model"]
    config_data = model_dict.get("feature_config")
    if config_data:
        config = FeatureConfig(**config_data)
    else:
        config = FeatureConfig(normalize=False, include_z=False)

    cap = cv2.VideoCapture(int(args.camera_index))

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
    )

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Check camera connection.")
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            features = extract_hand_features(hand_landmarks, config)
            if features is not None:
                data_aux = features
                prediction = model.predict([np.asarray(data_aux)])[0]
                predicted_character = str(prediction)

                x1 = max(int(min(x_) * W) - 10, 0)
                y1 = max(int(min(y_) * H) - 10, 0)
                x2 = min(int(max(x_) * W) + 10, W)
                y2 = min(int(max(y_) * H) + 10, H)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    frame,
                    predicted_character,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
