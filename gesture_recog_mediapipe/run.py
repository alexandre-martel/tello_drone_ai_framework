from gestureClassifierClass import GestureClassifier
import argparse

def main():
    parser = argparse.ArgumentParser(description="Gesture Recognition Trainer/Tester")

    parser.add_argument("--mode", type=str,
    choices=["train", "model_test", "cam_with_model_test", "collect_data", "print_data"],
    default="train",
    help="Mode of operation: 'train' to train/save, 'test' to load/test, 'cam_test' to run camera test, 'collect' to collect hand data, 'print_data' to print collected data"
    )
    parser.add_argument("--threshold", type=float, default=0.99, help="Threshold for gesture recognition confidence (only used in 'cam_test' and 'collect' modes)")
    parser.add_argument("--cam_id", type=int, default=0, help="Camera device ID (only used if mode is 'cam_test')")
    parser.add_argument("--data_filename", type=str, default="gestures.csv", help="Path to CSV data")
    parser.add_argument("--collect_filename", type=str, default="gestures.csv", help="Path to save collected gestures data")
    parser.add_argument("--load_filename", type=str, default="gesture_bundle.pt", help="Path to load model")
    parser.add_argument("--save_filename", type=str, default="gesture_bundle.pt", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--print_every", type=int, default=10, help="Print loss every N epochs")

    args = parser.parse_args()

    clf = GestureClassifier()

    if args.mode == "train":
        clf.load_data(args.data_filename)
        clf.sequential_model()
        clf.train(epochs=args.epochs, lr=args.learning_rate, print_every=args.print_every)
        clf.save(args.save_filename)
    elif args.mode == "cam_with_model_test":
        clf.load(args.load_filename)
        clf.recognize_from_cam(seuil=args.threshold, temperature=2.0, camera_id=args.cam_id)
    elif args.mode == "model_test":
        clf.load_data(args.data_filename)
        clf.load(args.load_filename)
        clf.evaluate()
    elif args.mode == "collect_data":
        clf.collect_hand_data(
            collect_path=args.collect_filename,
            camera_id=args.cam_id,
            seuil=args.threshold
        )
        clf.print_hand_data()
    elif args.mode == "print_data":
        clf.print_hand_data(args.collect_filename)
    else:
        raise ValueError("Invalid mode. Choose in [train, model_test, cam_with_model_test, collect_data, print_data].")

if __name__ == "__main__":
    main()