#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Defaults
CHOICE=""
DATA_FILENAME="gestures.csv"
COLLECT_FILENAME="gestures.csv"
SAVE_MODEL_FILENAME="gesture_bundle.pt"
LOAD_MODEL_FILENAME="gesture_bundle.pt"
CAM_ID=0
EPOCHS=40
LR=0.001
THRESHOLD=0.99
PRINT_EVERY=10
GESTURE_SCRIPT="gesture_recog_mediapipe/run.py"
FRONTEND_SCRIPT="drone_keyboard.py"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --choice) CHOICE="$2"; shift ;;
        --data_filename) DATA_FILENAME="$2"; shift ;;
        --collect_filename) COLLECT_FILENAME="$2"; shift ;;
        --save_filename) SAVE_MODEL_FILENAME="$2"; shift ;;
        --load_filename) LOAD_MODEL_FILENAME="$2"; shift ;;
        --cam) CAM_ID="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --threshold) THRESHOLD="$2"; shift ;;
        --print_every) PRINT_EVERY="$2"; shift ;;
        *) echo -e "${RED}Unknown parameter: $1${NC}"; exit 1 ;;
    esac
    shift
done

# Show help if no choice
if [ -z "$CHOICE" ]; then
    echo -e "${GREEN}=== Available choices ===${NC}"
    echo "  --choice 1 : Keyboard control of the drone"
    echo "  --choice 2 : Train the gesture recognition model"
    echo "  --choice 3 : Test the model with the camera (live test)"
    echo "  --choice 4 : Evaluate the model on CSV data"
    echo "  --choice 5 : Collect gestures from the camera"
    echo "  --choice 6 : Print saved gesture CSV"
    echo "  --choice 7 : Quit"
    echo ""
    echo "Example:"
    echo "./gesture_launcher.sh --choice 2 --data_filename mydata.csv --save_filename model.pt --epochs 100 --lr 0.0005"
    exit 0
fi

# Execute selected choice
case $CHOICE in
    1)
        echo -e "${GREEN}Starting drone control...${NC}"
        python "$FRONTEND_SCRIPT"
        ;;
    2)
        echo -e "${GREEN}Training the model...${NC}"
        python "$GESTURE_SCRIPT" \
            --mode train \
            --data_filename "$DATA_FILENAME" \
            --save_filename "$SAVE_MODEL_FILENAME" \
            --epochs "$EPOCHS" \
            --learning_rate "$LR" \
            --print_every "$PRINT_EVERY"
        ;;
    3)
        echo -e "${GREEN}Testing the model with the camera (live)...${NC}"
        python "$GESTURE_SCRIPT" \
            --mode cam_with_model_test \
            --load_filename "$LOAD_MODEL_FILENAME" \
            --threshold "$THRESHOLD" \
            --cam_id "$CAM_ID"
        ;;
    4)
        echo -e "${GREEN}Evaluating the model on CSV data...${NC}"
        python "$GESTURE_SCRIPT" \
            --mode model_test \
            --load_filename "$LOAD_MODEL_FILENAME"
        ;;
    5)
        echo -e "${GREEN}Collecting gestures from camera...${NC}"
        python "$GESTURE_SCRIPT" \
            --mode collect_data \
            --collect_filename "$COLLECT_FILENAME" \
            --threshold "$THRESHOLD" \
            --cam_id "$CAM_ID"
        ;;
    6)
        echo -e "${GREEN}Printing gesture data from CSV...${NC}"
        python "$GESTURE_SCRIPT" \
            --mode print_data \
            --collect_filename "$COLLECT_FILENAME"
        ;;
    7)
        echo -e "${RED}Exit...${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice: $CHOICE${NC}"
        exit 1
        ;;
esac
