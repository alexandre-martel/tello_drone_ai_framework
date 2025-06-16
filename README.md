# Gesture-Controlled Drone Interface

This project is a framework to control a drone (Tello edu) using gestures captured, face following, via the drone camera, using MediaPipe and a custom gesture classifier. The main launch script `run.sh` allows you to:

- Train a gesture recognition model with your personnal signs
- Test the model with a live camera
- Evaluate the model on CSV data
- Collect new gestures
- Print gesture data
- Control a drone by keyboard

Moreover, you can add new functions on the "drone_keyboard" using your trained signes 

---

## Prerequistes

- Python 3.8+
- Libraries: `mediapipe`, `torch`, `numpy`, etc. (Listed in the requirements.txt)
- DJI Tello edu
- Virtual environment recommended (.venv)

---

## Use

### Lunch interface :

```bash
bash run.sh
```

Without arguments, it shows the help menu:


- choice 1 : Keyboard control of the drone
- choice 2 : Train the gesture recognition model
- choice 3 : Test the model with the camera (live test)
- choice 4 : Evaluate the model on CSV data
- choice 5 : Collect gestures from the camera
- choice 6 : Print saved gesture CSV
- choice 7 : Quit

### Example Usages

To train a model:

```bash
bash run.sh --choice 2 --data_filename gestures.csv --save_filename gesture_bundle.pt --epochs 100 --lr 0.0005
```

```bash
bash run.sh --choice 3 --load_filename gesture_bundle.pt --threshold 0.98 --cam 0
```

```bash
bash run.sh --choice 5 --collect_filename gestures.csv --cam 0
```

### Script structure

- run.sh: main launcher script

- gesture_recog_mediapipe/run.py: contains the classifier and logic for training/testing

- drone_keyboard.py: keyboard-based or gesture-based drone control

### Add your own gestures

1. Collect examples readable labels :

```bash
bash run.sh --choice 5 --collect_filename gestures.csv
```
2. Retrain the model:

```bash
bash run.sh --choice 2 --data_filename gestures.csv
```

3. Add label-to-command logic in drone_keyboard.py to map labels to drone actions with that format
   
```python
def safe_move_forward(self):
    """
    Move forward safely if the drone is flying.
    """
    try:
        if self.tello.is_flying:
            self.tello.move_forward(self.speed)
        else:
            print("Drone not flying, cannot move forward.")
    except Exception as e:
        print(f"Error moving forward: {e}")
```

### Existing gesture_to_command functions

```python
safe_move_takeoff()
safe_move_land()
safe_move_left()
safe_move_right()
safe_move_up()
safe_move_down()
```

---

## Author

Project carried out during a semester at Yamade Lab, Tohoku University, by Alexandre MARTEL
Feel free to contribute or adapt the code for other use cases...
