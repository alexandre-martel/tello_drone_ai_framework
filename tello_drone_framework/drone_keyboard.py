from djitellopy import Tello, TelloException
import cv2
import pygame
import numpy as np
import time
from inference_sdk import InferenceHTTPClient
import logging
logging.getLogger("djitellopy").setLevel(logging.WARNING)
import threading

import torch
import mediapipe as mp
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



class FrontEnd(object):
    """ 
    Maintains the Tello display and moves it through the keyboard keys.
    Press escape key to quit.

    The controls are:
        - T: Takeoff
        - L: Land
        - Arrow keys: Forward, backward, left and right.
        - A and D: Counter clockwise and clockwise rotations (yaw)
        - W and S: Up and down.

    """

    def __init__(self, model_path = "models/gesture_bundle.pt", speed = 30):

        """ 
        Initializes the FrontEnd object.
        """

        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Frames per second of the pygame window display
        # A low number also results in input lag, as input information is processed once per frame.
        FPS = 120

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = speed
        self.send_rc_control = False
        self.tello.is_flying = False 

        # Load bundle
        bundle = torch.load(model_path)

        self.input_dim = bundle["input_dim"]
        self.num_classes = bundle["num_classes"]


        # Multithreading variables
        self.current_action_thread = None
        self.action_counter = [None, 0] # In order to validate a gesture, we need to count the number of consecutive frames with the same gesture.
        self.validation_required = 30 # Number of consecutive frames required to validate a gesture


        # Mediapipe variables
        self.gesture_mode = False  
        self.client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key="",
        )

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)
        
        # Rebuild model
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )
        self.model.load_state_dict(bundle["model_state"])
        self.model.eval()

        # Rebuild label encoder
        self.le = LabelEncoder()
        self.le.classes_ = bundle["label_classes"]
        self.gesture_list = list(self.le.classes_)

        # Rebuild scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = bundle["scaler_mean"]
        self.scaler.scale_ = bundle["scaler_scale"]

        # Gesture recognition config
        self.THRESHOLD = 0.85
        self.temperature = 2.0


        # Initialize MediaPipe Hands for gesture recognition
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils


    def safe_move_takeoff(self):
        """        
        Function for taking off with multithreading support.
        If the drone is already flying, it will not attempt to take off again.
        """
        try:  
            if not(self.tello.is_flying):
                self.tello.takeoff()
                self.tello.is_flying = True
            else:
                print("Drone already flying, no need to take off again.")
        except Exception as e:
            print(f"Error during takeoff: {e}")



    def safe_move_land(self):
        """
        Function for landing with multithreading support.
        If the drone is not flying, it will not attempt to land.
        """
        try:
            if self.tello.is_flying:
                self.tello.move_left(self.speed)
            else:
                print("Drone not flying, can't move_left.")
        except Exception as e:
            print(f"Error during landing: {e}")

    def safe_move_left(self):
        """
        Function for moving left with multithreading support.
        If the drone is not flying, it will not attempt to move.
        """
        try:
            if self.tello.is_flying:
                self.tello.land()
                self.tello.is_flying = False
            else:
                print("Drone not flying, no need to land.")  
        except Exception as e:
            print(f"Error moving left: {e}")

    def safe_move_right(self):
        """
        Function for moving right with multithreading support.
        If the drone is not flying, it will not attempt to move.
        """
        try:
            if self.tello.is_flying:
                self.tello.move_right(self.speed)
            else:
                print("Drone not flying, cannot move right.")
        except Exception as e:
            print(f"Error moving right: {e}")

    def safe_move_up(self):
        """
        Function for moving up with multithreading support.
        If the drone is not flying, it will not attempt to move.
        """
        try:
            if self.tello.is_flying:
                self.tello.move_up(self.speed)
            else:
                print("Drone not flying, cannot move up.")
        except Exception as e:
            print(f"Error moving up: {e}")


    def safe_move_down(self):
        """
        Function for moving down with multithreading support.
        If the drone is not flying, it will not attempt to move.
        """
        try:
            if self.tello.is_flying:
                self.tello.move_down(self.speed)
            else:
                print("Drone not flying, cannot move down.")
        except Exception as e:
            print(f"Error moving down: {e}")

    def safe_move_stop(self):
        """
        Function to safely stop all drone motion.
        Sends zero velocity commands only if drone is flying.
        """
        try:
            if self.tello.is_flying:
                self.tello.send_rc_control(0, 0, 0, 0)
            else:
                print("Drone not flying, nothing to stop.")
        except Exception as e:
            print(f"Error stopping drone: {e}")

    #-------------------------------------------


    # ADD NEW FONCTIONS HERE

    
    #-------------------------------------------


    def run_action_thread(self, action_func):
        """
        Function to run an action in a separate thread.
        This ensures that the action does not block the main thread.
        
        Arguments:
            action_func: The function to run in a separate thread.
        """
        if self.current_action_thread is None or not self.current_action_thread.is_alive():
            self.current_action_thread = threading.Thread(target=action_func)
            self.current_action_thread.start()

    def manage_gesture(self, frame):
        """
        Function to manage gesture recognition and execute corresponding actions.

        Arguments:
            frame: The current video frame from the Tello drone.
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks: # Check if any hands are detected
            
            hand = results.multi_hand_landmarks[0]
            data = []
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])

            X_input = np.array(data).reshape(1, -1)
            X_scaled = self.scaler.transform(X_input) # Scale the input data

            with torch.no_grad(): #Predict the gesture
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                output = self.model(X_tensor)
                probs = torch.softmax(output / self.temperature, dim=1)
                conf_value, predicted = torch.max(probs, dim=1)
                conf_value = conf_value.item()
                predicted = predicted.item()

                if conf_value >= self.THRESHOLD:
                    gesture = self.le.inverse_transform([predicted])[0]
                else:
                    gesture = "no_gesture"

                cv2.putText(frame, f"{gesture} ({conf_value:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if (self.current_action_thread is None) or (not self.current_action_thread.is_alive()):
                    
                    if self.action_counter[0] == gesture:
                            self.action_counter[1] += 1
                    else:
                        self.action_counter[0] = gesture
                        self.action_counter[1] = 0

                    if gesture in self.gesture_list:
                        if self.action_counter[1] > self.validation_required:
                            self.action_counter[1] = 0
                            print('Action done : ' + gesture)
                            self.run_action_thread(getattr(self, f'safe_move_{gesture}'))
                    else:
                        if self.action_counter[1] > self.validation_required:
                            self.action_counter[1] = 0
                            print('Stoping all actions')
                            self.run_action_thread(self.safe_move_stop)
                        

                    progress = self.action_counter[1]
                    bar_length = 60  # Progress bar length
                    block = int(round(bar_length * progress / self.validation_required))
                    progress_bar = f"[{'#' * block}{'.' * (bar_length - block)}] {progress}/{self.validation_required}"
                    print(f"Gesture: {gesture} | {progress_bar}", end='\r')



    def run(self):
        """
        Main loop of the FrontEnd object.
        Initializes the Tello drone, starts the video stream, and processes user input.
        """

        # Initialize the Tello drone
        try:  
            self.tello.connect()
            self.tello.send_command_with_return("command")
            self.tello.set_speed(self.speed)

            print(f"Drone connected: {self.tello.get_battery()}% battery")
        except Exception as e:
            print(f"Error connecting to drone: {e}")
            return

        # Start the video stream
        try:
            self.tello.streamoff()
            self.tello.streamon()
        except Exception as e:
            print(f"Error starting video stream: {e}")
            return

        frame_read = self.tello.get_frame_read()
        should_stop = False
        
        # Main loop
        while not should_stop:
            try:  
                for event in pygame.event.get(): # Process PyGame events
                    if event.type == pygame.USEREVENT + 1:
                        self.update()

                    elif event.type == pygame.QUIT: # Quit the program if the cross is clicked
                        should_stop = True

                    elif event.type == pygame.KEYDOWN: # Process key presses
                        if event.key == pygame.K_ESCAPE: 
                            should_stop = True
                        else:
                            self.keydown(event.key)

                    elif event.type == pygame.KEYUP: # Process key releases
                        self.keyup(event.key)

                if frame_read.stopped: 
                    break

                self.screen.fill([0, 0, 0])

                # Read the frame from the Tello drone
                frame = frame_read.frame
                if frame is None:
                    continue

                if self.gesture_mode: # Gesture mode is enabled
                    try:       
                        self.manage_gesture(frame)
                                
                    except Exception as e:
                        print(f"Error in gesture processing: {e}")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                text = "Battery: {}%".format(self.tello.get_battery())
                cv2.putText(frame, text, (5, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
                frame = np.rot90(frame)
                frame = np.flipud(frame)
                frame_surface = pygame.surfarray.make_surface(frame)
                self.screen.blit(frame_surface, (0, 0))
                pygame.display.update()
                

                time.sleep(1 / FPS)

            except Exception as e:
                print(f"Unexpected error in main loop: {e}")

        try:
            self.tello.end()
        except Exception as e:
            print(f"Error ending connection: {e}")



    def keydown(self, key):
        """ 
        Update velocities based on key pressed

        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = self.speed
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -self.speed
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -self.speed
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = self.speed
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = self.speed
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -self.speed
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -self.speed
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = self.speed
        elif key == pygame.K_g:
            self.gesture_mode = not self.gesture_mode
            print(f"Mode geste {'activé' if self.gesture_mode else 'désactivé'}")

    def keyup(self, key):
        """ 
        Update velocities based on key released

        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        try:
            if self.send_rc_control and not self.gesture_mode:  # Avoid RC in gesture mode
                self.tello.send_rc_control(
                    self.left_right_velocity,
                    self.for_back_velocity,
                    self.up_down_velocity,
                    self.yaw_velocity
                )
        except Exception as e:
            print(f"Error sending RC control: {e}")


def main():
    frontend = FrontEnd()

    # run frontend

    frontend.run()


if __name__ == '__main__':
    main()
