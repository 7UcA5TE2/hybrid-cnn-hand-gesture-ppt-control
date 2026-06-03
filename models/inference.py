import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import pyautogui
import pygetwindow as gw
import threading
import tkinter as tk

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# WINDOW FOCUS 
def focus_powerpoint():
    """Bring the main PowerPoint window (editing view) to front."""
    windows = gw.getWindowsWithTitle("PowerPoint")
    for win in windows:
        if "Slide Show" not in win.title:
            try:
                win.activate()
                time.sleep(0.1)
                return win
            except:
                pass
    return None


def focus_slideshow():
    """Bring the PowerPoint slideshow window to front."""
    windows = gw.getWindowsWithTitle("PowerPoint")
    for win in windows:
        if "Slide Show" in win.title:
            try:
                win.activate()
                time.sleep(0.1)
                return win
            except:
                pass
    return None


# LOAD MODEL
model = tf.keras.models.load_model("models/best_model.h5")

temp_gen = ImageDataGenerator().flow_from_directory(
    r"C:\Users\inamm\Desktop\HandGesturepptproject\dataset_split\train",
    target_size=(224, 224),
    batch_size=1
)

class_names = list(temp_gen.class_indices.keys())
print("Class order:", class_names)


#  MEDIAPIPE
base_options = python.BaseOptions(
    model_asset_path=r"C:\Users\inamm\Desktop\HandGesturepptproject\tools\hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


#  CAMERA 
cap = cv2.VideoCapture(0)

MARGIN = 20

previous_label = ""
stable_counter = 0
STABLE_THRESHOLD = 5
CLICK_CONFIDENCE_THRESHOLD = 0.70


# MODE 
current_mode = "IDLE"
mode_locked = False
mode_exit_time = 0
MODE_EXIT_DELAY = 1.0


#  TIMING
last_action_time = 0
last_click_time = 0
ACTION_COOLDOWN = 2
ACTION_DELAY = 3
CLICK_COOLDOWN = 0.8

# SWIPE (NAVIGATION MODE) 
prev_x = None
gesture_active = False
SWIPE_THRESHOLD = 120


# POINTER MODE (absolute positioning with overlay) 
screen_w, screen_h = pyautogui.size()
prev_hand_x, prev_hand_y = None, None
smooth_x, smooth_y = None, None
ALPHA = 0.8                   # smoothing factor for hand coordinates
MARGIN_POINTER = 0.05         # 5% margin from camera edges
INVERT_X = True               # set to True to fix mirrored camera direction

# SCREEN OVERLAY (visible red dot)
class PointerOverlay:
    def __init__(self):
        self.root = None
        self.canvas = None
        self.running = False
        self.thread = None
        self.pos = (0, 0)                     # shared position
        self.lock = threading.Lock()          # for thread safety

    def start(self):
        """Start the overlay in a separate thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """Run tkinter main loop."""
        self.root = tk.Tk()
        self.root.overrideredirect(True)          # no window decorations
        self.root.attributes('-topmost', True)    # always on top
        self.root.attributes('-transparentcolor', 'white')  # make white transparent
        self.root.geometry(f"{screen_w}x{screen_h}+0+0")    # full screen
        self.root.configure(bg='white')           # background will be transparent

        self.canvas = tk.Canvas(self.root, width=screen_w, height=screen_h,
                                highlightthickness=0, bg='white')
        self.canvas.pack()

        # Draw initial dot (will be updated)
        self.dot = self.canvas.create_oval(0, 0, 30, 30, fill='red', outline='red')
        self.root.update()

        # Start the periodic update loop
        self._update_loop()
        self.root.mainloop()

    def _update_loop(self):
        """Called repeatedly via after() to update dot position."""
        if not self.running:
            return
        # Get the latest position safely
        with self.lock:
            x, y = self.pos
        # Update the dot coordinates
        self.canvas.coords(self.dot, x-15, y-15, x+15, y+15)
        # Schedule next update (10 ms)
        self.root.after(10, self._update_loop)

    def update_position(self, x, y):
        """Called from main thread to set new position."""
        if not self.running:
            return
        with self.lock:
            self.pos = (x, y)

    def stop(self):
        """Stop the overlay."""
        self.running = False
        if self.root:
            self.root.quit()
        if self.thread:
            self.thread.join(timeout=1.0)

# Create a global overlay instance
overlay = PointerOverlay()


# main loop start here 
while True:
    ret, frame = cap.read()
    if not ret:
        break


    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x_min = max(0, int(min(xs) * w) - MARGIN)
        y_min = max(0, int(min(ys) * h) - MARGIN)
        x_max = min(w, int(max(xs) * w) + MARGIN)
        y_max = min(h, int(max(ys) * h) + MARGIN)

        roi = frame[y_min:y_max, x_min:x_max]

        # Default display label  (will be updated if ROI is valid)
        display_label = previous_label
        confidence = 0.0

        if roi.size > 0:

            roi = cv2.resize(roi, (224, 224))
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi, verbose=0)

            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            raw_label = class_names[class_index]

            #  STABILITY 
            if raw_label == previous_label:
                stable_counter += 1
            else:
                stable_counter = 1

            if stable_counter >= STABLE_THRESHOLD:
                display_label = raw_label
                current_time = time.time()

                # START / END
                if not mode_locked:
                    if display_label == "like":
                        if current_time - last_action_time > ACTION_DELAY:
                            print("START PPT")
                            win = focus_powerpoint()
                            if win:
                                print(f"Active window: {win.title}")
                            time.sleep(0.3)
                            pyautogui.press('f5')
                            last_action_time = current_time

                    elif display_label == "dislike":
                        if current_time - last_action_time > ACTION_DELAY:
                            print("END PPT")
                            win = focus_slideshow()
                            if win:
                                print(f"Slideshow window: {win.title}")
                            else:
                                print("No slideshow found - focusing editing window")
                                focus_powerpoint()
                            time.sleep(0.3)
                            pyautogui.press('esc')
                            last_action_time = current_time

                # ENTER MODES 
                if not mode_locked and current_time - mode_exit_time > MODE_EXIT_DELAY:
                    if display_label == "palm":
                        print("ENTER NAV MODE")
                        current_mode = "NAVIGATION"
                        mode_locked = True
                        prev_x = None
                        gesture_active = False

                    elif display_label == "one_finger":
                        print("ENTER POINTER MODE")
                        current_mode = "POINTER"
                        mode_locked = True
                        # Ensure slideshow is focused for best user experience
                        focus_slideshow()
                        # Reset smoothing variables
                        prev_hand_x, prev_hand_y = None, None
                        smooth_x, smooth_y = None, None
                        # Start the visible overlay
                        overlay.start()

                #  EXIT MODE 
                elif mode_locked:
                    if display_label == "fist":
                        print("EXIT MODE")
                        # Stop overlay when leaving pointer mode
                        overlay.stop()
                        current_mode = "IDLE"
                        mode_locked = False
                        prev_x = None
                        gesture_active = False
                        mode_exit_time = current_time

                # SWIPE (NAVIGATION MODE) 
                center_x = (x_min + x_max) // 2

                if current_mode == "NAVIGATION":
                    if not gesture_active:
                        prev_x = center_x
                        gesture_active = True
                    else:
                        diff = center_x - prev_x
                        if abs(diff) > SWIPE_THRESHOLD:
                            if current_time - last_action_time > ACTION_COOLDOWN:
                                win = focus_slideshow()
                                if win:
                                    print(f"Slideshow window: {win.title}")
                                else:
                                    print("WARNING: No slideshow window found!")
                                time.sleep(0.1)

                                if diff < 0:
                                    print("NEXT SLIDE (Right Arrow)")
                                    pyautogui.press('right')
                                else:
                                    print("PREVIOUS SLIDE (Left Arrow)")
                                    pyautogui.press('left')
                                last_action_time = current_time
                                gesture_active = False

                #  POINTER MODE (absolute positioning with overlay) 
                if current_mode == "POINTER":
                    # Get index finger tip (landmark ID 8)
                    index_tip = landmarks[8]
                    raw_x = index_tip.x * w
                    raw_y = index_tip.y * h

                    # Apply exponential smoothing to reduce jitter
                    if prev_hand_x is None:
                        smooth_x, smooth_y = raw_x, raw_y
                    else:
                        smooth_x = ALPHA * raw_x + (1 - ALPHA) * smooth_x
                        smooth_y = ALPHA * raw_y + (1 - ALPHA) * smooth_y

                    # Map smoothed hand coordinates to screen coordinates
                    norm_x = np.clip(smooth_x / w, MARGIN_POINTER, 1 - MARGIN_POINTER)
                    norm_y = np.clip(smooth_y / h, MARGIN_POINTER, 1 - MARGIN_POINTER)

                    # Invert x if needed to correct mirrored camera direction
                    if INVERT_X:
                        norm_x = 1 - norm_x

                    cursor_x = norm_x * screen_w
                    cursor_y = norm_y * screen_h

                    # Move the cursor and update the overlay
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0)
                    overlay.update_position(cursor_x, cursor_y)

                    # Update previous values for next frame
                    prev_hand_x, prev_hand_y = smooth_x, smooth_y

                    # (No clicks on two_finger or ok – they are ignored)
                    # If you want to add other actions, place them here.

            # Update previous_label for stability tracking
            previous_label = raw_label

        # DISPLAY INFO ON FRAME (draw if we have a valid ROI) 
        if roi.size > 0:
            conf_text = f"{confidence*100:.1f}%"
            if confidence < CLICK_CONFIDENCE_THRESHOLD:
                conf_text += " (low conf)"

            # Show current gesture and mode
            cv2.putText(frame,
                        f"{display_label} | {conf_text}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

            cv2.putText(frame,
                        f"Mode: {current_mode}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2)

            # Draw bounding box around hand
            cv2.rectangle(frame,
                          (x_min, y_min),
                          (x_max, y_max),
                          (255, 0, 0),
                          2)

    else:
        # No hand detected  reset gesture tracking
        gesture_active = False
        prev_x = None

    cv2.imshow("Gesture PPT Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
overlay.stop()   # ensure overlay is closed when the script ends

