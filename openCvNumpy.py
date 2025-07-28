import cv2
import numpy as np
from picamera2 import Picamera2, Preview
import time
import tkinter as tk


def exitCleanup():
    picam2.stop()
    cv2.destroyAllWindows()
    exit()

root = tk.Tk()
screenWidth = root.winfo_screenwidth()
screenHeight = root.winfo_screenheight()
root.destroy()

print("Screen Width: " + str(screenWidth))
print("Screen Height: " + str(screenHeight))
picam2 = Picamera2()
picam2.options["quality"] = 90
picam2.options["compress_level"] = 2
config = picam2.create_preview_configuration(main={"size": (800, 600)})
picam2.start_preview(Preview.NULL)
picam2.configure(config)
picam2.start()
time.sleep(2)

# Load background image (same resolution)
try:
    bg = cv2.imread("800x600.jpg", cv2.IMREAD_UNCHANGED)
except:
    print("ERROR: Could not find background image")
    exitCleanup()


hb, wb = bg.shape[:2]

# Define green range and make mask
lower = np.array([36, 50, 50])
upper = np.array([90, 255, 255])

# Create fullscreen window
cv2.namedWindow("Greenscreen Composite", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Greenscreen Composite", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    frame_rgba = picam2.capture_array("main")  # BGR
    # Remove the alpha channel from the image
    frame = frame_rgba[..., :3]
    # Grab the height and width from the frame
    h, w = frame.shape[:2]
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # Make sure the frame mask and frame background will be the same size
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if bg.shape != (h, w):
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_inv = cv2.bitwise_not(mask)

    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bk = cv2.bitwise_and(bg, bg, mask=mask)
    output = cv2.add(fg, bk)
    cv2.imshow("Greenscreen Composite", output)
    # Press the escape key with the Greenscreen Composite window in focus to end the program
    if cv2.waitKey(1) == 27:
        break

exitCleanup()