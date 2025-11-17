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
#picam2.options["quality"] = 90
#picam2.options["compress_level"] = 2
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.start_preview(Preview.NULL)
picam2.configure(config)
picam2.start()
time.sleep(2)

# Create fullscreen window
cv2.namedWindow("Greenscreen Composite", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Greenscreen Composite", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FULLSCREEN)

while True:
    frame_rgba = picam2.capture_array("main")  # BGR
    # Remove the alpha channel from the image
    frame = frame_rgba[..., :3]
    cv2.imshow("Greenscreen Composite", frame_rgba)
    # Press the escape key with the Greenscreen Composite window in focus to end the program
    if cv2.waitKey(1) == 27:
        break

exitCleanup()