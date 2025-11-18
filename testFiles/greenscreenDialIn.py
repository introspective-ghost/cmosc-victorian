import cv2
import numpy as np

# --- Load your images ---
# Foreground: image with green background
foreground = cv2.imread("C:\Users\josh2\Downloads\greenscreen.jpg")
# Background: image to appear behind subject
background = cv2.imread("C:\Users\josh2\OneDrive\Pictures\catalina_staff_pic.jpg")

# Ensure both images are the same size
background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

# --- Convert to HSV for better color masking ---
hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)

# Define green range in HSV
lower_green = np.array([35, 40, 40])   # adjust as needed
upper_green = np.array([85, 255, 255]) # adjust as needed

# Create a mask where green pixels are detected
mask = cv2.inRange(hsv, lower_green, upper_green)

# Invert mask to get subject
mask_inv = cv2.bitwise_not(mask)

# Extract the subject from the foreground
subject = cv2.bitwise_and(foreground, foreground, mask=mask_inv)

# Extract background only where green was detected
bg_part = cv2.bitwise_and(background, background, mask=mask) 

# Combine background and subject
final = cv2.add(bg_part, subject)

# --- Save / Show result ---
cv2.imshow("Greenscreen Result", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("greenscreen_result.jpg", final)
