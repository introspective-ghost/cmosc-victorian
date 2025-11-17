import os
import sys
import time
import cv2
import numpy as np
import subprocess
from datetime import datetime
from picamera2 import Picamera2

# --- CONFIG ---
FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920
CROP_W = 868
CROP_H = 1169
BG_IMAGE_PATH = "backgroundImages/backdrop01.jpg"
MAX_CONSECUTIVE_ERRORS = 5
WATCHDOG_DELAY = 3  # seconds before restart if unrecoverable

# --- Logging ---
LOG_FILE = f"logs/greenscreen_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
def log_msg(level, msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{level}] {msg}"
    print(entry)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

def startup_checks():
    if CROP_W > FRAME_WIDTH or CROP_H > FRAME_HEIGHT:
        log_msg("ERROR", f"Crop size ({CROP_W}x{CROP_H}) exceeds frame size ({FRAME_WIDTH}x{FRAME_HEIGHT})")
        sys.exit(1)
    if not os.path.exists(BG_IMAGE_PATH):
        log_msg("ERROR", f"Background image not found: {BG_IMAGE_PATH}")
        sys.exit(1)
    bg_test = cv2.imread(BG_IMAGE_PATH)
    if bg_test is None:
        log_msg("ERROR", "Failed to read background image")
        sys.exit(1)

def match_frame_to_target(img, target_channels=3):
    """
    Ensures consistent channel count and order for compositing.
    Converts to BGR with no alpha unless target_channels=4 requested.
    """
    if img is None:
        return None
    
    # If grayscale → promote to BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Handle alpha channels explicitly
    if img.shape[2] == 4:
        if target_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    # If it came in as RGB instead of BGR (from another lib)
    # Detect using heuristics or enforce conversion if known source is RGB
    if target_channels == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def run_pipeline():
    error_count = 0
    prev_time = time.time()
    frame_count = 0

    bg_img_original = cv2.imread(BG_IMAGE_PATH)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)
    log_msg("INFO", "Camera started successfully")

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Crop X", "Controls", 0, FRAME_WIDTH - CROP_W, lambda x: None)
    cv2.createTrackbar("Crop Y", "Controls", 0, FRAME_HEIGHT - CROP_H, lambda x: None)
    cv2.createTrackbar("H Low", "Controls", 40, 179, lambda x: None)
    cv2.createTrackbar("H High", "Controls", 80, 179, lambda x: None)
    cv2.createTrackbar("S Low", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("S High", "Controls", 255, 255, lambda x: None)
    cv2.createTrackbar("V Low", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("V High", "Controls", 255, 255, lambda x: None)

    while True:
        try:
            frame = picam2.capture_array()
            if frame is None or frame.size == 0:
                raise ValueError("Empty frame")
        except Exception as e:
            error_count += 1
            log_msg("WARNING", f"Frame capture failed ({error_count}): {e}")
            if error_count >= MAX_CONSECUTIVE_ERRORS:
                log_msg("ERROR", "Too many consecutive errors, triggering watchdog restart")
                raise RuntimeError("Watchdog restart")
            continue

        error_count = 0  # reset on success

        crop_x = cv2.getTrackbarPos("Crop X", "Controls")
        crop_y = cv2.getTrackbarPos("Crop Y", "Controls")
        h_low = cv2.getTrackbarPos("H Low", "Controls")
        h_high = cv2.getTrackbarPos("H High", "Controls")
        s_low = cv2.getTrackbarPos("S Low", "Controls")
        s_high = cv2.getTrackbarPos("S High", "Controls")
        v_low = cv2.getTrackbarPos("V Low", "Controls")
        v_high = cv2.getTrackbarPos("V High", "Controls")

        if crop_x + CROP_W > FRAME_WIDTH or crop_y + CROP_H > FRAME_HEIGHT:
            log_msg("WARNING", f"Crop out of bounds: X={crop_x}, Y={crop_y}")
            continue

        cropped = frame[crop_y:crop_y + CROP_H, crop_x:crop_x + CROP_W]
        if cropped.size == 0:
            log_msg("WARNING", "Cropped frame empty")
            continue

        

        bg_img = cv2.resize(bg_img_original, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)

        bg_img  = match_frame_to_target(bg_img, cropped.shape[2])
        cropped = match_frame_to_target(cropped, bg_img.shape[2])
        


        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([h_low, s_low, v_low]), np.array([h_high, s_high, v_high]))
        
        if cropped.shape[:2] != bg_img.shape[:2]:
            print("[DEBUG] Resizing background to match cropped frame")
            bg_img = cv2.resize(bg_img, (cropped.shape[1], cropped.shape[0]))

        if cropped.shape[:2] != mask.shape[:2]:
            print("[DEBUG] Resizing mask to match cropped frame")
            mask = cv2.resize(mask, (cropped.shape[1], cropped.shape[0]), interpolation=cv2.INTER_NEAREST)


        mask_inv = cv2.bitwise_not(mask)

        fg = cv2.bitwise_and(cropped, cropped, mask=mask_inv)
        bg = cv2.bitwise_and(bg_img, bg_img, mask=mask)
        composite = cv2.add(fg, bg)

        frame_count += 1
        if frame_count >= 10:
            now = time.time()
            fps = frame_count / (now - prev_time)
            cv2.putText(composite, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            prev_time = now
            frame_count = 0

        if error_count > 0:
            cv2.putText(composite, f"Errors: {error_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Greenscreen Composite", composite)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            log_msg("INFO", "Exit requested by user")
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"composite_{ts}.jpg"
            if cv2.imwrite(out_name, composite):
                log_msg("INFO", f"Snapshot saved: {out_name}")
            else:
                log_msg("ERROR", "Snapshot save failed")

    cv2.destroyAllWindows()
    log_msg("INFO", "Pipeline stopped cleanly")

if __name__ == "__main__":
    startup_checks()
    while True:
        try:
            run_pipeline()
            break  # exit if run_pipeline completes without watchdog trigger
        except RuntimeError as e:
            log_msg("WARNING", f"Watchdog caught error: {e}, restarting in {WATCHDOG_DELAY}s")
            time.sleep(WATCHDOG_DELAY)
        except Exception as e:
            log_msg("ERROR", f"Fatal error: {e}")
            break