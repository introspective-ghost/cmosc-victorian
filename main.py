import os
import sys
import time
import signal
import cv2
import numpy as np
from datetime import datetime
import gc
from buttonHandler import ButtonHandler

# --- CONFIG ---
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
FRAME_WIDTH = 1420
FRAME_HEIGHT = 1080
BG_IMAGE_PATH = "backgroundImages/backdrop01.jpg"
MAX_CONSECUTIVE_ERRORS = 5
WATCHDOG_DELAY = 3  # seconds before restart if unrecoverable
BUTTON_PIN = 17  # GPIO pin for button

cap = None
canvas = None
button = None

# --- LOGGING ---
os.makedirs("logs", exist_ok=True)
LOG_FILE = os.path.join("logs", f"greenscreen_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
def logMsg(level, msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{level}] {msg}"
    print(entry)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

def startupChecks():
    if FRAME_WIDTH > CANVAS_WIDTH or FRAME_HEIGHT > CANVAS_HEIGHT:
        logMsg("ERROR", f"Cropped frame size ({FRAME_WIDTH}x{FRAME_HEIGHT}) exceeds canvas size ({CANVAS_WIDTH}x{CANVAS_HEIGHT})")
        sys.exit(1)
    if not os.path.exists(BG_IMAGE_PATH):
        logMsg("ERROR", f"Background image not found: {BG_IMAGE_PATH}")
        sys.exit(1)
    bgTest = cv2.imread(BG_IMAGE_PATH)
    if bgTest is None:
        logMsg("ERROR", "Failed to read background image")
        sys.exit(1)

_cleanupDone = False
def cleanupAndExit(signum=None, frame=None):
    """Gracefully stop camera + destroy OpenCV windows."""
    global _cleanupDone
    if _cleanupDone:
        return
    _cleanupDone = True
    try:
        if cap is not None:
            cap.release()
            logMsg("INFO", "Camera stopped and closed cleanly")
    except Exception as e:
        logMsg("WARNING", f"Camera cleanup issue: {e}")

    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # flush GUI events
        logMsg("INFO", "OpenCV windows destroyed")
    except Exception as e:
        logMsg("WARNING", f"OpenCV cleanup issue: {e}")

    try:
        if button is not None:
            button.cleanup()
            logMsg("INFO", "GPIO cleaned up")
    except Exception as e:
        logMsg("WARNING", f"GPIO cleanup issue: {e}")

    try:
        gc.collect()
        logMsg("INFO", "Garbage collection completed")
    except Exception as e:
        logMsg("WARNING", f"Garbage collection issue: {e}")

    logMsg("INFO", "Program terminated via signal")
    sys.exit(0)

# Register handlers for Ctrl+C and kill
signal.signal(signal.SIGINT, cleanupAndExit)   # Ctrl+C
signal.signal(signal.SIGTERM, cleanupAndExit)  # kill

# --- IMAGE HELPERS ---
def matchFrameColorChannelsToTarget(img, targetChannels=3):
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
        if targetChannels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    # If it came in as RGB instead of BGR (from another lib)
    # Detect using heuristics or enforce conversion if known source is RGB
    if targetChannels == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def centerInCanvas(frame, bgImg, canvasWidth=CANVAS_WIDTH, canvasHeight=CANVAS_HEIGHT):
    """
    Places `frame` centered inside a fixed-size canvas.
    Side padding is filled with the background image
    
    frame: the composite (cropped subject)
    bgImg: the original background image (will be resized to canvas size)
    """
    h, w = frame.shape[:2]
    # Resize background to match canvas
    bgResized = cv2.resize(bgImg, (canvasWidth, canvasHeight), interpolation=cv2.INTER_AREA)

    # Start with the background as the canvas
    canvas = bgResized.copy()

    # Compute offsets for centering the frame
    xOffset = (canvasWidth - w) // 2
    yOffset = (canvasHeight - h) // 2

    if xOffset < 0 or yOffset < 0:
        raise ValueError("Frame larger than canvas — increase canvas size")

    # Place the frame into the center of the canvas
    canvas[yOffset:yOffset+h, xOffset:xOffset+w] = frame

    return canvas

def fitAndCropBackground(bgImg, frameWidth=FRAME_WIDTH, frameHeight=FRAME_HEIGHT,
                         canvasWidth=CANVAS_WIDTH, canvasHeight=CANVAS_HEIGHT,
                         isZoomingWidth=True):
    """
    Scale the background image to cover the canvas either by width or height,
    then crop/pad to canvas size, and finally cut out a region the same
    size as the frame.

    - If isZoomingWidth=True: scale so width matches canvasWidth.
    - If isZoomingWidth=False: scale so height matches canvasHeight.
    """

    if isZoomingWidth:
        # --- Zoom by width ---
        scale = canvasWidth / bgImg.shape[1]
        newW = canvasWidth
        newH = int(bgImg.shape[0] * scale)
        bgScaled = cv2.resize(bgImg, (newW, newH), interpolation=cv2.INTER_LINEAR)

        if newH > canvasHeight:
            yStart = (newH - canvasHeight) // 2
            bgCanvas = bgScaled[yStart:yStart+canvasHeight, :]
        else:
            # If we messed up and made the bgImg too small, pad it
            padTop = (canvasHeight - newH) // 2
            padBottom = canvasHeight - newH - padTop
            bgCanvas = cv2.copyMakeBorder(bgScaled, padTop, padBottom, 0, 0,
                                          cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        # --- Zoom by height ---
        scale = canvasHeight / bgImg.shape[0]
        newH = canvasHeight
        newW = int(bgImg.shape[1] * scale)
        bgScaled = cv2.resize(bgImg, (newW, newH), interpolation=cv2.INTER_LINEAR)

        if newW > canvasWidth:
            xStart = (newW - canvasWidth) // 2
            bgCanvas = bgScaled[:, xStart:xStart+canvasWidth]
        else:
            # If we messed up and made the bgImg too small, pad it
            padLeft = (canvasWidth - newW) // 2
            padRight = canvasWidth - newW - padLeft
            bgCanvas = cv2.copyMakeBorder(bgScaled, 0, 0, padLeft, padRight,
                                          cv2.BORDER_CONSTANT, value=(0,0,0))

    # --- Final crop to frame size (centered inside canvas) ---
    xFrame = (canvasWidth - frameWidth) // 2
    yFrame = (canvasHeight - frameHeight) // 2
    return bgCanvas[yFrame:yFrame+frameHeight, xFrame:xFrame+frameWidth]

# --- BUTTON HANDLER ---
def onButtonPress(channel):
    global pendingCapture, captureStartTime
    logMsg("INFO", "Button pressed")
    pendingCapture = True
    captureStartTime = time.time()

# --- MAIN  PROCESSING LOOP ---
def runPipeline():
    global cap, button, pendingCapture
    errCnt = 0
    bgImgOriginal = cv2.imread(BG_IMAGE_PATH)
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        time.sleep(0.5)
        logMsg("INFO", "Camera started successfully")

        cv2.namedWindow("Greenscreen Composite", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Greenscreen Composite", 1080, 720)
        
        button = ButtonHandler(BUTTON_PIN, onButtonPress)
        # 0-212
        cropX = 0
        # 0-751
        cropY = 0
        # HSV thresholds for green screen
        hLow, sLow, vLow = 35, 40, 40
        hHigh, sHigh, vHigh = 85, 255, 255

        errCnt = 0
        while True:
            try:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    logMsg("ERROR", "Initial camera frame capture failed")
                    raise RuntimeError("Initial camera frame capture failed")
            except Exception as e:
                errCnt += 1
                logMsg("WARNING", f"Frame capture failed ({errCnt}): {e}")
                if errCnt >= MAX_CONSECUTIVE_ERRORS:
                    logMsg("ERROR", "Too many consecutive errors, triggering watchdog restart")
                    raise RuntimeError("Watchdog restart")
                continue
            errCnt = 0  # reset error count on success
            
            if cropX + FRAME_WIDTH > CANVAS_WIDTH or cropY + FRAME_HEIGHT > CANVAS_HEIGHT:
                raise ValueError(f"Crop out of bounds: X={cropX}, Y={cropY}")

            cropped = frame[cropY:cropY + FRAME_HEIGHT, cropX:cropX + FRAME_WIDTH]
            if cropped.size == 0:
                raise ValueError("ERROR", "Cropped frame empty")

            # Zoom horizontally
            greenScreenImg = fitAndCropBackground(bgImgOriginal, FRAME_WIDTH, FRAME_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT, True)

            greenScreenImg  = matchFrameColorChannelsToTarget(greenScreenImg, cropped.shape[2])
            cropped = matchFrameColorChannelsToTarget(cropped, greenScreenImg.shape[2])
            
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([hLow, sLow, vLow]), np.array([hHigh, sHigh, vHigh]))

            if cropped.shape[:2] != mask.shape[:2]:
                print("[DEBUG] Resizing mask to match cropped frame")
                mask = cv2.resize(mask, (cropped.shape[1], cropped.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            if cropped.shape[:2] != greenScreenImg.shape[:2]:
                print("[DEBUG] Resizing background to match cropped frame")
                greenScreenImg = cv2.resize(greenScreenImg, (cropped.shape[1], cropped.shape[0]))
            
            maskInv = cv2.bitwise_not(mask)

            fg = cv2.bitwise_and(cropped, cropped, mask=maskInv)
            bg = cv2.bitwise_and(greenScreenImg, greenScreenImg, mask=mask)
            composite = cv2.add(fg, bg)

            padded = centerInCanvas(composite, bgImgOriginal, CANVAS_WIDTH, CANVAS_HEIGHT)
            cv2.imshow("Greenscreen Composite", padded)

            if pendingCapture and (time.time() - captureStartTime >= 3):
                grayCanvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                filename = f"logs/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, grayCanvas)
                print(f"Saved delayed capture: {filename}")
                pendingCapture = False  # reset


            # Handle exit and snapshot keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                logMsg("INFO", "Exit requested by user")
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                outName = f"composite_{ts}.jpg"
                if cv2.imwrite(outName, composite):
                    logMsg("INFO", f"Snapshot saved: {outName}")
                else:
                    logMsg("ERROR", "Snapshot save failed")

    except Exception as err:
        logMsg("ERROR", f"Fatal pipeline error: {err}")
        raise # let the watchdog loop catch the error
    finally:
        # Always run the cleanup, even on an error
        cleanupAndExit()

if __name__ == "__main__":
    startupChecks()
    # Watchdog loop
    while True:
        try:
            runPipeline()
            break  # exit if run_pipeline completes without watchdog trigger
        except RuntimeError as e:
            logMsg("WARNING", f"Watchdog caught runtime error: {e}, restarting in {WATCHDOG_DELAY}s")
            time.sleep(WATCHDOG_DELAY)
            continue
        except Exception as e:
            logMsg("WARNING", f"Watchdog caught Fatal error: {e}, restarting in {WATCHDOG_DELAY}s")
            continue
