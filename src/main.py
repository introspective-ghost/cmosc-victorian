import os
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'

import sys
import time
import signal
import cv2
import numpy as np
import subprocess
import libcamera
import pygame

from pathlib import Path
from screeninfo import get_monitors
from threading import Thread
from gc import collect
from datetime import datetime
from picamera2 import Picamera2
from buttonHandler import ButtonHandler
from piFileTransfer import LocalNetworkPicTransfer
time.sleep(3)

# --- CONFIG ---
# Frame-to-canvas size ratios (preserved when scaling to actual monitor resolution)
FRAME_WIDTH_RATIO  = 1350 / 1920
FRAME_HEIGHT_RATIO = 1080 / 1080

PATH_TO_REPO = Path.home() / "cmosc-victorian"
BACKUP_BG_IMG_PATH = PATH_TO_REPO / "backgroundImages/backdrop01.jpg"
LOCAL_BG_FOLDER = PATH_TO_REPO / "backgroundImages"
USB_MOUNT_BASE = Path("/media")
BG_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_CONSECUTIVE_ERRORS = 5
WATCHDOG_DELAY = 3  # seconds before restart if unrecoverable
BUTTON_PIN = 17  # GPIO pin for button

canvas = None
button = None
fileTransporter = None
pendingCapture = False
debounceActive = False

# Setup monitors
_monitors = sorted(get_monitors(), key=lambda m: m.x)
monitor0 = {"width": _monitors[0].width, "height": _monitors[0].height, "x": _monitors[0].x, "y": _monitors[0].y}
monitor1 = {"width": _monitors[1].width, "height": _monitors[1].height, "x": _monitors[1].x, "y": _monitors[1].y}
print(_monitors)
CANVAS_WIDTH  = monitor0["width"]
CANVAS_HEIGHT = monitor0["height"]
FRAME_WIDTH   = int(CANVAS_WIDTH  * FRAME_WIDTH_RATIO)
FRAME_HEIGHT  = int(CANVAS_HEIGHT * FRAME_HEIGHT_RATIO)

# Total desktop size (side-by-side layout assumed)
total_width = CANVAS_WIDTH + monitor1["width"]
total_height = max(CANVAS_HEIGHT, monitor1["height"])

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((total_width, total_height), pygame.RESIZABLE)

# --- LOGGING ---
logDir = PATH_TO_REPO / "logs"
logDir.mkdir(parents=True, exist_ok=True)
LOG_FILE = logDir / f"greenscreen_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
def logMsg(level, msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{level}] {msg}"
    print(entry)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

def startupChecks():
    if FRAME_WIDTH > CANVAS_WIDTH or FRAME_HEIGHT > CANVAS_HEIGHT:
        logMsg("ERROR", f"Frame size ({FRAME_WIDTH}x{FRAME_HEIGHT}) exceeds canvas size ({CANVAS_WIDTH}x{CANVAS_HEIGHT})")
        sys.exit(1)
    if not BACKUP_BG_IMG_PATH.exists():
        logMsg("ERROR", f"Background image not found: {str(BACKUP_BG_IMG_PATH)}")
        sys.exit(1)
    bgTest = cv2.imread(str(BACKUP_BG_IMG_PATH))
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
        picam2.stop_preview()
        picam2.stop()
        picam2.close()
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
        collect()
        logMsg("INFO", "Garbage collection completed")
    except Exception as e:
        logMsg("WARNING", f"Garbage collection issue: {e}")
    
    try:
        fileTransporter.close()
        logMsg("INFO", "SSH connection closed")
    except Exception as e:
        logMsg("WARNING", "Problem closing ssh connection")
        
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

def scaleBgToCanvas(bgImg, canvasWidth=CANVAS_WIDTH, canvasHeight=CANVAS_HEIGHT, isZoomingWidth=True):
    """
    Scale background to exactly (canvasWidth x canvasHeight) using zoom-and-crop. 
    This is the single source of truth for background scaling so
    that the frame region and the canvas padding always match.
    """
    if isZoomingWidth:
        scale = canvasWidth / bgImg.shape[1]
        newW = canvasWidth
        newH = int(bgImg.shape[0] * scale)
        bgScaled = cv2.resize(bgImg, (newW, newH), interpolation=cv2.INTER_LINEAR)
        if newH >= canvasHeight:
            yStart = (newH - canvasHeight) // 2
            return bgScaled[yStart:yStart+canvasHeight, :]
        else:
            padTop = (canvasHeight - newH) // 2
            padBottom = canvasHeight - newH - padTop
            return cv2.copyMakeBorder(bgScaled, padTop, padBottom, 0, 0,
                                      cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        scale = canvasHeight / bgImg.shape[0]
        newH = canvasHeight
        newW = int(bgImg.shape[1] * scale)
        bgScaled = cv2.resize(bgImg, (newW, newH), interpolation=cv2.INTER_LINEAR)
        if newW >= canvasWidth:
            xStart = (newW - canvasWidth) // 2
            return bgScaled[:, xStart:xStart+canvasWidth]
        else:
            padLeft = (canvasWidth - newW) // 2
            padRight = canvasWidth - newW - padLeft
            return cv2.copyMakeBorder(bgScaled, 0, 0, padLeft, padRight,
                                      cv2.BORDER_CONSTANT, value=(0,0,0))

def centerFrameInCanvas(frame, bgImg, canvasWidth=CANVAS_WIDTH, canvasHeight=CANVAS_HEIGHT):
    """
    Places `frame` centered inside a fixed-size canvas.
    Side padding is filled with the background image scaled the same way as the
    frame region, so the two always blend seamlessly regardless of source image size.
    """
    h, w = frame.shape[:2]
    canvas = scaleBgToCanvas(bgImg, canvasWidth, canvasHeight).copy()
    xOffset = (canvasWidth - w) // 2
    yOffset = (canvasHeight - h) // 2
    if xOffset < 0 or yOffset < 0:
        raise ValueError("Frame larger than canvas — increase canvas size")
    canvas[yOffset:yOffset+h, xOffset:xOffset+w] = frame
    return canvas

def fitAndCropGreenscreenBackground(bgImg, frameWidth=FRAME_WIDTH, frameHeight=FRAME_HEIGHT,
                         canvasWidth=CANVAS_WIDTH, canvasHeight=CANVAS_HEIGHT,
                         isZoomingWidth=True):
    """
    Scale background to canvas size (via scaleBgToCanvas), then crop the center
    region matching the frame. We will use this cropped center image as our
    greenscreen image. Uses the same scaling as centerFrameInCanvas so the
    frame and canvas padding are always pixel-aligned.
    """
    bgCanvas = scaleBgToCanvas(bgImg, canvasWidth, canvasHeight, isZoomingWidth)
    xFrame = (canvasWidth - frameWidth) // 2
    yFrame = (canvasHeight - frameHeight) // 2
    return bgCanvas[yFrame:yFrame+frameHeight, xFrame:xFrame+frameWidth]

# --- USB BACKGROUND SOURCE ---
def findUsbBackgroundFolder():
    """Search /media/ (up to 2 levels) for a mounted drive with a 'backgrounds' folder
    that contains at least one image file. Handles physical yanking gracefully."""
    if not USB_MOUNT_BASE.exists():
        return None
    try:
        for entry in USB_MOUNT_BASE.iterdir():
            # /media/<drive>/backgrounds
            candidate = entry / "backgrounds"
            if candidate.is_dir():
                try:
                    if any(f.suffix.lower() in BG_IMAGE_EXTENSIONS for f in candidate.iterdir() if f.is_file()):
                        return candidate
                except OSError:
                    pass
            # /media/<user>/<drive>/backgrounds
            if entry.is_dir():
                try:
                    for subEntry in entry.iterdir():
                        candidate = subEntry / "backgrounds"
                        if candidate.is_dir():
                            try:
                                if any(f.suffix.lower() in BG_IMAGE_EXTENSIONS for f in candidate.iterdir() if f.is_file()):
                                    return candidate
                            except OSError:
                                pass
                except OSError:
                    pass
    except OSError:
        pass
    return None

def loadBackgrounds(folder):
    """Return a sorted list of image file Paths from the given folder."""
    try:
        return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in BG_IMAGE_EXTENSIONS)
    except OSError:
        return []

def getBackgroundSource():
    """Return (folder_path, is_usb). Prefers USB 'backgrounds' folder; falls back to local."""
    usbFolder = findUsbBackgroundFolder()
    if usbFolder is not None:
        return usbFolder, True
    return LOCAL_BG_FOLDER, False

# --- BUTTON HANDLER ---
lastPressTime = 0

def onButtonPress(channel):
    global lastPressTime, pendingCapture, captureStartTime
    now = time.time()
    if now - lastPressTime < 3:
        return  # Ignore if less than 3 seconds since last press
    lastPressTime = now
    logMsg("INFO", "Button pressed")
    pendingCapture = True
    captureStartTime = now
    
# --- PYGAME SETUP ---
def cv2ToPygame(img):
    """Convert OpenCV image (BGR) to pygame surface."""
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(imgRgb.swapaxes(0, 1))

def showImg(imgPath, monitor, offsetX):
    if not imgPath.exists():
        logMsg("ERROR", f"{imgPath} not found")
        return

    img = cv2.imread(str(imgPath))
    if img is None:
        logMsg("Error", f"Could not load {imgPath}")
        return

    # Scale to fit monitor
    img = cv2.resize(img, (monitor["width"], monitor["height"]))
    surface = cv2ToPygame(img)

    # Draw at offset_x (left edge of that monitor)
    screen.blit(surface, (offsetX, 0))
    pygame.display.update()
    
def showStream(surface):
    # Draw at top left corner of monitor 0
    screen.blit(surface, (0, 0))
    pygame.display.update()

# --- MAIN  PROCESSING LOOP ---
def runPipeline():
    global picam2, button, pendingCapture, fileTransporter
    
    bgFolder, usingUsb = getBackgroundSource()
    backgrounds = loadBackgrounds(bgFolder)
    logMsg("INFO", f"Backgrounds loaded from {'USB' if usingUsb else 'local folder'}: {bgFolder} ({len(backgrounds)} images)")
    if not backgrounds:
        logMsg("ERROR", f"No background images found in {bgFolder}")
        raise RuntimeError()
    cropX = 0
    cropY = 0
    rotate180 = libcamera.Transform(hflip=True, vflip=True)
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(sensor={"output_size": [2304,1296], "bit_depth": 10},main={"size": [CANVAS_WIDTH,CANVAS_HEIGHT],"format":"BGR888"},buffer_count=6, transform=rotate180)
        picam2.configure(config)
        offset = [1000,1000]#[int((2304/2) - (FRAME_WIDTH/2)),int((1296/2) - (FRAME_HEIGHT/2))]
        picam2.set_controls({"ScalerCrop": offset + [2304,1296]})
        picam2.start()
        
        time.sleep(0.5)
        logMsg("INFO", "Camera started successfully")
        
        button = ButtonHandler(BUTTON_PIN, onButtonPress)
        # victorian1 has a static IPv4 address
        fileTransporter = LocalNetworkPicTransfer("victorian1.local", "cmosc")
        
        # HSV thresholds for green screen
        hLow, sLow, vLow = 38,50,75
        hHigh, sHigh, vHigh = 85, 255, 255

        errCnt = 0
        pictureCnt = 0
        backgroundCnt = 0
        # select the last image in the list to be the first background so our first button press shows the 0th image in the array
        bgImgOriginal = cv2.imread(str(backgrounds[len(backgrounds) - 1]))

        # Show last captured pic0 on monitor1 at startup if it exists
        pic0Path = PATH_TO_REPO / "pics/pic0.jpg"
        if pic0Path.exists():
            showImg(pic0Path, monitor1, CANVAS_WIDTH)

        while True:                   
            frame = picam2.capture_array()
            if frame is None or frame.size == 0:
                logMsg("ERROR", "Initial camera frame capture failed")
                raise RuntimeError()
            # flip image over y-axis
            frame = cv2.flip(frame, 1)
            
#             if cropX + FRAME_WIDTH > CANVAS_WIDTH or cropY + FRAME_HEIGHT > CANVAS_HEIGHT:
#                 raise ValueError(f"Crop out of bounds: X={cropX}, Y={cropY}")

# 			# THIS IS DOING THE FRAME CROPPING OF THE CAMERA IMAGE
            cropped = frame[cropY:cropY + FRAME_HEIGHT, cropX:cropX + FRAME_WIDTH]
            #if cropped.size == 0:
#                 raise ValueError("ERROR", "Cropped frame empty")
            
            # Zoom horizontally
            greenScreenImg = fitAndCropGreenscreenBackground(bgImgOriginal, FRAME_WIDTH, FRAME_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT, True)
            
            # fix color channels if needed
            greenScreenImg  = matchFrameColorChannelsToTarget(greenScreenImg, cropped.shape[2])
            cropped = matchFrameColorChannelsToTarget(cropped, greenScreenImg.shape[2])
            
            # Create greenscreen mask
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([hLow, sLow, vLow]), np.array([hHigh, sHigh, vHigh]))
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel) # remove small noise
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel) # close small holes
            
            # Resize mask to match cropped frame
            if cropped.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (cropped.shape[1], cropped.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Resize background to match frame
            if cropped.shape[:2] != greenScreenImg.shape[:2]:
                greenScreenImg = cv2.resize(greenScreenImg, (cropped.shape[1], cropped.shape[0]))
            
            maskInv = cv2.bitwise_not(mask)

            fg = cv2.bitwise_and(cropped, cropped, mask=maskInv)
            bg = cv2.bitwise_and(greenScreenImg, greenScreenImg, mask=mask)
            composite = cv2.add(fg, bg)
            composite = cv2.medianBlur(composite, 3)

            padded = centerFrameInCanvas(composite, bgImgOriginal, CANVAS_WIDTH, CANVAS_HEIGHT)
            streamSurface = cv2ToPygame(padded)
            
            showStream(streamSurface)

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    cleanupAndExit()

            if pendingCapture and (time.time() - captureStartTime >= 3):
                if pictureCnt == 3:
                    pictureCnt = 0
                # create greyscale image
                grayCanvas = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
                # handle image path creation and save to path
                folderPath = PATH_TO_REPO / "pics/"
                folderPath.mkdir(exist_ok=True)
                fileName = f"pic{pictureCnt}.jpg"
                
                cv2.imwrite(str(folderPath / fileName), grayCanvas)
                logMsg("INFO", f"Saved delayed capture: {fileName}")
        
        # display image on screen for 3 seconds
                showImg(folderPath / fileName, monitor0, 0)
                time.sleep(3)                
        
        # pic0 is displayed on monitor1
                if pictureCnt == 0:
                    showImg(folderPath / fileName, monitor1, CANVAS_WIDTH)
                # pic1 and pic2 get sent to follower rpi
                if pictureCnt == 1 or pictureCnt == 2:
                    try:
                        fileTransporter.sendFile(str(folderPath / fileName), f"~/pics/{fileName}")
                    except Exception as e:
                        logMsg("ERROR", f"rsync file transfer failed: {e}")
                        raise RuntimeError("Follower Pi not found. Check Ethernet cable/connection")
                pictureCnt += 1
                
                # Check if USB was inserted or removed since last capture
                newBgFolder, newUsingUsb = getBackgroundSource()
                if newBgFolder != bgFolder:
                    newBackgrounds = loadBackgrounds(newBgFolder)
                    if newBackgrounds:
                        bgFolder = newBgFolder
                        usingUsb = newUsingUsb
                        backgrounds = newBackgrounds
                        backgroundCnt = 0
                        logMsg("INFO", f"Background source switched to {'USB' if usingUsb else 'local folder'}: {bgFolder}")
                    else:
                        logMsg("WARNING", f"New background source {newBgFolder} is empty, keeping current")

                # update the background image
                if backgroundCnt >= len(backgrounds):
                    backgroundCnt = 0
                bgImgOriginal = cv2.imread(str(backgrounds[backgroundCnt]))
                backgroundCnt += 1
                
                pendingCapture = False  # reset

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
            logMsg("ERROR", f"Watchdog caught Fatal error: {e}, Exiting...")
            break
