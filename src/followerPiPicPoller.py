import cv2
import os
import time
import pygame
from pathlib import Path
from screeninfo import get_monitors

# Paths to the images
PIC_DIR = Path.home() / "pics"
PIC1 = PIC_DIR / "pic1.jpg"
PIC2 = PIC_DIR / "pic2.jpg"

lastCaptureTime = {"pic1": None, "pic2": None}

# Setup monitors
monitor0 = {"width":1920,"height":1080,"x":0,"y":0}
monitor1 = {"width":1920,"height":1080,"x":1920,"y":0}

# Total desktop size (side-by-side layout assumed)
total_width = monitor0["width"] + monitor1["width"]
total_height = max(monitor0["height"], monitor1["height"])

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((total_width, total_height), pygame.NOFRAME)

def cv2ToPygame(img):
    """Convert OpenCV image (BGR) to pygame surface."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))

def showImg(imgPath, monitor, offsetX):
    if not imgPath.exists():
        print(f"Warning: {imgPath} not found")
        return

    img = cv2.imread(str(imgPath))
    if img is None:
        print(f"Error: Could not load {imgPath}")
        return

    # Scale to fit monitor
    img = cv2.resize(img, (monitor["width"], monitor["height"]))
    surface = cv2ToPygame(img)

    # Draw at offsetX (left edge of that monitor)
    screen.blit(surface, (offsetX, 0))
    pygame.display.update()

def check_and_update():
    global lastCaptureTime

    if PIC1.exists():
        mtime1 = PIC1.stat().st_mtime
        if lastCaptureTime["pic1"] != mtime1:
            showImg(PIC1, monitor0, 0)
            lastCaptureTime["pic1"] = mtime1

    if PIC2.exists():
        mtime2 = PIC2.stat().st_mtime
        if lastCaptureTime["pic2"] != mtime2:
            showImg(PIC2, monitor1, monitor0["width"])
            lastCaptureTime["pic2"] = mtime2

def main():
    print("Displaying initial images...")
    if PIC1.exists():
        lastCaptureTime["pic1"] = PIC1.stat().st_mtime
        showImg(PIC1, monitor0, 0)

    if PIC2.exists():
        lastCaptureTime["pic2"] = PIC2.stat().st_mtime
        showImg(PIC2, monitor1, monitor0["width"])

    print("Monitoring ~/pics for changes...")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        check_and_update()
        time.sleep(.1)

    pygame.quit()

if __name__ == "__main__":
    main()
