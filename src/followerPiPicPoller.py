import cv2
import os
import time
import pygame
from pathlib import Path
from screeninfo import get_monitors

# os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = "1"
# Paths to the images
PIC_DIR = Path.home() / "pics"
PIC1 = PIC_DIR / "pic1.jpg"
PIC2 = PIC_DIR / "pic2.jpg"

lastCaptureTime = {"pic1": None, "pic2": None}

monitors = sorted(get_monitors(), key=lambda m: m.x)

monitor0 = {"width": monitors[0].width, "height": monitors[0].height,
            "x": monitors[0].x, "y": monitors[0].y}
monitor1 = {"width": monitors[1].width, "height": monitors[1].height,
            "x": monitors[1].x, "y": monitors[1].y}

total_width = sum(m.width for m in monitors)
total_height = max(m.height for m in monitors)

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
    
    if PIC1.exists():
        showImg(PIC1, monitor0, 0)

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
