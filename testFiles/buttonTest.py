# buttonTest.py
import time
from buttonHandler import ButtonHandler  # import your class

BUTTON_PIN = 17  # GPIO17 (physical pin 11)

def onButtonPress(channel):
    print("Button Pressed")

def main():
    # Create a ButtonHandler instance with your callback
    button = ButtonHandler(pin=BUTTON_PIN, callback=onButtonPress)

    print("Waiting for button press... (Press CTRL+C to exit)")
    try:
        while True:
            time.sleep(1)  # idle loop, callback does the work
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        button.cleanup()

if __name__ == "__main__":
    main()
