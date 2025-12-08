# buttonHandler.py
import RPi.GPIO as GPIO

class ButtonHandler:
    def __init__(self, pin=17, callback=None, bouncetime=300):
        self.pin = pin
        self.callback = callback

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        if self.callback:
            GPIO.add_event_detect(self.pin,
                                  GPIO.FALLING,
                                  callback=self.callback,
                                  bouncetime=bouncetime)

    def cleanup(self):
        GPIO.cleanup(self.pin)
