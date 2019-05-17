import serial
import json
import threading
import time

from gpiozero import LED, Button, Buzzer

from enum import Enum

from lib.state import State

# deserialized JSON Object


class JSONObject(object):
    def __init__(self, j):
        self.__dict__ = json.loads(j)


class Communication:
    """
    Class Communication is used to communicate with the different parts of the train, or even the humans building it

    UART to communicate with Arduino, with a serialized JSON Object. See /UART/communication.schema.json.
    GPIO 4  = Buzzer
    GPIO 12 = Start button
    GPIO 26 = LED
    """

    class __impl:
        def __init__(self, logger):
            self.logger = logger
            self.state = State()
            self.serial = serial.Serial(
                port='/dev/serial0',
                baudrate=9600,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=0)
            self.buzzer = Buzzer(4)
            self.led = LED(26)
            self.button = Button(12, True)
            self.button.when_pressed = lambda: self.toggleHypertrain()

            self.thread = threading.Thread(target=self.readThread)
            self.threadStop = False

        def toggleHypertrain(self):
            self.state.Stopped = not self.state.Stopped
            # self.state.Loaded = not self.state.Loaded
            self.logger.info(
                "Button pressed, new state Stopped: " + str(self.state.Stopped))
            self.led.blink(1, 1, 1)
            if (self.state.Stopped):
                self.sendStartStop('stop')
                self.state.reset()
            else:
                self.sendStartStop('start')

        def sendStartStop(self, action):
            data = {}
            data['sender'] = 'raspberry'
            data['action'] = action
            self.write(json.dumps(data))

        def sendApproachStop(self):
            data = {}
            data['sender'] = 'raspberry'
            data['action'] = 'approachstop'
            self.write(json.dumps(data))

        def sendSpeedPercent(self, acceleration):
            if (acceleration != self.state.LastAccelerationPercent):
                self.state.LastAccelerationPercent = acceleration
                data = {}
                data['sender'] = 'raspberry'
                data['action'] = 'accelerate'
                data['payload'] = acceleration
                self.write(json.dumps(data))

        def buzzSignalNumber(self, num):
            self.buzzer.beep(1, 1, num)

        def readThreadStart(self):
            self.thread.start()

        def readThreadStop(self):
            self.threadStop = True

        def readThread(self):
            while (not self.threadStop):
                self.read()
                time.sleep(self.state.ThreadSleepingThreshold)

        def read(self):
            incoming = ""
            if (self.serial.in_waiting > 0):
                while not "}" in incoming:
                    incoming += self.serial.read(self.serial.inWaiting()
                                                 ).decode('ascii')

                incomingMsgs = incoming.splitlines()
                for incoming in incomingMsgs:
                    if incoming:
                        self.logger.info("Receiving: " + incoming)
                        jsonObj = None
                        try:
                            jsonObj = json.loads(incoming)
                            if (jsonObj["sender"] == "arduino"):
                                if (jsonObj["action"] == "loaded"):
                                    self.led.blink(1, 1, 1)
                                    self.buzzSignalNumber(1)
                                    if (not self.state.Loaded):
                                        self.state.Loaded = True
                                if (jsonObj["action"] == "speed"):
                                    continue
                                if (jsonObj["action"] == "way" and jsonObj["payload"]):
                                    self.state.CoveredDistance = int(
                                        jsonObj["payload"])
                                    continue
                        except AttributeError as e:
                            self.logger.error(
                                "AttributeError in JSON: " + str(e))
                        except Exception as e:
                            self.logger.error("Unknown message: " + str(e))

        def write(self, message):
            if (message):
                self.logger.info("Sending: " + message)
                self.serial.write(message.encode())

    # Singleton
    __inst = None

    def __init__(self, logger):
        # Check whether we already have an instance
        if Communication.__inst is None:
            Communication.__inst = Communication.__impl(logger)

        # Store instance reference in the handle
        self.__dict__["_Communication__inst"] = Communication.__inst

    # Delegate attribute getters/setters to instance
    def __getattr__(self, attr):
        return getattr(self.__inst, attr)

    def __setattr__(self, attr, value):
        return setattr(self.__inst, attr, value)
