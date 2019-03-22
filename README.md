# hypertrain
## Image recognition of an autonomous train

### Instructions:
Pull Raspberry Image from here: https://drive.google.com/drive/folders/1YQowDY1B15QcnUpEM-oBepjDDtdwZNe5?usp=sharing
comes with precompiled sources & dependencies:\
Python 3.5.3, OpenCV 3.4.3, imutils, numpy, scipy, matplotlib

#### Dependencies (install in cv):
Further dependencies have to be installed to use i2c bus:
> sudo apt-get install build-essential libi2c-dev i2c-tools python-dev libffi-dev\
> pip install smbus-cffi==0.5.1

As of 22.03.2019, rpi.gpio and gpiozero is a new dependency:
> pip install RPi.GPIO\
> pip install gpiozero

#### Client setup:
Install X-Ming to be able to see image output (causes exceptions if not installed and not using headless mode)\
https://sourceforge.net/projects/xming/

Putty Settings:\
hypertrain.local:22\
Connections / SSH / Enable X11 forwarding = True

Login: pi // raspberry\
Hostname: hypertrain

#### Run hypertrain.py:
On login, execute following commands to start:
> source ~/.profile\
> workon cv\
> cd hypertrain\
> python hypertrain.py
