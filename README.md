# hypertrain
Image recognition of an autonomous train

Instructions:
Pull Raspberry Image from here: https://drive.google.com/drive/folders/1YQowDY1B15QcnUpEM-oBepjDDtdwZNe5?usp=sharing
comes with precompiled sources & dependencies:
Python 3.5.3, OpenCV 2, imutils, numpy, scipy, matplotlib

Install X-Ming to be able to see image output (causes exceptions if not installed and not using headless mode)
https://sourceforge.net/projects/xming/

Putty Settings:
hypertrain.local:22
Connections / SSH / Enable X11 forwarding = True

Login: pi // raspberry
Hostname: hypertrain

On login, execute following commands to start :
> source ~/.profile
> workon cv
> cd hypertrain
> python hypertrain.py
