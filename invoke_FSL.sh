#!/bin/bash

## dummy script to play with the bash invoked in my python code

# commands so that FSL commands that need a GUI inside the WSL bash can access the XLaunch one (should be already running)
echo "export DISPLAY=$(ip route | awk '/default via / {print $3; exit}' 2>/dev/null):0" >> ~/.bashrc
echo "export LIBGL_ALWAYS_INDIRECT=1" >> ~/.bashrc

cd /usr/local/fsl/bin/            # where the FSL bins where installed by fslinstaller.py
./fslsplit