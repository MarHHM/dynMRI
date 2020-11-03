import os
import subprocess

#%% play with os.system() & invoking the WSL bash shell through it

go = '"' + r'C:\Program Files\WindowsApps\CanonicalGroupLimited.Ubuntu18.04onWindows_1804.2020.824.0_x64__79rhkp1fndgsc' \
           r'\ubuntu1804.exe' + '"' + ' ' \
          + 'run' + ' ' + './invoke_FSL.sh'
           # + '-help'
# os.system(go)
subprocess.run(go)