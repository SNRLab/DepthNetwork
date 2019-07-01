import os, sys, subprocess
# Loads the weights created by train.sh and allows viewing of the results

os.chdir("../../")
sys.path.append(os.getcwd())
print(os.getcwd())
# Add to environment variables beforehand

command = "python tools/train.py tools/paper/train.yaml"
os.system(command)