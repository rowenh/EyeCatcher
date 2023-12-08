import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import settings

def generate_landmarks(path):
    path = path[:-1] if path[-1]=="/" else path
    if os.path.exists(path + "/" + settings.landmarks_path):
        print("Landmarks already extracted.")
        return
    cached_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))).replace("\\", "/") + "/hrnet")
    c = os.system("python3 " + os.path.dirname(os.path.dirname(os.path.realpath(__file__))).replace("\\", "/") + "/hrnet/tools/valid.py --input '" + path + "/frames' --output '" + path + "'") 
    os.chdir(cached_cwd)
    if c != 0 and c != "0":
        raise Exception("HRNet threw an error.")