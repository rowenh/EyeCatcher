# Copyright 2023 Rowen Horbach, Eline R. de Groot, Jeroen Dudink, Ronald Poppe.

# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with 
# this program. If not, see <https://www.gnu.org/licenses/>.

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