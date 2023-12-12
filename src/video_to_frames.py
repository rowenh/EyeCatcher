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
import cv2
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import print_progress_bar

def extract_frames(input_path, output_path, target_fps, direction):
    if not os.path.exists(input_path):
        print("Video path does not exist.")
        return

    if os.path.exists(output_path):
        print("Folder already exists.")
        return
    print("Writing output to " + output_path + "...")
    os.mkdir(output_path)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if target_fps < 0:
        target_fps = fps
    if fps < target_fps:
        print("Video's 'fps' is too low for target!")
        return
    delta_time = 1.0 / fps
    target_delta_time = 1.0 / target_fps
    virtual_timer = target_delta_time # Start with first frame
    index = 0
    total_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result = []
    while 1:
        print_progress_bar(index, total_images, prefix= "Progress:", suffix="Complete",length=50)
        success = cap.grab()
        if not success:
            break
        if virtual_timer >= target_delta_time:
            virtual_timer = virtual_timer - target_delta_time
            _, image = cap.retrieve()
            frame = image

            if direction == "left":
                frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
            elif direction == "right":
                frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif direction == "up":
                pass
            elif direction == "down":
                frame = cv2.rotate(frame,cv2.ROTATE_180)

            if (output_path != None):
                cv2.imwrite(output_path + "/" + str(index) + ".jpg", frame)
            result.append((frame, str(index), virtual_timer))
        virtual_timer = virtual_timer + delta_time
        index = index + 1

    return result
