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

eye_resolution=56
eye_scale=1.2
landmark_threshold=0.1
landmarks_path="output_body.json"
rem_frames=6
sleep_window=60 # Number of seconds to consider for predicting a sleep stage
default_fps=-1 # If -1, does not change FPS of video
window_seconds=1 # Window time for predicting eye state
