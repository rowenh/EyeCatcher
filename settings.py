hrnet_path="hrnet"

# Model directories:
visibility_model_path="visibility_model"
open_model_path="open_model"
rem_model_path="rem_model"
sleep_model_path="sleep_model"

# General settings
eye_resolution=56
eye_scale=1.2
landmark_threshold=0.1
landmarks_path="output_body.json"
rem_frames=6
sleep_window=60 # Number of seconds to consider for predicting a sleep stage
default_fps=-1 # If -1, does not change FPS of video
window_seconds=1 # Window time for predicting eye state
