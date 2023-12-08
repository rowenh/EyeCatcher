import sys
import cv2
import os
import numpy as np
import tensorflow as tf
import settings
import src.video_to_frames
import src.model_helper
import rem_model.update_samples
import src.eye_extractor
import src.hrnet_api
from src.utils import print_progress_bar
from src.frame_level_graph_generator import generate_graph
from src.eye_state_generator import generate_states
from visibility_model.model import VisibilityModel
from open_model.model import OpenModel
from rem_model.model import RemModel
from sleep_model.model import SleepModel

def predict_video(video_path, output_directory, id, body_direction):
    output_directory = output_directory if output_directory[-1] == "/" else output_directory + "/"
    if not os.path.exists(output_directory + "samples"):
        generate_dependencies(video_path, output_directory, id, body_direction)
    print("Generating final sleep graph.")
    sm = SleepModel()
    smm = sm.load_model()
    cached_dir = sm.dir_
    sm.dir_ = output_directory
    samples = sm.get_samples()
    sm.dir_ = cached_dir
    inp = np.array([np.array(entry[0]) for entry in samples])
    preds = tf.argmax(smm.predict(inp), 1).numpy().tolist()
    sm.save_sleep_graph(samples, preds, output_directory + "sleep_output.png")
    with open(output_directory + "sleep_output", "w") as f:
        f.write("Predictions: " + str(preds))
    print("Finished (successfully).")

def generate_dependencies(video_path, output_directory, id, body_direction):
    fps = settings.rem_frames / settings.window_seconds # Most efficient/minimal FPS; higher FPS is unlikely to improve results.
    src.video_to_frames.extract_frames(video_path, output_directory + "frames", fps, body_direction)
    if not os.path.exists(output_directory + settings.landmarks_path):
        src.hrnet_api.generate_landmarks(output_directory)

    # Initialization
    vm = VisibilityModel()
    om = OpenModel()
    rm = RemModel()
    vmm = vm.load_model()
    omm = om.load_model()
    rmm = rm.load_model()
    visibility_threshold = 0.5
    open_threshold = 0.5
    rem_threshold = 0.9
    window_length = max(int(settings.window_seconds * fps), settings.rem_frames)
    window_indices = rem_model.update_samples.get_indices(window_length, settings.rem_frames)
    start_offset = -int(window_length / 2.0)
    history = []
    rem_confidence_history = []
    original_frame_ids = sorted(os.listdir(output_directory + "frames"),key=lambda x: int(x.replace('.jpg','')))
    landmarks = src.eye_extractor.get_landmarks(output_directory + settings.landmarks_path, original_frame_ids)
    total_images = len(os.listdir(output_directory + "frames"))

    final_images = []
    for i,frame_id in enumerate(original_frame_ids):
        print_progress_bar(i, total_images, prefix= "Progress:", suffix="Complete",length=50)

        original_frame = cv2.imread(output_directory + "frames/" + frame_id)

        if (i + start_offset) < 0 or (i + start_offset + window_length - 1) >= total_images:
            final_images.append(original_frame)
            continue

        sample_frame_ids = original_frame_ids[i + start_offset:i + start_offset + window_length]

        left_accuracy = 0
        right_accuracy = 0
        acceptable_locations = []
        for i_,frame_id_ in enumerate(sample_frame_ids):
            if frame_id_ not in landmarks:
                continue
            (orig_nose, orig_eye1, orig_eye2) = landmarks[frame_id_]
            if orig_nose[2] > settings.landmark_threshold and orig_eye1[2] > settings.landmark_threshold and orig_eye2[2] > settings.landmark_threshold:
                acceptable_locations.append(((orig_nose[0], orig_nose[1]), (orig_eye1[0], orig_eye1[1]), (orig_eye2[0], orig_eye2[1])))
                left_accuracy = left_accuracy + orig_eye1[2]
                right_accuracy = right_accuracy + orig_eye2[2]
        if len(acceptable_locations) == 0:
            original_frame = put_text(original_frame, "Landmarks")
            final_images.append(original_frame)
            history.append("x")
            rem_confidence_history.append(np.nan)
            print("Skipping due to unreliable landmarks.")
            continue
        left_accuracy = left_accuracy / len(acceptable_locations)
        right_accuracy = right_accuracy / len(acceptable_locations)

        final_nose_location = (0, 0)
        final_left_eye_location = (0, 0)
        final_right_eye_location = (0, 0)
        for (nose_location, left_eye_location, right_eye_location) in acceptable_locations:
            final_nose_location = (final_nose_location[0] + nose_location[0], final_nose_location[1] + nose_location[1])
            final_left_eye_location = (final_left_eye_location[0] + left_eye_location[0], final_left_eye_location[1] + left_eye_location[1])
            final_right_eye_location = (final_right_eye_location[0] + right_eye_location[0], final_right_eye_location[1] + right_eye_location[1])
        multiplier = 1.0 / len(acceptable_locations)
        final_nose_location = (final_nose_location[0] * multiplier, final_nose_location[1] * multiplier)
        final_left_eye_location = (final_left_eye_location[0] * multiplier, final_left_eye_location[1] * multiplier)
        final_right_eye_location = (final_right_eye_location[0] * multiplier, final_right_eye_location[1] * multiplier)

        # Deal with too much movement
        final_nose = np.array([final_nose_location[0], final_nose_location[1]])
        final_eye1 = np.array([final_left_eye_location[0], final_left_eye_location[1]])
        final_eye2 = np.array([final_right_eye_location[0], final_right_eye_location[1]])
        eye_width = src.eye_extractor.get_eye_width(final_nose, final_eye1, final_eye2)
        movement_threshold = 0.5 * eye_width
        motion_skip = False
        for (nose_location, left_eye_location, right_eye_location) in acceptable_locations:
            eye1 = np.array([left_eye_location[0], left_eye_location[1]])
            eye2 = np.array([right_eye_location[0], right_eye_location[1]])
            if np.linalg.norm(final_eye1 - eye1) > movement_threshold or np.linalg.norm(final_eye2 - eye2) > movement_threshold:
                original_frame = put_text(original_frame, "Motion")
                final_images.append(original_frame)
                history.append("m")
                rem_confidence_history.append(np.nan)
                print("Skipping due to too much motion.")
                motion_skip = True
                break
        if motion_skip:
            continue

        selected_frame_indices = [i + start_offset + index for index in window_indices]

        left_visibility = 0
        right_visibility = 0
        skip_left = False
        skip_right = False

        left_sample = []
        right_sample = []
        for index in selected_frame_indices:
            frame_id_ = original_frame_ids[index]
            (left_eye_image, right_eye_image) = src.eye_extractor.get_eye_images(cv2.imread(output_directory + "frames/" + frame_id_), final_nose_location, final_left_eye_location, final_right_eye_location)
            left_eye_image = cv2.flip(left_eye_image, 1) # Flip to normalize with right eye
            left_sample.append(left_eye_image)
            right_sample.append(right_eye_image)

            # TODO: May perform slightly more accuracte cropping based on individual landmarks rather than the average here.
            Y_visibility = vmm.predict(np.array([np.array(vm.normalize_image(left_eye_image)), np.array(vm.normalize_image(right_eye_image))])[...,np.newaxis])
            if Y_visibility[0][0] < visibility_threshold:
                skip_left = True
            if Y_visibility[1][0] < visibility_threshold:
                skip_right = True
            left_visibility = left_visibility + Y_visibility[0][0]
            right_visibility = right_visibility + Y_visibility[1][0]

        if skip_left and skip_right:
            original_frame = put_text(original_frame, "Occluded")
            final_images.append(original_frame)
            history.append("x")
            rem_confidence_history.append(np.nan)
            print("Skipping due to lack of visibility")
            continue

        left_sample = [rm.normalize_image(x) for x in left_sample]
        right_sample = [rm.normalize_image(x) for x in right_sample]

        left_visibility = left_visibility / len(selected_frame_indices)
        right_visibility = right_visibility / len(selected_frame_indices)

        Y = rmm.predict(np.array([np.array(left_sample), np.array(right_sample)])[...,np.newaxis])

        if skip_left:
            take_left = False
        elif skip_right:
            take_left = True
        else:
            take_left = left_visibility > right_visibility # Take most visible eye.
        take_index = 1
        if take_left:
            take_index = 0
        eye_state = np.argmax(Y[take_index]) # Label index.
        eye_state_confidence = Y[take_index][eye_state]
        rem_confidence = 0

        (left_eye_image, right_eye_image) = src.eye_extractor.get_eye_images(cv2.imread(output_directory + "frames/" + frame_id), final_nose_location, final_left_eye_location, final_right_eye_location)
        left_eye_image = cv2.flip(left_eye_image, 1) # Flip to normalize with right eye.

        Y_openclosed = omm.predict(np.array([np.array(om.normalize_image(left_eye_image)), np.array(om.normalize_image(right_eye_image))]))
        eye_state = "c"
        if Y_openclosed[take_index][0] >= open_threshold:
            eye_state = "o"
        if eye_state_confidence >= rem_threshold:
            eye_state = eye_state + "r"

        rem_confidence = eye_state_confidence

        text = "Eye state:  " + eye_state.upper() + " " + str(max(Y[take_index]))

        if take_left:
            text = text + "\n(Left Eye) " + str(left_visibility)
        else:
            text = text + "\n(Right Eye) " + str(right_visibility)

        history.append(eye_state)
        rem_confidence_history.append(rem_confidence)

        final_image = put_text(original_frame, text)
        final_images.append(final_image)
    print_progress_bar(total_images, total_images, prefix= "Progress:", suffix="Complete",length=50)

    h, w, _ = original_frame.shape
    print("Creating video...")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter(output_directory + "annotated_video.avi", fourcc, fps, (w, h))
    print("Found " + str(len(final_images)) + " frames.")
    for im in final_images:
        out.write(im)
    out.release()

    print("Creating graph...")
    generate_graph(settings.window_seconds, history, rem_confidence_history, 1.0/fps, 1.0, output_directory + "frame_level_graph")

    print("Generating minute-level samples...")
    os.mkdir(output_directory + "samples")
    generate_states(settings.window_seconds, history, 1.0/fps, output_directory + "samples/" + id)

def put_text(frame, text):
    # Text settings.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)
    thickness = 2

    y0, dy = 150, 50
    result = frame
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        result = cv2.putText(result, line, (100, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return result

if __name__ == "__main__":
    predict_video(sys.argv[1], sys.argv[2] if sys.argv[2][-1] == "/" else sys.argv[2] + "/", sys.argv[3], sys.argv[4])
