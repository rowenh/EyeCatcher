import os
import sys
import random
import cv2
import video_to_frames
import hrnet_api
import eye_extractor
import settings

def annotate(video_path, output_directory, id, labels, body_direction, min_sec_interval = 1):
    if len(labels) > 9:
        print("Annotations with more than 9 labels not supported.")
        return
    output_path = output_directory + "annotations_" + "+".join(labels)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print("Saving annotations to " + output_path + ".")

    print("Starting initialization.")
    print("Extracting frames.")
    video_to_frames.extract_frames(video_path, output_directory + "frames", 1/min_sec_interval, body_direction)
    original_frame_ids = sorted(os.listdir(output_directory + "frames"), key=lambda x: int(x.replace('.jpg','')))
    print("Extracting landmarks.")
    hrnet_api.generate_landmarks(output_directory)
    landmarks = eye_extractor.get_landmarks(output_directory + settings.landmarks_path, original_frame_ids)
    print("Initialization finished; ready to annotate.")

    while len(original_frame_ids) > 0:
        frame_id = original_frame_ids.pop(random.randrange(len(original_frame_ids)))
        frame = cv2.imread(output_directory + "frames/" + frame_id)
        (orig_nose, orig_eye1, orig_eye2) = landmarks[frame_id]
        (left_eye_image, right_eye_image) = eye_extractor.get_eye_images(frame, orig_nose, orig_eye1, orig_eye2)
        print("Next frame. Frame ID: " + frame_id)
        print("Options: None <SPACE>, Quit <Q>, " + ", ".join([labels[i] + " <" + str(i + 1) + ">" for i in range(len(labels))]) + ".")
        msg = process_frame(output_path + "/", id + "-frame=" + frame_id, left_eye_image if random.random() < 0.5 else right_eye_image, labels, id)
        if msg == 1:
            break

    print("Finished.")

def process_frame(dir_, file_name, frame, labels, id):
    cv2.imshow("", frame)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 32 or key == 13:
        return 0
    if key == ord("q") or key == ord("Q"):
        return 1
    for i in range(len(labels)):
        if key == ord(str(i + 1)):
            cv2.imwrite(dir_ + labels[i] + "_" + file_name.split(".")[0] + ".png", frame)
            print("Labeled as " + labels[i])
            return 0
    print("Unexpected input, choose again...")
    process_frame(dir_, file_name, frame, labels, id)

if __name__ == "__main__":
    annotate(sys.argv[1], sys.argv[2] if sys.argv[2][-1] == "/" else sys.argv[2] + "/", sys.argv[3], sys.argv[4].split("+"), sys.argv[5])
