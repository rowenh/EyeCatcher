import pathlib
import sys
sys.path.append(pathlib.Path(__file__).parent.resolve())
import settings

def generate_states(window_time, eye_states, state_interval, output_path):
    annotation_window = settings.sleep_window
    x = window_time / 2.0
    state_counter = {}
    annotation_id = 0
    for s in eye_states:
        state_counter[s] = state_counter.get(s, 0) + 1.0
        if (x % annotation_window) > ((x + state_interval) % annotation_window):
            write_annotation_state(state_counter, output_path + "_" + str(int(annotation_window)) + "_" + str(annotation_id))
            state_counter.clear()
            annotation_id = annotation_id + 1
        x = x + state_interval
    write_annotation_state(state_counter, output_path + "_" + str(int(annotation_window)) + "_" + str(annotation_id))

def write_annotation_state(state_counter, output_path):
    f = open(output_path, "w+")

    f.write("<<<LABELS>>>")
    f.write("\n???")

    f.write("\n<<<INPUT>>>")
    for k in state_counter.keys():
        accepted_states = state_counter.get("c", 0) + state_counter.get("o", 0) + state_counter.get("cr", 0) + state_counter.get("or", 0)
        if k == "m":
            accepted_states = accepted_states + state_counter.get(k, 0)
        if k == "x":
            accepted_states = accepted_states + state_counter.get(k, 0) + state_counter.get("m", 0)

        if accepted_states == 0:
            continue
        f.write("\n" + k + " " + str(state_counter.get(k, 0) / accepted_states))

    f.close()
