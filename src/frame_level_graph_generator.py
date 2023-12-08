import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

def get_multimode(lst):
    result = []
    lst_ = Counter(lst)
    tmp = lst_.most_common(1)[0][1]
    for e in lst:
        if lst.count(e) == tmp:
            result.append(e)
    return list(set(result))

# Return list of candidates to be picked from (prioritizing REM later)
def decide_state(pool):
    rems = [p for p in pool if p.endswith("r")]
    if len(rems) == 0:
        return get_multimode(pool)
    return get_multimode(rems)

def generate_graph(window_time, eye_states, rem_confidences, state_interval, output_interval, output_path):
    t = 0
    x = window_time / 2.0 + output_interval * -0.5 # Plot at the center of intervals
    X = []
    Y = []
    lines = []
    pool = []
    X_rem = []
    Y_rem = []
    if output_interval < state_interval:
        output_interval = state_interval
    for s in eye_states:
        pool.append(s)
        if t >= output_interval:
            most_frequent_state = decide_state(pool)
            rem_candidate = False
            for item in most_frequent_state:
                if item.endswith("r"):
                    rem_candidate = True
            if rem_candidate:
                most_frequent_state = [s for s in most_frequent_state if s.endswith("r")]
            most_frequent_state = random.choice(most_frequent_state)

            pool.clear()
            t = t - output_interval
            if most_frequent_state[0] == "o":
                X.append(x)
                Y.append(1)
            if most_frequent_state[0] == "c":
                X.append(x)
                Y.append(0)
            if most_frequent_state[0] == "x" or most_frequent_state[0] == "m":
                X.append(x)
                Y.append(np.nan)
            if most_frequent_state.endswith("r"):
                lines.append(x)
        t = t + state_interval
        x = x + state_interval

    plt.clf()
    plt.plot(X, Y)
    plt.vlines(x=lines, ymin=-1, ymax=2, colors='purple', ls='--', lw=1, label='REM')
    plt.legend()
    plt.axis([0, state_interval * len(eye_states) + window_time, -1, 2])
    plt.yticks([-1, 0, 1, 2], ['', 'Closed', 'Opened', ''])
    plt.xlabel('Timestamp (in seconds)')
    plt.savefig(output_path + ".png")

    x_rem = window_time / 2.0 + output_interval * -0.5 # Plot at the center of intervals
    for _ in rem_confidences:
        X_rem.append(x_rem)
        x_rem = x_rem + state_interval
    Y_rem = rem_confidences

    plt.clf()
    plt.plot(X_rem, Y_rem)
    plt.axis([0, state_interval * len(eye_states) + window_time, 0, 1])
    plt.savefig(output_path + "_REM.png")
