import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to parse each event file
def parse_event_file(keys, event_file):
    data = {
        k: [] for k in keys
    }
    for example in tf.compat.v1.train.summary_iterator(event_file):
        for value in example.summary.value:
            if value.tag in data.keys():
                data[value.tag].append((example.step, value.simple_value))
    return data

# Loop through each subdirectory in the base log directory
def gather_data(base_logdir, keys):
    all_data = {k: {} for k in keys}

    # Load data from file
    for sub_dir in os.listdir(base_logdir):
        full_path = os.path.join(base_logdir, sub_dir)
        # Parse the filename to get system size
        size = float(full_path.split('_')[2])

        if os.path.isdir(full_path):
            event_files = [file for file in os.listdir(full_path) if 'tfevents' in file]
            for event_file in event_files:
                event_file_path = os.path.join(full_path, event_file)
                data = parse_event_file(keys, event_file_path)

        try:
            for k in data.keys():
                all_data[k][size].append(data[k])
        except KeyError:
            for k in data.keys():
                all_data[k][size] = [data[k]]

    # Organize into np.arrays
    for k in keys:
        for s in all_data[k].keys():
            num_exps = len(all_data[k][s])
            steps = np.array(all_data[k][s][0]).T[0]
            values = np.vstack(
                [np.array(all_data[k][s][i]).T[1] for i in range(num_exps)]
            )
            all_data[k][s] = (steps, values)

    return all_data


def plot_single(data, legend, omit=1):
    values = data[1][:, :-omit]
    med_values = np.median(values, axis=0)
    percentile_low = np.percentile(values, 25, axis=0)
    percentile_high = np.percentile(values, 75, axis=0)
    xticks = data[0][:-omit] / 20
    print(med_values[-1], xticks[-1])

    plt.plot(xticks, med_values, label=legend)
    plt.fill_between(xticks, percentile_low, percentile_high, alpha=0.2)
    plt.legend()


def plot(data, var_name, omit=1):
    """ Take in data of a particular tag as a dictionary of the format
    {sizes: (x_tick, values)}
    """
    for s in sorted(data.keys()):
        plot_single(data[s], r'${}={}$'.format(var_name, int(s)), omit=omit)
