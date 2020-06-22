import numpy as np
import open3d


# def _label_to_colors(labels):
#     map_label_to_color = {
#         0: [255, 255, 255],  # white
#         1: [0, 0, 255],  # blue
#         2: [128, 0, 0],  # maroon
#         3: [255, 0, 255],  # fuchsia
#         4: [0, 128, 0],  # green
#         5: [0, 0, 128],  # navy
#         6: [128, 0, 128],  # purple
#         7: [255, 0, 0],  # red
#         8: [128, 128, 0],  # olive
#         9: [0, 128, 128]
#     }
#     return np.array([map_label_to_color[label] for label in labels]).astype(np.int32)


def _label_to_colors_by_name(labels, name):
    if name == 'semantic':
        map_label_to_color = {
            0: [255, 255, 255],  # white
            1: [0, 0, 255],
            2: [128, 0, 128],
            3: [0, 128, 128],
            4: [0, 128, 0],
            5: [128, 0, 0],
            6: [255, 0, 255],
            7: [255, 0, 0],
            8: [128, 128, 0]
        }
    elif name == 'npm':
        map_label_to_color = {
            0: [255, 255, 255],  # white
            1: [0, 0, 255],  # blue
            2: [128, 0, 0],  # maroon
            3: [255, 0, 255],  # fuchsia
            4: [0, 128, 0],  # green
            5: [0, 0, 128],  # navy
            6: [128, 0, 128],  # purple
            7: [255, 0, 0],  # red
            8: [128, 128, 0],  # olive
            9: [0, 128, 128]
        }
    elif name == 'common':
        map_label_to_color = {
            0: [255, 255, 255],  # white
            1: [0, 0, 255],  # blue
            2: [0, 128, 128],
            3: [128, 0, 0],
            4: [255, 0, 255],
            5: [128, 128, 0]
        }
    else:
        raise ValueError
    return np.array([map_label_to_color[label] for label in labels]).astype(np.int32)


# def common_label_to_colors(labels):
#     map_label_to_color = {
#         0: [255, 255, 255],  # white
#         1: [0, 0, 255],  # blue
#         2: [0, 255, 255],
#         3: [255, 0, 0],
#         4: [255, 0, 255],
#         5: [255, 255, 0]
#     }
#     return np.array([map_label_to_color[label] for label in labels]).astype(np.int32)


def _label_to_colors_one_hot(labels):
    map_label_to_color = np.array(
        [
            [255, 255, 255],
            [0, 0, 255],
            [128, 0, 0],
            [255, 0, 255],
            [0, 128, 0],
            [255, 0, 0],
            [128, 0, 128],
            [0, 0, 128],
            [128, 128, 0],
        ]
    )
    num_labels = len(labels)
    labels_one_hot = np.zeros((num_labels, 9))
    labels_one_hot[np.arange(num_labels), labels] = 1
    return np.dot(labels_one_hot, map_label_to_color).astype(np.int32)


def load_labels(label_path):
    # Assuming each line is a valid int
    with open(label_path, "r") as f:
        labels = [int(line) for line in f]
    return np.array(labels, dtype=np.int32)

mn
def write_labels(label_path, labels):
    with open(label_path, "w") as f:
        for label in labels:
            f.write("%d\n" % label)
