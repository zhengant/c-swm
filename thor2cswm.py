import os
import sys
import pickle

import numpy as np
import h5py
from PIL import Image

# 0-7 cut out
ACTION_KEY = [
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
    "NoOp",
    "FillObjectwithLiquid",
    "TeleportFull",
]

TWO_TARGET_ACTIONS = set(["PutObject", "SliceObject", "FillObjectwithLiquid"])


def load_data(data_path):
    with open(data_path, "rb") as data_fp:
        data = pickle.load(data_fp)

    return data


def get_object_xy(object_id, bbox_keys, bounding_boxes, normalize=True):
    # # get name of object from object_id
    # # name is everything before the underscore
    # object_name = object_id[:object_id.index('_')]

    # # find index within bbox_keys
    # object_idx = None
    # for i, bbox_key in enumerate(bbox_keys):
    #     # name of object is everything before first '|' in bbox_key
    #     bbox_object_name = bbox_key[:bbox_key.index('|')]
    #     if bbox_object_name == object_name:
    #         object_idx = i
    #         break

    # object_id from all_obj_info_after matches format in bbox_keys
    # account for missing bounding boxes
    try:
        object_idx = bbox_keys.index(object_id)
    except ValueError:
        return None

    bounding_box = bounding_boxes[object_idx]
    # convert bounding box to coordinates of center point
    # bounding boxes are start1, start2, end1, end2
    # not sure which is x, which is y, but that's okay
    coord1 = (bounding_box[0] + bounding_box[2]) / 2
    coord2 = (bounding_box[1] + bounding_box[3]) / 2

    if normalize:  # put between 0 and 1
        coord1 /= 300
        coord2 /= 300

    return [coord1, coord2]


def extract_episode(seq_idx, data):
    action_sequence = data["action_sequence"][seq_idx]
    start = data["seq_start"][seq_idx]
    if seq_idx == len(data["seq_start"]) - 1:
        end = -1
    else:
        end = data["seq_start"][seq_idx + 1] - 1

    # format for c-swm
    episode = {"obs": [], "action": [], "next_obs": []}

    action_sequence_idx = 0
    for i in range(start, end + 1):
        before = data["obs_before"][i]["image"]
        after = data["obs_after"][i]["image"]
        bbox_keys = data["obs_before"][i]["bbox_keys"]
        bounding_boxes = data["obs_before"][i]["bounding_boxes"]
        action = (
            data["action"][i] - 8
        )  # actions are numbered 8-16, normalize so they are indices

        # # find entry in action_sequence that matches this action
        # while not action_sequence[action_sequence_idx][0] == ACTION_KEY[action]:
        #     action_sequence_idx += 1

        obj_info = data["all_obj_info_after"][i]
        if ACTION_KEY[action] in TWO_TARGET_ACTIONS:
            # with two targets, targets are index 0 and 1
            target_id1 = obj_info[0]["objectId"]
            target_id2 = obj_info[1]["objectId"]

            target1_coords = get_object_xy(target_id1, bbox_keys, bounding_boxes)
            target2_coords = get_object_xy(target_id2, bbox_keys, bounding_boxes)

            if not target1_coords or not target2_coords:
                return None

            action = np.concatenate([[action], target1_coords, target2_coords])
        else:
            # all other interactions have one target
            # with one target, index is -1
            target_id = obj_info[-1]["objectId"]
            target_coords = get_object_xy(target_id, bbox_keys, bounding_boxes)

            if not target_coords:
                return None

            # fill in 0,0 for other target
            action = np.concatenate([[action], target_coords, [0, 0]])

        # convert to RGB
        before = before[:, :, ::-1]
        after = after[:, :, ::-1]

        # resize to 50x50 (leave at 300 x 300 now)
        # before = np.array(Image.fromarray(before).resize((50, 50)))
        # after = np.array(Image.fromarray(after).resize((50, 50)))

        # normalize to be between 0 and 1
        before = before / 255
        after = after / 255

        # pytorch wants channels first
        before = np.transpose(before, (2, 0, 1))
        after = np.transpose(after, (2, 0, 1))

        episode["obs"].append(before)
        episode["action"].append(action)
        episode["next_obs"].append(after)

    return episode


def convert_data(data):
    replay_buffer = []
    for seq_idx in range(len(data["seq_start"])):
        episode = extract_episode(seq_idx, data)
        if episode and len(episode["obs"]) > 0:
            replay_buffer.append(extract_episode(seq_idx, data))

    return replay_buffer


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, "w") as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def main():
    data_path = sys.argv[1]
    out_path = sys.argv[2]
    
    data = load_data(data_path)
    replay_buffer = convert_data(data)
    save_list_dict_h5py(replay_buffer, out_path)


if __name__ == "__main__":
    main()
