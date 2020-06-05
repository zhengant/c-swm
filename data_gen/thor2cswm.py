import os
import sys
import pickle

import numpy as np
import h5py
from tqdm import trange
from PIL import Image
from sklearn.model_selection import train_test_split

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
    episode_masks = []

    action_sequence_idx = 0
    for i in range(start, end + 1):
        before = data["obs_before"][i]["image"]
        after = data["obs_after"][i]["image"]
        bbox_keys = data["obs_before"][i]["bbox_keys"]
        bounding_boxes = data["obs_before"][i]["bounding_boxes"]
        action = (
            data["action"][i]["action"] - 8
        )  # actions are numbered 8-16, normalize so they are indices

        # # find entry in action_sequence that matches this action
        # while not action_sequence[action_sequence_idx][0] == ACTION_KEY[action]:
        #     action_sequence_idx += 1

        obj_info = data["all_obj_info_after"][i]
        if ACTION_KEY[action] in TWO_TARGET_ACTIONS:
            # with two targets, targets are index 0 and 1
            target_id1 = obj_info[0]["name"]  # obj_info[0]["objectId"]
            target_id2 = obj_info[1]["name"]  # obj_info[1]["objectId"]

            target1_coords = get_object_xy(target_id1, bbox_keys, bounding_boxes)
            target2_coords = get_object_xy(target_id2, bbox_keys, bounding_boxes)

            if not target1_coords or not target2_coords:
                return None, None

            action = np.concatenate([[action], target1_coords, target2_coords])
        else:
            # all other interactions have one target
            # with one target, index is -1
            target_id = obj_info[-1]["name"]
            target_coords = get_object_xy(target_id, bbox_keys, bounding_boxes)

            if not target_coords:
                return None, None

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

        # extract instance masks
        masks = []
        for _, mask in data["instance_masks"][i].items():
            masks.append(np.expand_dims(mask, 0))  # make masks have shape 1x300x300
        masks = np.concatenate(masks, axis=0)

        episode["obs"].append(before)
        episode["action"].append(action)
        episode["next_obs"].append(after)
        episode_masks.append(masks)

    return episode, episode_masks


def convert_data(data):
    replay_buffer = []
    all_masks = []
    for seq_idx in trange(len(data["seq_start"])):
        episode, masks = extract_episode(seq_idx, data)
        if episode and len(episode["obs"]) > 0:
            replay_buffer.append(episode)
            all_masks.append(masks)

    return replay_buffer, all_masks


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
    data_name = sys.argv[1]
    data_path = sys.argv[2]
    out_dir = sys.argv[3]

    train_path = os.path.join(out_dir, data_name + "_train.h5")
    test_path = os.path.join(out_dir, data_name + "_eval.h5")
    train_masks_path = os.path.join(out_dir, data_name + "_masks_train.pkl")
    test_masks_path = os.path.join(out_dir, data_name + "_masks_eval.pkl")

    data = load_data(data_path)
    replay_buffer, all_masks = convert_data(data)

    print(len(replay_buffer))
    train_buffer, test_buffer, train_masks, test_masks = train_test_split(
        replay_buffer, all_masks, test_size=0.2
    )

    save_list_dict_h5py(train_buffer, train_path)
    save_list_dict_h5py(test_buffer, test_path)

    with open(train_masks_path, 'wb') as train_masks_fp:
        pickle.dump(train_masks, train_masks_fp)
    with open(test_masks_path, 'wb') as test_masks_fp:
        pickle.dump(test_masks, test_masks_fp)


if __name__ == "__main__":
    main()
