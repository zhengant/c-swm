import argparse
import torch
import utils
import os
import pickle


from torch.utils import data
import numpy as np
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import modules

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-folder", type=str, default="checkpoints", help="Path to checkpoints."
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1,
        help="Number of prediction steps to evaluate.",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/shapes_eval.h5", help="Dataset string."
    )
    parser.add_argument(
        "--masks", type=str, default="data/thor_masks_eval.h5", help="Path to object masks"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable CUDA training."
    )

    args_eval = parser.parse_args()

    meta_file = os.path.join(args_eval.save_folder, "metadata.pkl")
    model_file = os.path.join(args_eval.save_folder, "model.pt")

    args = pickle.load(open(meta_file, "rb"))["args"]

    args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
    args.batch_size = 100
    args.dataset = args_eval.dataset
    args.seed = 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    dataset = utils.PathDataset(hdf5_file=args.dataset, path_length=args_eval.num_steps)
    eval_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Get data sample
    data_iter = eval_loader.__iter__()
    batch = data_iter.next()
    obs = batch[0]
    input_shape = obs[0][0].size()

    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_objects=args.num_objects,
        target_object_dim=args.target_object_dim,
        action_embed_dim=args.action_embed_dim,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder,
    ).to(device)

    model.load_state_dict(torch.load(model_file))
    model.eval()

    output_dir = os.path.join("outputs", args.name)
    os.makedirs(output_dir, exist_ok=True)

    sample_idx = 0

    for data_batch in eval_loader:
        data_batch = [[t.to(device) for t in tensor] for tensor in data_batch]
        observations, actions = data_batch

        obs = observations[0]
        all_feature_maps = model.obj_extractor(obs)
        num_masks = list(all_feature_maps.shape)[1]
        batch_size = list(all_feature_maps.shape)[0]

        for i in range(batch_size):
            sample_dir = os.path.join(output_dir, str(sample_idx).zfill(4))
            os.makedirs(sample_dir, exist_ok=True)

            # save masks
            for mask_idx in range(num_masks):
                feature_map = all_feature_maps[i, mask_idx, :, :].cpu().detach().numpy()
                mask_path = os.path.join(sample_dir, 'mask' + str(mask_idx).zfill(4) + '.png')

                plt.imshow(feature_map, cmap='Greys_r')
                plt.savefig(mask_path)

            # save original and resized images
            original_image = (np.array(obs[i].cpu()) * 255).transpose((1, 2, 0)).astype(np.int32)
            # resized_image = np.array(Image.fromarray(original_image).resize((10, 10)))

            original_image_path = os.path.join(sample_dir, 'original.png')
            # resized_image_path = os.path.join(sample_dir, 'resized.png')

            plt.imshow(original_image)
            plt.savefig(original_image_path)

            sample_idx += 1

            # plt.imshow(resized_image)
            # plt.savefig(resized_image_path)
