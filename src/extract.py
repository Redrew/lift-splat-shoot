"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import matplotlib as mpl
import torch

mpl.use("Agg")
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from .data import compile_data
from .models import compile_model


def extract_model_preds(
    version,
    modelf,
    dataroot="/data/nuscenes",
    output_path="model_predictions.pkl",
    gpuid=1,
    segmentation=False,
    H=900,
    W=1600,
    resize_lim=(0.193, 0.225),
    final_dim=(128, 352),
    bot_pct_lim=(0.0, 0.22),
    rot_lim=(-5.4, 5.4),
    rand_flip=True,
    xbound=[-50.0, 50.0, 0.5],
    ybound=[-50.0, 50.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[4.0, 45.0, 1.0],
    bsz=4,
    nworkers=10,
):
    grid_conf = {
        "xbound": xbound,
        "ybound": ybound,
        "zbound": zbound,
        "dbound": dbound,
    }
    cams = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ]
    data_aug_conf = {
        "resize_lim": resize_lim,
        "final_dim": final_dim,
        "rot_lim": rot_lim,
        "H": H,
        "W": W,
        "rand_flip": rand_flip,
        "bot_pct_lim": bot_pct_lim,
        "cams": cams,
        "Ncams": 5,
    }
    _, valloader = compile_data(
        version,
        dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=bsz,
        nworkers=nworkers,
        parser_name="noaugsegmentationdata",
    )

    device = torch.device("cpu") if gpuid < 0 else torch.device(f"cuda:{gpuid}")

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print("loading", modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    # get frames lookup
    with open(
        "/home/ashen3/autonomy-stack/dataset/nuscenes-train/labels.pkl", "rb"
    ) as f:
        train_labels = pickle.load(f)
    with open("/home/ashen3/autonomy-stack/dataset/nuscenes-val/labels.pkl", "rb") as f:
        val_labels = pickle.load(f)
    labels = {**train_labels, **val_labels}
    frames_lookup = {}
    for seq_id, frames in labels.items():
        for frame in frames:
            frames_lookup[f"{seq_id}:{frame['timestamp_ns']}"] = frame

    model.eval()
    # remove last classification layer
    if not segmentation:
        model.bevencode.up2[-1] = nn.Identity()
    if False:
        counter = 0
        predictions = {}
        with torch.no_grad():
            for batchi, (
                imgs,
                rots,
                trans,
                intrins,
                post_rots,
                post_trans,
                binimgs,
            ) in tqdm(enumerate(valloader), total=len(valloader)):
                out = model(
                    imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                )
                out = out.cpu()  # B, 128, 200, 200

                for si in range(imgs.shape[0]):
                    rec = valloader.dataset.ixes[counter]
                    feature = out[si]
                    seq_id = rec["scene_token"]
                    timestamp = rec["timestamp"] * 1000
                    frame = frames_lookup[f"{seq_id}:{timestamp}"]
                    rot = np.linalg.inv(
                        Quaternion(frame["ego2global_rotation"]).rotation_matrix
                    )
                    ego_translation = (
                        frame["translation"] - frame["ego2global_translation"]
                    ) @ rot.T
                    bin_translation = ego_translation * 2 + 100
                    # get feature at center coord
                    centers = torch.LongTensor(np.round(bin_translation)[:, :2])
                    in_range = torch.all((centers >= 0) & (centers < 200), axis=1)
                    centers, track_ids = (
                        centers[in_range],
                        frame["track_id"][in_range.numpy()],
                    )
                    center_features = feature.permute(1, 2, 0)[
                        centers[:, 0], centers[:, 1]
                    ]
                    for center_feature, track_id in zip(center_features, track_ids):
                        predictions[f"{seq_id}:{timestamp}:{track_id}"] = center_feature

                    counter += 1
        with open(f"center_features.pkl", "wb") as f:
            pickle.dump(predictions, f)
    else:
        counter = 0
        predictions = []
        with torch.no_grad():
            for batchi, (
                imgs,
                rots,
                trans,
                intrins,
                post_rots,
                post_trans,
                binimgs,
            ) in tqdm(enumerate(valloader), total=len(valloader)):
                out = model(
                    imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                )
                out = out.cpu()  # B, 128, 200, 200

                for si in range(imgs.shape[0]):
                    rec = valloader.dataset.ixes[counter]
                    counter += 1
                    predictions.append({"rec": rec, "feature_map": out[si]})
                index = batchi + 1
                if index % 1000 == 0 or index == len(valloader):
                    with open(
                        f"/data/ashen3/predictions/lift-splat-shoot/{index}.pkl", "wb"
                    ) as f:
                        pickle.dump(predictions, f)
                    predictions.clear()
