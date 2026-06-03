import cv2
import torch
import numpy as np

def preprocess(rgb, depth):

    rgb = cv2.resize(rgb, (224, 224))
    depth = cv2.resize(depth, (224, 224))

    rgb = rgb.astype(np.float32) / 255.0

    depth = depth.astype(np.float32)
    depth = depth / (depth.max() + 1e-6)

    depth = np.expand_dims(depth, axis=2)

    x = np.concatenate([rgb, depth], axis=2)

    x = torch.tensor(
        x,
        dtype=torch.float32
    ).permute(2,0,1)

    return x