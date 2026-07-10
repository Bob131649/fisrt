import os

import h5py
import numpy as np
from PIL import Image


def cache_path(dataset_path, image_size):
    h, w = image_size
    return f"{dataset_path}.train_replay_{h}x{w}.zarr"


def process_image(image, image_size, rgb_crop):
    image = np.asarray(image)
    top, bottom, left, right = rgb_crop
    if image.ndim == 3 and image.shape[0] >= bottom and image.shape[1] >= right:
        image = image[top:bottom, left:right, :3]
    if image.shape[:2] != image_size:
        image = Image.fromarray(image[..., :3]).resize(
            (image_size[1], image_size[0]),
            resample=Image.BILINEAR,
        )
        image = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(image[..., :3])


def rgb_key(data):
    if "rgb" in data:
        return "rgb"
    if "images" in data:
        return "images"
    raise KeyError(f"Dataset has no rgb/images key: {data.filename}")


def valid_rgb_cache(root, image_size):
    if "data" not in root or "rgb" not in root["data"]:
        return False
    rgb = root["data"]["rgb"]
    return tuple(rgb.shape[1:]) == (image_size[0], image_size[1], 3)


def ensure_rgb_cache(dataset_path, image_size, rgb_crop):
    import zarr

    path = cache_path(dataset_path, image_size)
    if os.path.exists(path):
        root = zarr.open_group(path, mode="r")
        if valid_rgb_cache(root, image_size):
            return path
        print(f"rebuild incomplete rgb cache: {path}")

    with h5py.File(dataset_path, "r") as f:
        raw_images = f[rgb_key(f)]
        root = zarr.open_group(path, mode="w")
        root.attrs["raw_rgb_shape"] = tuple(raw_images.shape)
        root.attrs["bad_indices"] = []
        data = root.require_group("data")
        chunk = min(128, len(raw_images))
        rgb = data.create_dataset(
            "rgb",
            shape=(len(raw_images), image_size[0], image_size[1], 3),
            chunks=(chunk, image_size[0], image_size[1], 3),
            dtype="uint8",
            compressor=None,
        )

        for start in range(0, len(raw_images), chunk):
            end = min(start + chunk, len(raw_images))
            batch = [process_image(image, image_size, rgb_crop) for image in raw_images[start:end]]
            rgb[start:end] = np.asarray(batch, dtype=np.uint8)

    return path


def cached_bad_indices(path):
    import zarr

    root = zarr.open_group(path, mode="r")
    return set(int(index) for index in root.attrs.get("bad_indices", []))


def open_cached_rgb(path):
    import zarr

    return zarr.open_group(path, mode="r")["data"]["rgb"]
