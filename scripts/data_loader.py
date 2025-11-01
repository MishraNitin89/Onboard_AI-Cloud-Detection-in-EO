import rasterio
import numpy as np
import os

def load_tiff(path):
    with rasterio.open(path) as src:
        img = src.read()
        return np.moveaxis(img, 0, -1)

def normalize_image(img):
    return img / 10000.0  # Sentinel-2 scaling

def load_dataset(img_dir, mask_dir):
    images, masks = [], []
    for fname in os.listdir(img_dir):
        if fname.endswith('.tif'):
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, fname.replace('.tif', '_mask.tif'))
            if os.path.exists(mask_path):
                images.append(normalize_image(load_tiff(img_path)))
                masks.append(load_tiff(mask_path)[..., 0])
    return np.array(images), np.array(masks)
