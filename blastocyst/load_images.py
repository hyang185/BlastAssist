import numpy as np
import torch
import torchvision.transforms.functional as transform
from PIL import Image


def load_images_for_blastocyst_segmentation(image_filenames):
    """
    Parameters
    ----------
    image_filenames : iterable of filename-like objects
        The image filenames must point to pre-cropped images (350, 350), and
        reshaped to (200, 200) and centered on the zona using attentionbox.

    Returns
    -------
    torch.Tensor, shape (?, 3, 200, 200)
        Tensor of the shape, normalization, and data type required for
        the fragmentation classifier
    """
    inputs = []
    for filename in image_filenames:
        image = Image.open(filename).convert('RGB')
        image_tensor = transform.to_tensor(image)
        image_norm = transform.normalize(
            image_tensor,
            (0.5, 0.5, 0.5),
            (0.25, 0.25, 0.25))
        inputs.append(image_norm)
    return torch.stack(inputs,0)
