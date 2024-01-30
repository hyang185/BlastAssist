import io
import os

import numpy as np
import torch

from BlastAssist.managedata.util import load_and_crop_image, augment_focus
from BlastAssist.managedata.localfolders import embryovision_folder
from BlastAssist.predictor import Predictor, load_classifier
from BlastAssist.managedata.managedata import FilenameParser
from BlastAssist.blastocyst.load_images import load_images_for_blastocyst_segmentation

class BlastSegment(Predictor):
    loadname = os.path.join(
        embryovision_folder, 'blastocyst', 'blastocystclassifier.pkl')
    load_network = staticmethod(load_classifier)
    input_shape = (200, 200)

    def _predict(self, filenames):
        inputs_x = load_images_for_blastocyst_segmentation(filenames)
        inputs_x = inputs_x.to(self.device, dtype=torch.float32)
        blast_labels = self.network.predict(inputs_x)
        return blast_labels
