import os

from BlastAssist.managedata.util import read_images_for_torch
from BlastAssist.managedata.localfolders import embryovision_folder
from BlastAssist.predictor import Predictor, load_classifier
from BlastAssist.managedata.managedata import FilenameParser


class ZonaPredictor(Predictor):
    loadname = os.path.join(embryovision_folder, 'zona', 'zonaclassifier.pkl')
    load_network = staticmethod(load_classifier)

    def _predict(self, filenames):
        images = read_images_for_torch(filenames).to(self.device)
        labels = self.network.predict(images).astype('uint8')
        infos = [FilenameParser.get_imageinfo_from_filename(nm)
                 for nm in filenames]
        return self.pack_into_annotated_images(infos, labels)
