import os
import sys

import numpy as np

from BlastAssist.managedata.localfolders import embryovision_folder
from BlastAssist.managedata.managedata import AnnotatedDataset, AnnotatedImage

CONFIDENCE_THRESHOLD = 0.985

class PredictedSymmetryMeasurer(object):
    loadfolder = embryovision_folder
    confidence_threshold = CONFIDENCE_THRESHOLD

    def __init__(self, embryo):
        self.embryo = embryo

        self._stage = self._load_stage(embryo)
        self._blastomeres = self._load_blastomeres(embryo)

    def measure_ncell_symmetry(self, n):
        detections = self._grab_valid_ncell_detections(n)
        return [self._measure_symmetry(d) for d in detections]

    def _measure_symmetry(self, detection):
        symmetry = measure_symmetry(*detection.annotation)
        return AnnotatedImage(detection.info, symmetry)

    def _grab_valid_ncell_detections(self, n):
        polygons = list()
        for i in self._blastomeres.iterate_over_images():
            these_polygons = self._grab_confident_detections(i)
            valid = (
                n == self._stage[i.info].annotation and  # stage says n-cell
                n == len(these_polygons))  # n detections
            if valid:
                polygons.append(AnnotatedImage(i.info, these_polygons))
        return polygons

    def _grab_confident_detections(self, image):
        these_polygons = [
            d['xy_polygon']
            for d in image.annotation
            if d['confidence'] > self.confidence_threshold]
        return these_polygons

    def _load_stage(self, embryo):
        loadname = os.path.join(
            self.loadfolder,
            embryo.slide,
            self._get_wellname(embryo.well),
            'stage_smooth.pkl')
        return AnnotatedDataset.load_from_pickle(loadname)

    def _load_blastomeres(self, embryo):
        loadname = os.path.join(
            self.loadfolder,
            embryo.slide,
            self._get_wellname(embryo.well),
            'blastomeres.pkl')
        return AnnotatedDataset.load_from_pickle(loadname)

    def _get_wellname(self, wellnumber):
        return "WELL" + str(wellnumber).rjust(2, '0')


def measure_symmetry(*xy_polygons):
    areas = [get_area(p) for p in xy_polygons]
    return np.std(areas) / np.mean(areas)


def get_area(polygon):
    x, y = np.transpose(polygon)
    shoelace = x * np.roll(y, 1) - np.roll(x, 1) * y
    return 0.5 * np.abs(shoelace.sum())
