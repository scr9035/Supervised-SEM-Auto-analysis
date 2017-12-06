57# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import HoleProperty
from ...analysis import GeneralProcess
from ...analysis import holes
import math
import numpy as np

from PyQt5.QtCore import pyqtSignal

class CHDimple(HoleProperty):
    
    data_transfer_sig = pyqtSignal(list, list, dict)
    calib_dict = {'100K' : 1.116503,
                  '50K' : 2.233027,
                  '20K' : 5.582623,
                  '12K' : 9.304371,
                  '10K' : 11.16579}
    _lvl_name=['area', 'eccentricity']
    information = ('Measurement: Channel hole dimple analysis',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__()
        self._auto_holes = self._AutoDimple
        self.data = {}
        self._area_thres = 0.01
        self._mag = '100K'
        self._calib = self.calib_dict[self._mag]
    
    def _update_plugin(self):
        self._on_new_image(self._image)
        super()._update_plugin()
    
    def _AutoDimple(self, image):
        bi_fig = GeneralProcess.BinaryConverter(image, thres='otsu', scale=1)
        
        regions = holes.TopGeometryAnalysis(bi_fig)
        
        self.data = {'area' : [],
                            'major' : [],
                            'minor' : [],
                            'orientation' : [],
                            'center' : [],
                            'eccentricity' : [],
                            'count' : 0}
        center = []
        width = []
        height = []
        angle = []
        max_row, max_col = image.shape
        for i in range(len(regions)):
            prop = regions[i]
            minr, minc, maxr, maxc = prop.bbox
            if minc <= 1 or minr <= 1 or maxc >= max_col-2 or maxr >= max_row-2:
                continue
            if prop.area < max_row * max_col * self._area_thres / 100.0:
                continue
            y0, x0 = prop.centroid
            self.data['area'].append(prop.area)
            self.data['major'].append(prop.major_axis_length)
            self.data['minor'].append(prop.minor_axis_length)
            self.data['orientation'].append(prop.orientation) # Unit is radius, clock wise
            self.data['center'].append([x0, y0])
            self.data['eccentricity'].append(prop.eccentricity)
            self.data['count'] += 1

            orientation = prop.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * prop.major_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * prop.major_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * prop.minor_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * prop.minor_axis_length
            major = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            minor = np.sqrt((x2 - x0)**2 + (y2 - y0)**2)
            
            center.append([x0, y0])
            width.append(2*major)
            height.append(2*minor)
            angle.append((2*np.pi - orientation) * 180/np.pi)     
        return self.data['count'], center, width, height, angle
    
    
    def data_transfer(self):
        """Function override to transfer raw data to measurement data """
#        raw_data = np.transpose(np.array(self._cd_data, dtype=np.float)) * self._calib

#        for i in range(self._lvl_count):
#            self.data[self._lvl_name[i]] = raw_data[i]
        hori_header = ['Hole %i' %n for n in range(1,self._count+1)]
        self.data_transfer_sig.emit(self._lvl_name, hori_header, self.data)