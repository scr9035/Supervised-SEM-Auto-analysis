57# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import HoleProperty
from ...analysis import GeneralProcess
from ...analysis import holes
import numpy as np

from PyQt5.QtCore import pyqtSignal

class CHCap(HoleProperty):
    
    data_transfer_sig = pyqtSignal(list, list, dict)
    _lvl_name=['Missing', 'Total']
    information = ('Measurement: Channel hole capping count',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__()
        self._auto_holes = self._AutoCapping
        self._show_grid = True
        self._open_limit = 40
        self._angle_diff = 1.5
        self.data = {}
        self._area_thres = 0.01
        
        self._patch_handle_prop = dict(marker='X', markersize=20, color='r', mfc='r', 
                                 ls='none', alpha=1, visible=True)
    
    def _update_plugin(self):
        self._on_new_image(self._image)
        super()._update_plugin()
    
    def _AutoCapping(self, image, scale=0.6, iteration=2,):
        
        self.data = {'Missing' : 0,
                     'Total' : 0}
        threshold = GeneralProcess.GaussianMixThres(image[350:370, :], components=2, scale=scale)
         
        bi_fig_fit = GeneralProcess.BinaryConverter(image, thres=threshold, scale=scale, 
                                     iteration=iteration)
            
        open_count, cap_count, tot_count, miss_point, lines = \
            holes.GridMatching(bi_fig_fit, grid='rect', open_limit=self._open_limit, angle_diff=self._angle_diff)
        
        self.data['Missing'] = [cap_count]
        self.data['Total'] = [tot_count]
        return cap_count, miss_point, np.zeros(cap_count), np.zeros(cap_count), np.zeros(cap_count)
    
    
    def data_transfer(self):
        """Function override to transfer raw data to measurement data """
#        raw_data = np.transpose(np.array(self._cd_data, dtype=np.float)) * self._calib

#        for i in range(self._lvl_count):
#            self.data[self._lvl_name[i]] = raw_data[i]
        
        hori_header = [' '] if len(self.data) > 0 else []
        self.data_transfer_sig.emit(self._lvl_name, hori_header, self.data)