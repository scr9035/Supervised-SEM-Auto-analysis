# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import Boundary
from ...analysis import GeneralProcess, channel
import numpy as np
from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal

class FinDepth(Boundary):
    
    data_transfer_sig = pyqtSignal(list, list, dict)
    calib_dict = {'500k' : 0.248047,
                  '100K' : 1.116503,
                  '50K' : 2.233027,
                  '20K' : 5.582623,
                  '12K' : 9.304371,
                  '10K' : 11.16579}
    name = 'Fin CD and edges'
    _lvl_name=['Depth']
    information = ('Depth between Fins',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__(mode='Vertical')
        self._auto_boundary = self._AutoFinBound
        self._auto_CD = self._FinStemAutoDepth
        self.data = {}
        self._mag = '500k'
        self._calib = self.calib_dict[self._mag]
        
        self._threshold = 1
        self._scan_to_avg = 3
        
        self._show_boundary = False
        self._show_profile = True
        
        self._extra_control_widget.append(QLabel('Magnification:'))
        self._choose_mag = QComboBox()
        for key in self.calib_dict.keys():
            self._choose_mag.addItem(key)
        self._choose_mag.setCurrentText(self._mag)
        self._choose_mag.activated[str].connect(self._set_mag)
        self._extra_control_widget.append(self._choose_mag)
        
        self._extra_control_widget.append(QLabel('Threshold (%):'))
        self._input_thres = QLineEdit()
        self._input_thres.setText(str(self._threshold))
        self._input_thres.editingFinished.connect(self._change_thres)
        self._extra_control_widget.append(self._input_thres)
        
    def _update_plugin(self):
        self._on_new_image(self._image)
        super()._update_plugin()
    
    def _set_mag(self, magnification):
        self._mag = magnification
        self._calib = self.calib_dict[self._mag]
        self._update_plugin()
        
    def _change_thres(self):
        try:
            thres = float(self._input_thres.text())
            if thres >= 100:
                thres = 99
                self._input_thres.setText(str(thres))
            if thres <= 0:
                thres = 1
                self._input_thres.setText(str(thres))
            self._threshold = thres           
            self._update_plugin()
        except:
            return
        
    def data_transfer(self):
        """Function override to transfer raw data to measurement data """
        raw_data = np.transpose(self._cd_data) * self._calib
        
        for i in range(self._lvl_count):
            self.data[self._lvl_name[i]] = raw_data[i]
        hori_header = ['Ch %i' %n for n in range(1,self._channel_count+1)]
        self.data_transfer_sig.emit(self._lvl_name, hori_header, self.data)
    
    def _AutoFinBound(self, image, ref_high_pts=None, ref_low_pts=None):
        
        y_lim, x_lim = image.shape     
        if ref_high_pts is None or ref_low_pts is None:
            ref_low_y = channel.IntensInterface(image, ref_range=[10,np.Inf])
            ref_high_y = channel.IntensInterface(image, ref_range=[10,int(y_lim/2)])
        else:
            ref_low_y = int((ref_low_pts[0][1] + ref_low_pts[1][1])/2)
            ref_high_y = int((ref_high_pts[0][1] + ref_high_pts[1][1])/2)
        
        if ref_high_y == ref_low_y:
                    ref_high_y -= 20
                    ref_low_y += 20
        channel_count, channel_center, plateau = channel.VertiChannel(image, 0, 
                                                                      target='dark')
        objs_boundary = [None for _ in range(channel_count)]
        objs_center = [None for _ in range(channel_count)]
        for i in range(channel_count):
            left = int(plateau[i])
            right = int(plateau[i+1])
            obj = image[ref_high_y:ref_low_y, left:right]
            thres = GeneralProcess.GaussianMixThres(obj)
            bi_obj = GeneralProcess.BinaryConverter(obj, thres=thres)
            obj_bound = GeneralProcess.BinaryDialateEdge(bi_obj)
            coord = np.nonzero(obj_bound)
            y, x = coord
            y = y + ref_high_y
            x = x + left
            objs_boundary[i] = [x, y]
            objs_center[i] = [np.mean(x), np.mean(y)]
        return channel_count, ref_high_y, ref_low_y, objs_center, objs_boundary
      
    def _FinStemAutoDepth(self, image, interface=None):
        y_lim, x_lim = image.shape
        if interface is None:
            ref_range = [int(y_lim/10), int(y_lim/2)]
            ref_line_y = channel.IntensInterface(image, ref_range=ref_range)
        else:
            ref_line_y = int((interface[0][1] + interface[1][1])/2)  
            
        channel_count, _depth, _depth_points, channel_center, plateau \
                        = channel.StemDepth(image, ref_line_y, threshold=self._threshold, 
                                            scan=self._scan_to_avg, target='bright')        
        length = [[] for _ in range(channel_count)]
        cd_points = [[] for _ in range(channel_count)]
               
        for i in range(channel_count):
            length[i].append(_depth[i])
            cd_points[i].append(_depth_points[i])
        # DYL: Here depth array must be FLOAT ARRAY in order to use the numpy.nan
        # There is no numpy.nan for INT!!!    
        return channel_count, ref_line_y, length, cd_points
    