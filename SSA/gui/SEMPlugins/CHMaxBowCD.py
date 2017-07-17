# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal
import numpy as np
import json

from ..plugins import NormalDist
from ...analysis import ChannelCD

class CHMaxBowCD(NormalDist):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'Channel Hole Max Bow CD Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    calib_dict = {'100K' : 1.116503,
                  '50K' : 2.233027,
                  '20K' : 5.582623,
                  '12K' : 9.304371,
                  '10K' : 11.16579}
    _lvl_name=['Max Bow']
    
    information = ('Measurement level: Max Bow (maximum CD through channel)',
                   'Vertical height: 1; Iteration: 0;')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__(mode='Horizontal', add_bot_lim=True, add_top_lim=True)
        self._auto_CD = self.AutoCHMaxBowCD
        self.data = {}
        
        try:
            with open('CHMaxBowSetting.json', 'r') as f:
                setting_dict = json.load(f)
                self._scan_to_avg = setting_dict['Scan']
                self._mag = setting_dict['Mag']
                self._threshold = setting_dict['Threshold']
        except:
            self._scan_to_avg = 5    
            self._mag = '50K'   
            self._threshold = 100
        
        self._calib = self.calib_dict[self._mag]
        
        self._extra_control_widget.append(QLabel('Magnification:'))
        self._choose_mag = QComboBox()
        for key in self.calib_dict.keys():
            self._choose_mag.addItem(key)
        self._choose_mag.setCurrentText(self._mag)
        self._choose_mag.activated[str].connect(self._set_mag)
        self._extra_control_widget.append(self._choose_mag)
        
        self._extra_control_widget.append(QLabel('Scan to Avg:'))
        self._input_scan_avg = QLineEdit()
        self._input_scan_avg.setText(str(self._scan_to_avg))
        self._input_scan_avg.editingFinished.connect(self._change_scan_avg)
        self._extra_control_widget.append(self._input_scan_avg)
        
        self._extra_control_widget.append(QLabel('Threshold:'))
        self._input_thres = QLineEdit()
        self._input_thres.setText(str(self._threshold))
        self._input_thres.editingFinished.connect(self._change_thres)
        self._extra_control_widget.append(self._input_thres)
    
    def clean_up(self):
        self._saveSettings('CHMaxBowSetting.json')
        super().clean_up()
    
    def _saveSettings(self, file_name):                
        setting_dict = {'Scan' : self._scan_to_avg,
                        'Mag' : self._mag,
                        'Threshold' : self._threshold}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
            
    def _set_mag(self, magnification):
        self._mag = magnification
        self._calib = self.calib_dict[self._mag]
        self._update_plugin()
    
    def _change_scan_avg(self):
        try:
            self._scan_to_avg = int(self._input_scan_avg.text())
            self._update_plugin()
        except:
            return
    
    def _change_thres(self):
        try:
            thres = float(self._input_thres.text())
            if thres > 100:
                thres = 100
                self._input_thres.setText(str(thres))
            if thres < 0:
                thres = 0
                self._input_thres.setText(str(thres))
            self._threshold = thres           
            self._update_plugin()
        except:
            return
    
    def _update_plugin(self):
        self._on_new_image(self._full_image, same_img=True)
        super()._update_plugin()
        
    def data_transfer(self):
        """Function override to transfer raw data to measurement data """
        raw_data = np.transpose(self._cd_data) * self._calib   
        for i in range(self._lvl_count):
            self.data[self._lvl_name[i]] = raw_data[i]
        hori_header = ['Ch %i' %n for n in range(1,self._channel_count+1)]
        self.data_transfer_sig.emit(self._lvl_name, hori_header, self.data)
    
    def AutoCHMaxBowCD(self, image, interface=None):
        y_lim, x_lim = image.shape
        bow_lvl = np.arange(10, y_lim, self._scan_to_avg)
        
        
        # DYL: Don't need reference here
        channel_count, ref_line, bow_full_cd, bow_full_points, _center, _plateau \
                        = ChannelCD(image, bow_lvl, algo='fit', find_ref=None, 
                                    scan=self._scan_to_avg, threshold=self._threshold, 
                                    noise=1000, iteration=0)
                        
        max_bow_cd = [[] for _ in range(channel_count)]
        max_bow_points = [[] for _ in range(channel_count)]
        for i in range(channel_count):
            bow_idx = np.argmax(bow_full_cd[i])
            max_bow_cd[i].append(bow_full_cd[i][bow_idx])
            max_bow_points[i].append(bow_full_points[i][bow_idx])
        # DYL: return reference line as None
        return channel_count, None, max_bow_cd, max_bow_points