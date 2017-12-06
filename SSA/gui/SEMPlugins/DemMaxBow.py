# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import HVDistance
from ...analysis.channel import ChannelCD, IntensInterface
import numpy as np
from PyQt5.QtWidgets import (QLabel, QLineEdit, QMessageBox)
from PyQt5.QtCore import pyqtSignal
import json


class DemMaxBow(HVDistance):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'DEM Max Bow Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    
    information = ('Max bow measurements, with user defined name of the bow.',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        
        super().__init__(add_bot_lim=True, add_top_lim=True)
        self._auto_CD = self.AutoDemMaxBowCD
        self.data = {}
        self._setting_file = self._setting_folder + 'DemMaxBowSetting.json'
        self._preset_ref = None
        try:
            with open(self._setting_file, 'r') as f:
                setting_dict = json.load(f)
                self._lvl_name = setting_dict['Level_name']
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
        except:
            self._lvl_name = ['MaxBow']
            self._scan_to_avg = 10
            self._threshold = 95

        self._manual_calib = 1
        self._extra_control_widget.append(QLabel('Manual Calibration (nm/pixel):'))
        self._input_manual_calib = QLineEdit()
        if self._calib is np.nan:
            self._input_manual_calib.setText(str(self._manual_calib))
        self._input_manual_calib.editingFinished.connect(self._change_manual_calib)
        self._extra_control_widget.append(self._input_manual_calib)
        
        self._extra_control_widget.append(QLabel('Bow Name:'))
        self._input_lvl_name = QLineEdit()
        self._input_lvl_name.setText(self._lvl_name[0])
        self._input_lvl_name.editingFinished.connect(self._change_lvl_name)
        self._extra_control_widget.append(self._input_lvl_name)
                
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
        self._saveSettings(self._setting_file)
        super().clean_up()
    
    def _saveSettings(self, file_name):                
        setting_dict = {'Level_name' : self._lvl_name,
                        'Scan' : self._scan_to_avg,                        
                        'Threshold' : self._threshold}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
    
    def _change_lvl_name(self):
        try:
            self._lvl_name = [self._input_lvl_name.text()]
            self._update_plugin()
        except:
            return
        
    def _change_manual_calib(self):
        try:
            self._manual_calib = float(self._input_manual_calib.text())
            self._update_plugin()
        except:
            return
            
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
    
    def _receive_calib(self, calib):
        super()._receive_calib(calib)
        if self._calib is not np.nan:
            self._input_manual_calib.setEnabled(False) 
    
    def _update_plugin(self):
        if self._full_image is not None:
            self._on_new_image(self._full_image, same_img=True)        
            super()._update_plugin()
        
    def data_transfer(self):
        """Function override to transfer raw data to measurement data """
        if self._calib is not np.nan:
            calib = self._calib * 10**9
        else:
            calib = self._manual_calib
        
        raw_data = np.transpose(self._cd_data) * calib
        
        for i in range(self._lvl_count):
            self.data[self._lvl_name[i]] = raw_data[i]
        hori_header = ['Ch %i' %n for n in range(1,self._channel_count+1)]
        self.data_transfer_sig.emit(self._lvl_name, hori_header, self.data)
    
    def AutoDemMaxBowCD(self, image, interface=None):
        
        y_lim, x_lim = image.shape
        bow_lvl = np.arange(10, y_lim, self._scan_to_avg)

        channel_count, bow_full_cd, bow_full_points, _center, _plateau \
                        = ChannelCD(image, bow_lvl, 0,
                                    scan=self._scan_to_avg, threshold=self._threshold, 
                                    noise=1000, iteration=0)
        
        max_bow_cd = [[] for _ in range(channel_count)]
        max_bow_points = [[] for _ in range(channel_count)]
        for i in range(channel_count):
            bow_idx = np.argmax(bow_full_cd[i])
            max_bow_cd[i].append(bow_full_cd[i][bow_idx])
            max_bow_points[i].append(bow_full_points[i][bow_idx])
        return channel_count, None, max_bow_cd, max_bow_points, ['Horizontal']
    
        
        