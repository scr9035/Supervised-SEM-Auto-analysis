# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import HVDistance
from ...analysis import ChannelCD
import numpy as np
from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal
import json


class CHTopCD(HVDistance):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'Channel Hole Top CD Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    _lvl_name=['TEOS','SSL2']
    
    information = ('Measurement level: TEOS (maximum CD within 100 nm), SSL2',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        
        super().__init__()
        self._auto_CD = self.AutoCHTopCD
        # DYL: dictionary to store data corresponding to each levels
        self.data = {}
        self._setting_file = self._setting_folder + 'CHTopCDSetting.json'
        try:
            with open(self._setting_file, 'r') as f:
                setting_dict = json.load(f)
                self._TEOS = setting_dict['TEOS']
                self._SSL2 = setting_dict['SSL2']
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
                self._preset_ref = setting_dict['PresetRef']
        except:
            self._TEOS = 100
            self._SSL2 = 420
            self._scan_to_avg = 5
            self._threshold = 100
            self._preset_ref = False

        self._manual_calib = 1
        
        self._extra_control_widget.append(QLabel('Manual Calibration (nm/pixel):'))
        self._input_manual_calib = QLineEdit()
        if self._calib is np.nan:
            self._input_manual_calib.setText(str(self._manual_calib))
        self._input_manual_calib.editingFinished.connect(self._change_manual_calib)
        self._extra_control_widget.append(self._input_manual_calib)

        self._extra_control_widget.append(QLabel('SSL2 Level (nm):'))
        self._input_ssl2 = QLineEdit()
        self._input_ssl2.setText(str(self._SSL2))
        self._input_ssl2.editingFinished.connect(self._change_ssl2)
        self._extra_control_widget.append(self._input_ssl2)
        
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
        setting_dict = {'TEOS' : self._TEOS, 
                        'SSL2' : self._SSL2,
                        'Scan' : self._scan_to_avg,
                        'Threshold' : self._threshold,
                        'PresetRef' : self._preset_ref}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
    
    def _change_manual_calib(self):
        try:
            self._manual_calib = float(self._input_manual_calib.text())
            self._update_plugin()
        except:
            return
            
    def _change_ssl2(self):
        try:
            self._SSL2 = int(self._input_ssl2.text())
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
    
    def AutoCHTopCD(self, image, interface=None):
        if self._calib is not np.nan:
            calib = self._calib * 10**9
        else:
            calib = self._manual_calib
        
        TEOS_lvl = np.arange(5, int(round(self._TEOS/calib)), self._scan_to_avg)
        SSL2_lvl = np.array([int(round(self._SSL2/calib))])
        y_lim, x_lim = image.shape
        
        if interface is None:            
            ref_range = [0, int(y_lim*0.8)]
            channel_count, ref_line, TEOS_full_cd, TEOS_full_points, _center, _plateau \
                            = ChannelCD(image, TEOS_lvl, find_ref=True, ref_range=ref_range,
                                        scan=self._scan_to_avg, threshold=self._threshold, 
                                        noise=1000, iteration=0, mode='up')
            
            channel_count, ref_line, SSL2_cd, SSL2_points, _center, _plateau \
                            = ChannelCD(image, SSL2_lvl, find_ref=True, ref_range=ref_range, 
                                        scan=self._scan_to_avg, threshold=self._threshold, 
                                        noise=1000, iteration=0, mode='up')
        else:
            channel_count, ref_line, TEOS_full_cd, TEOS_full_points, _center, _plateau \
                            = ChannelCD(image, TEOS_lvl, find_ref=interface, 
                                        scan=self._scan_to_avg, threshold=self._threshold, 
                                        noise=1000, iteration=0, mode='up')
            channel_count, ref_line, SSL2_cd, SSL2_points, _center, _plateau \
                            = ChannelCD(image, SSL2_lvl, find_ref=interface, 
                                        scan=self._scan_to_avg, threshold=self._threshold, 
                                        noise=1000, iteration=0, mode='up')        
        TEOS_cd = [[] for _ in range(channel_count)]
        TEOS_points = [[] for _ in range(channel_count)]
        for i in range(channel_count):
            bow_idx = np.argmax(TEOS_full_cd[i])
            TEOS_cd[i].append(TEOS_full_cd[i][bow_idx])
            TEOS_points[i].append(TEOS_full_points[i][bow_idx])
        cd_points = np.concatenate((TEOS_points, SSL2_points), axis=1)
        channel_CD = np.concatenate((TEOS_cd, SSL2_cd), axis=1)
        line_modes = ['Horizontal', 'Horizontal']
        return channel_count, ref_line, channel_CD, cd_points, line_modes