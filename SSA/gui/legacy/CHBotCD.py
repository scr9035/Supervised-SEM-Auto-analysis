# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

# TODO: implement the SEM plugin to measure channel hole bot CD

import numpy as np
import json
from ..plugins import HVDistance
from ...analysis import ChannelCD
from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal


class CHBotCD(HVDistance):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'Channel Hole Bot CD Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    _lvl_name=['WL17', 'DMY0', 'BCD']
    
    information = ('Measurement level: WL17, DMY0, BCD',
                   '')
    information = '\n'.join(information)
    
    
    def __init__(self):
        super().__init__()
        self._auto_CD = self.AutoCHBotCD
        # DYL: dictionary to store data corresponding to each levels
        self.data = {}
        self._setting_file = self._setting_folder + 'CHBotCDSetting.json'
        try:
            with open(self._setting_file, 'r') as f:
                setting_dict = json.load(f)
                self._WL17 = setting_dict['WL17']
                self._DMY0 = setting_dict['DMY0']
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
                self._preset_ref = setting_dict['PresetRef']
        except:
            self._WL17 = 1280
            self._DMY0 = 230
            self._scan_to_avg = 3
            self._threshold = 100
            self._preset_ref = True

        self._manual_calib = 1
        self._extra_control_widget.append(QLabel('Manual Calibration (nm/pixel):'))
        self._input_manual_calib = QLineEdit()
        if self._calib is np.nan:
            self._input_manual_calib.setText(str(self._manual_calib))
        self._input_manual_calib.editingFinished.connect(self._change_manual_calib)
        self._extra_control_widget.append(self._input_manual_calib)
        
        self._extra_control_widget.append(QLabel('WL17 Level:'))
        self._input_WL17 = QLineEdit()
        self._input_WL17.setText(str(self._WL17))
        self._input_WL17.editingFinished.connect(self._change_WL17)
        self._extra_control_widget.append(self._input_WL17)
        
        self._extra_control_widget.append(QLabel('DMY0 Level:'))
        self._input_DMY0 = QLineEdit()
        self._input_DMY0.setText(str(self._DMY0))
        self._input_DMY0.editingFinished.connect(self._change_DMY0)
        self._extra_control_widget.append(self._input_DMY0)
        
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
        setting_dict = {'WL17' : self._WL17, 
                        'DMY0' : self._DMY0,
                        'Scan' : self._scan_to_avg,
                        'Threshold' : self._threshold,
                        'PresetRef' : self._preset_ref}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
    
    def AutoCHBotCD(self, image, interface=None):
        if self._calib is not np.nan:
            calib = self._calib * 10**9
        else:
            calib = self._manual_calib
            
        WL17_lvl = round(self._WL17/calib)
        DMY0_lvl = round(self._DMY0/calib)
        BCD_lvl = 0
        bot_lvls = np.array([WL17_lvl, DMY0_lvl, BCD_lvl])
        
        y_lim, x_lim = image.shape
        
        if interface is None:    
            ref_range = [int(y_lim/2), y_lim]
            channel_count, ref_line, bot_cd, bot_cd_points, _center, _plateau \
                            = ChannelCD(image, bot_lvls, find_ref=True, ref_range=ref_range,
                                scan=self._scan_to_avg, threshold=self._threshold, 
                                noise=1000, iteration=0, mode='down')
        else:
            channel_count, ref_line, bot_cd, bot_cd_points, _center, _plateau \
                            = ChannelCD(image, bot_lvls, find_ref=interface, 
                                scan=self._scan_to_avg, threshold=self._threshold, 
                                noise=1000, iteration=0, mode='down')
        
        line_modes = ['Horizontal', 'Horizontal', 'Horizontal']
        return channel_count, ref_line, bot_cd, bot_cd_points, line_modes
    
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
                
    def _receive_calib(self, calib):
        super()._receive_calib(calib)
        if self._calib is not np.nan:
            self._input_manual_calib.setEnabled(False)
    
    def _update_plugin(self):
        self._on_new_image(self._full_image, same_img=True)
        super()._update_plugin()
        
    def _change_manual_calib(self):
        try:
            self._manual_calib = float(self._input_manual_calib.text())
            self._update_plugin()
        except:
            return
    
    def _change_WL17(self):
        try:
            self._WL17 = int(self._input_WL17.text())
            self._update_plugin()
        except:
            return
    
    def _change_DMY0(self):
        try:
            self._DMY0 = int(self._input_DMY0.text())
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