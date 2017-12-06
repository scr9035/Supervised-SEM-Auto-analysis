# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal
import numpy as np
import json

from ..plugins import HVDistance
from ...analysis.channel import IntensInterface, ChannelDepth

class DemEtchDepth(HVDistance):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'Channel Hole Etch Depth Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    _lvl_name=['Depth']
    
    information = ('Measurement: Channel hole depth from interface to etch front',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__(add_right_lim=True, add_bot_lim=True)        
        self._auto_CD = self.AutoDemEtchDepth
        self.data = {}
        self._setting_file = self._setting_folder + 'DemDepthSetting.json'
        try:
            with open(self._setting_file, 'r') as f:
                setting_dict = json.load(f)
                self._number_of_channel = setting_dict['Channel']
                self._scan_to_avg = setting_dict['Scan']
                self._preset_ref = setting_dict['PresetRef']
        except:
            self._number_of_channel = 20
            self._scan_to_avg = 3
            self._preset_ref = False
        
        self._manual_calib = 1
        self._extra_control_widget.append(QLabel('Manual Calibration (nm/pixel):'))
        self._input_manual_calib = QLineEdit()
        if self._calib is np.nan:
            self._input_manual_calib.setText(str(self._manual_calib))
        self._input_manual_calib.editingFinished.connect(self._change_manual_calib)
        self._extra_control_widget.append(self._input_manual_calib)
                
        self._threshold = 100
        self._extra_control_widget.append(QLabel('Number of Channel:'))
        self._input_channel_number = QLineEdit()
        self._input_channel_number.setText(str(self._number_of_channel))
        self._input_channel_number.editingFinished.connect(self._change_channel_number)
        self._extra_control_widget.append(self._input_channel_number)
        
        self._extra_control_widget.append(QLabel('Scan to Avg:'))
        self._input_scan_avg = QLineEdit()
        self._input_scan_avg.setText(str(self._scan_to_avg))
        self._input_scan_avg.editingFinished.connect(self._change_scan_avg)
        self._extra_control_widget.append(self._input_scan_avg)
    
    def clean_up(self):
        self._saveSettings(self._setting_file)
        super().clean_up()
    
    def _saveSettings(self, file_name):                
        setting_dict = {'Scan' : self._scan_to_avg,
                        'Channel' : self._number_of_channel,
                        'PresetRef' : self._preset_ref}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
    
    def _receive_calib(self, calib):
        super()._receive_calib(calib)
        if self._calib is not np.nan:
            self._input_manual_calib.setEnabled(False) 
    
    def _change_manual_calib(self):
        try:
            self._manual_calib = float(self._input_manual_calib.text())
            self._update_plugin()
        except:
            return
            
    def _change_channel_number(self):
        try:
            self._number_of_channel = int(self._input_channel_number.text())
            self._update_plugin()
        except:
            return
    
    def _change_scan_avg(self):
        try:
            self._scan_to_avg = int(self._input_scan_avg.text())
            self._update_plugin()
        except:
            return
    
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
    
    def AutoDemEtchDepth(self, image, interface=None):
        spare_channel = 1
        y_lim, x_lim = image.shape
        if interface is None:
            ref_range = [int(y_lim/10.), int(4*y_lim/5.)]
            ref_line_y = IntensInterface(image, ref_range=ref_range)
        else:
            ref_line_y = int((interface[0][1] + interface[1][1])/2) 
            
        channel_count, _depth, _depth_points, channel_center, plateau \
                        = ChannelDepth(image, ref_line_y, threshold=self._threshold, 
                                       scan=self._scan_to_avg, mag='low')            
        num = self._number_of_channel
        if self._number_of_channel > channel_count - spare_channel:
            num = channel_count - spare_channel
        
        length = [[] for _ in range(num)]
        cd_points = [[] for _ in range(num)]
        
        for i in range(num):
            length[num-1-i].append(_depth[channel_count-i-spare_channel])
            cd_points[num-1-i].append(_depth_points[channel_count-i-spare_channel])
        # DYL: Here depth array must be FLOAT ARRAY in order to use the numpy.nan
        # There is no numpy.nan for INT!!!
        line_modes = ['Vertical']
        return num, ref_line_y, length, cd_points, line_modes