# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal
import numpy as np
import json

from ..plugins import NormalDist
from ...analysis.channel import IntensInterface, RemainMask

class CHRMask(NormalDist):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'Channel Hole Remaining Mask Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    _lvl_name=['RMask', 'Bulk']
    information = ('Measurement: Channel hole remaining mask from interface to mask',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__(mode='Vertical')
        self._auto_CD = self.AutoCHRMask
        # DYL: Set False so won't show profile in default
        self._show_profile = False
        self.data = {}
        
        try:
            with open('CHRMaskSetting.json', 'r') as f:
                setting_dict = json.load(f)
                self._scan_to_avg = setting_dict['Scan']
        except:
            self._scan_to_avg = 1   
        
        self._manual_calib = 1 
        self._extra_control_widget.append(QLabel('Manual Calibration (nm/pixel):'))
        self._input_manual_calib = QLineEdit()
        if self._calib is np.nan:
            self._input_manual_calib.setText(str(self._manual_calib))
        self._input_manual_calib.editingFinished.connect(self._change_manual_calib)
        self._extra_control_widget.append(self._input_manual_calib)
        
        self._extra_control_widget.append(QLabel('Scan to Avg:'))
        self._input_scan_avg = QLineEdit()
        self._input_scan_avg.setText(str(self._scan_to_avg))
        self._input_scan_avg.editingFinished.connect(self._change_scan_avg)
        self._extra_control_widget.append(self._input_scan_avg)
    
    def clean_up(self):
        self._saveSettings('CHRMaskSetting.json')
        super().clean_up()
    
    def _saveSettings(self, file_name):                
        setting_dict = {'Scan' : self._scan_to_avg}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)


    def _change_scan_avg(self):
        try:
            self._scan_to_avg = int(self._input_scan_avg.text())
            self._update_plugin()
        except:
            return
    
    def _receive_calib(self, calib):
        super()._receive_calib(calib)
        if self._calib is not np.nan:
            self._input_manual_calib.setEnabled(False)
    
    def _change_manual_calib(self):
        try:
            self._manual_calib = float(self._input_manual_calib)
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
        
        raw_data = np.transpose(np.array(self._cd_data, dtype=np.float)) * calib
        for i in range(self._lvl_count):
            self.data[self._lvl_name[i]] = raw_data[i]
        hori_header = ['Ch %i' %n for n in range(1,self._channel_count+1)]
        self.data_transfer_sig.emit(self._lvl_name, hori_header, self.data)
        
    def AutoCHRMask(self, image, interface=None):
        y_lim, x_lim = image.shape
        if interface is None:
            ref_range = [int(y_lim/2), y_lim]
            ref_line_y = IntensInterface(image, ref_range=ref_range)
        else:
            ref_line_y = int((interface[0][1] + interface[1][1])/2)
        
        bulk_top = IntensInterface(image, ref_range=[0, int(y_lim*0.8)])    
        
        channel_count, height, top_points, channel_center, plateau \
                        = RemainMask(image, ref_line_y, scan=self._scan_to_avg)
        length = [[] for _ in range(channel_count)]
        cd_points = [[] for _ in range(channel_count)]
        
        for i in range(channel_count):
            length[i].append(height[i])
            length[i].append(None)
            cd_points[i].append(top_points[i])
            cd_points[i].append(None)
        
        length[0][1] = ref_line_y - bulk_top
        cd_points[0][1] = [[int(x_lim/2), bulk_top], [int(x_lim/2), ref_line_y]]
        return channel_count, ref_line_y, length, cd_points