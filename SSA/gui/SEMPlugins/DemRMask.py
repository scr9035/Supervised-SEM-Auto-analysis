# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import HVDistance
from ...analysis.channel import IntensInterface, RemainMask
import numpy as np
from PyQt5.QtWidgets import (QLabel, QLineEdit, QCheckBox)
from PyQt5.QtCore import pyqtSignal, Qt
import json


class DemRMask(HVDistance):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'DEM Remain Mask Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    
    information = ('Remain mask measurements, with option to measure bulk height',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        
        super().__init__()
        self._auto_CD = self.AutoDemRMask
        self.data = {}
        self._setting_file = self._setting_folder + 'DemRMaskSetting.json'
        try:
            with open(self._setting_file, 'r') as f:
                setting_dict = json.load(f)
                self._meas_bulk = setting_dict['Measure_bulk']
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
                self._preset_ref = setting_dict['PresetRef']
        except:
            self._meas_bulk = True
            self._scan_to_avg = 5
            self._threshold = 95
            self._preset_ref = False

        self._manual_calib = 1
        self._extra_control_widget.append(QLabel('Manual Calibration (nm/pixel):'))
        self._input_manual_calib = QLineEdit()
        if self._calib is np.nan:
            self._input_manual_calib.setText(str(self._manual_calib))
        self._input_manual_calib.editingFinished.connect(self._change_manual_calib)
        self._extra_control_widget.append(self._input_manual_calib)
        
        self._bulk_cb = QCheckBox('Include Bulk')
        if self._meas_bulk:
            self._bulk_cb.toggle()
        self._bulk_cb.stateChanged.connect(self._choose_measure_bulk)
        self._extra_control_widget.append(self._bulk_cb)
                
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
        setting_dict = {'Measure_bulk' : self._meas_bulk,
                        'Scan' : self._scan_to_avg,                        
                        'Threshold' : self._threshold,
                        'PresetRef' : self._preset_ref}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
    
    def _choose_measure_bulk(self, state):
        if state == Qt.Checked:
            self._meas_bulk = True
            self._update_plugin()
        else:
            self._meas_bulk = False
            self._update_plugin()
        
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
        
        if self._meas_bulk:
            lvl_name = ['RMask', 'Bulk']
        else:
            lvl_name = ['RMask']
        for i in range(self._lvl_count):
            self.data[lvl_name[i]] = raw_data[i]
        hori_header = ['Ch %i' %n for n in range(1,self._channel_count+1)]

        self.data_transfer_sig.emit(lvl_name, hori_header, self.data)
    
    def AutoDemRMask(self, image, interface=None):

        y_lim, x_lim = image.shape             
        if interface is None:
            ref_range = [int(y_lim/2), y_lim]
            ref_line_y = IntensInterface(image, ref_range=ref_range)
        else:
            ref_line_y = int((interface[0][1] + interface[1][1])/2)
        
        channel_count, height, top_points, channel_center, plateau \
                        = RemainMask(image, ref_line_y, scan=self._scan_to_avg)
                        
        length = [[] for _ in range(channel_count)]
        cd_points = [[] for _ in range(channel_count)]
        for i in range(channel_count):
            length[i].append(height[i])
            cd_points[i].append(top_points[i]) 
        line_modes = ['Vertical']
        
        if self._meas_bulk:
            bulk_top = IntensInterface(image, ref_range=[0, int(y_lim*0.8)])    
            for i in range(channel_count):
                length[i].append(None)
                cd_points[i].append(None)
            length[0][1] = ref_line_y - bulk_top
            cd_points[0][1] = [[int(x_lim/2), bulk_top], [int(x_lim/2), ref_line_y]]
            line_modes.append('Vertical')                                         
        return channel_count, ref_line_y, length, cd_points, line_modes
        

      



