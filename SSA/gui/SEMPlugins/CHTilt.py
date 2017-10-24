# -*- coding: utf-8 -*-

from ..plugins import NormalDist
from ...analysis import ChannelCD, BulkCD
from ...analysis.channel import IntensInterface, FourierTilt
import numpy as np
from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox, QFrame, QPushButton)
from PyQt5.QtCore import pyqtSignal
from ..utils import dialogs
import json
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as spysig



class CHTilt(NormalDist):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    
#    calib_dict = {'100K' : 1.116503,
#                  '50K' : 2.233027,
#                  '27K' : 4.135239,
#                  '25K' : 4.466,
#                  '22K' : 5.075224,
#                  '20K' : 5.582623,
#                  '12K' : 9.304371,
#                  '10K' : 11.16579}
    
    name = 'Channel Hole Tilt Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    special_data_transfer_sig = pyqtSignal(dict)
    
    information = ('Channel Hole Tilt Measurements Using Fourier Transform' ,
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__()
        self._auto_CD = self.AutoTilt
        # DYL: dictionary to store data corresponding to each levels
        self.data = {}
        
        self._show_profile = False        
        self._special_data = None
        
        try:
            with open('CHTilt.json', 'r') as f:
                setting_dict = json.load(f)
                self._scan_to_avg = setting_dict['Scan']
#                self._mag = setting_dict['Mag']
                self._start_from = setting_dict['Start']
                self._end_at = setting_dict['End']
                self._period_count = setting_dict['Count']
        except:
            self._scan_to_avg = 5
#            self._mag = '12K'
            self._start_from = 400
            self._end_at = 3500
            self._period_count = 60
                                
#        self._calib = self.calib_dict[self._mag]
#        self._extra_control_widget.append(QLabel('Magnification:'))
#        self._choose_mag = QComboBox()
#        for key in self.calib_dict.keys():
#            self._choose_mag.addItem(key)
#        self._choose_mag.setCurrentText(self._mag)
#        self._choose_mag.activated[str].connect(self._set_mag)
#        self._extra_control_widget.append(self._choose_mag)
        
        self._manual_calib = 1
        self._extra_control_widget.append(QLabel('Manual Calibration (nm/pixel):'))
        self._input_manual_calib = QLineEdit()
        if self._calib is np.nan:
            self._input_manual_calib.setText(str(self._manual_calib))
        self._input_manual_calib.editingFinished.connect(self._change_manual_calib)
        self._extra_control_widget.append(self._input_manual_calib)        

        self._extra_control_widget.append(QLabel('Start from (nm):'))
        self._input_start_from = QLineEdit()
        self._input_start_from.setText(str(self._start_from))
        self._input_start_from.editingFinished.connect(self._change_start_from)
        self._extra_control_widget.append(self._input_start_from)
        
        self._extra_control_widget.append(QLabel('End at (nm):'))
        self._input_end_at = QLineEdit()
        self._input_end_at.setText(str(self._end_at))
        self._input_end_at.editingFinished.connect(self._change_end_at)
        self._extra_control_widget.append(self._input_end_at)
        
        self._extra_control_widget.append(QLabel('Approx number of channels:'))
        self._input_period_count = QLineEdit()
        self._input_period_count.setText(str(self._period_count))
        self._input_period_count.editingFinished.connect(self._change_period_count)
        self._extra_control_widget.append(self._input_period_count)
        
        self._extra_control_widget.append(QLabel('Scan to Avg (pixel):'))
        self._input_scan_avg = QLineEdit()
        self._input_scan_avg.setText(str(self._scan_to_avg))
        self._input_scan_avg.editingFinished.connect(self._change_scan_avg)
        self._extra_control_widget.append(self._input_scan_avg)
      
    def AutoTilt(self, image, interface=None):
        if self._calib is not np.nan:
            calib = self._calib * 10**9
        else:
            calib = self._manual_calib
        
        y_lim, x_lim = image.shape
        if interface is None:
            ref_range = [0, int(y_lim/10)]
            ref_line_y = IntensInterface(image, ref_range=ref_range)
        else:
            ref_line_y = int((interface[0][1] + interface[1][1])/2)     
        start = int(self._start_from/calib)
        end = int(self._end_at/calib)
        depth, shift = FourierTilt(image, ref_line_y, self._period_count, 
                                   start=start, end=end, scan=self._scan_to_avg)
        self._special_data = {'Depth': depth * calib,
                              'Shift': shift * calib}
        return 0, ref_line_y, [[]], []
    
    def clean_up(self):
        self._saveSettings('CHTilt.json')
        super().clean_up()

    def _saveSettings(self, file_name):                
        setting_dict = {'Scan' : self._scan_to_avg,
                        'Start' : self._start_from,
#                        'Mag' : self._mag,
                        'End' : self._end_at,
                        'Count' : self._period_count}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)    
    
#    def _set_mag(self, magnification):
#        self._mag = magnification
#        self._calib = self.calib_dict[self._mag]
#        self._update_plugin()

    
    def data_transfer(self):
        """Function override to transfer raw data to measurement data """
        if self._special_data is not None:            
            self.special_data_transfer_sig.emit(self._special_data)
            
            
    def _data_analysis(self):
        pass
    
    def data_merge(self, historical_data):
        pass
    
    def _update_plugin(self):
        self._on_new_image(self._full_image, same_img=True)
        super()._update_plugin()
    
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
    
    def _change_scan_avg(self):
        try:
            self._scan_to_avg = int(self._input_scan_avg.text())
            self._update_plugin()
        except:
            return
        
    def _change_start_from(self):
        try:
            self._start_from = int(self._input_start_from.text())
            self._update_plugin()
        except:
            return
        
    def _change_end_at(self):
        try:
            self._end_at = int(self._input_end_at.text())
            self._update_plugin()
        except:
            return
        
    def _change_period_count(self):
        try:
            self._period_count = int(self._input_period_count.text())
            self._update_plugin()
        except:
            return