# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal
import numpy as np
import json

from ..plugins import NormalDist
from ...analysis.channel import IntensInterface, ChannelDepth

class CHEtchDepth(NormalDist):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'Channel Hole Etch Depth Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    calib_dict = {'100K' : 1.116503,
                  '50K' : 2.233027,
                  '20K' : 5.582623,
                  '12K' : 9.304371,
                  '10K' : 11.16579}
    _lvl_name=['Depth']
    
    information = ('Measurement: Channel hole depth from interface to etch front',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__(mode='Vertical', add_right_lim=True, add_bot_lim=True)
        
        self._auto_CD = self.AutoCHEtchDepth
        # DYL: Set False so won't show profile in default
        self._show_profile = False
        self.data = {}
        
        try:
            with open('CHDepthSetting.json', 'r') as f:
                setting_dict = json.load(f)
                self._number_of_channel = setting_dict['Channel']
                self._scan_to_avg = setting_dict['Scan']
                self._mag = setting_dict['Mag']
        except:
            self._number_of_channel = 20
            self._scan_to_avg = 3     
            self._mag = '12K'
        
        self._threshold = 100
              
        self._calib = self.calib_dict[self._mag]        
        self._extra_control_widget.append(QLabel('Magnification:'))
        self._choose_mag = QComboBox()
        for key in self.calib_dict.keys():
            self._choose_mag.addItem(key)
        self._choose_mag.setCurrentText(self._mag)
        self._choose_mag.activated[str].connect(self._set_mag)
        self._extra_control_widget.append(self._choose_mag)
        
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
        self._saveSettings('CHDepthSetting.json')
        super().clean_up()
    
    def _saveSettings(self, file_name):                
        setting_dict = {'Scan' : self._scan_to_avg,
                        'Mag' : self._mag,
                        'Channel' : self._number_of_channel}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
    
    def _set_mag(self, magnification):
        self._mag = magnification
        self._calib = self.calib_dict[self._mag]
        self._update_plugin()
    
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
        raw_data = np.transpose(self._cd_data) * self._calib
        for i in range(self._lvl_count):
            self.data[self._lvl_name[i]] = raw_data[i]
        hori_header = ['Ch %i' %n for n in range(1,self._channel_count+1)]
        self.data_transfer_sig.emit(self._lvl_name, hori_header, self.data)
    
    def AutoCHEtchDepth(self, image, interface=None):
        y_lim, x_lim = image.shape
        if interface is None:
            ref_range = [int(y_lim/10), y_lim]
            ref_line_y = IntensInterface(image, ref_range=ref_range)
        else:
            ref_line_y = int((interface[0][1] + interface[1][1])/2)  
            
        channel_count, _depth, _depth_points, channel_center, plateau \
                        = ChannelDepth(image, ref_line_y, threshold=self._threshold, 
                                       scan=self._scan_to_avg, mag='low')            
        num = self._number_of_channel
        if self._number_of_channel > channel_count-5:
            num = channel_count-5
        
        length = [[] for _ in range(num)]
        cd_points = [[] for _ in range(num)]
        
        for i in range(num):
            length[num-1-i].append(_depth[channel_count-i-5])
            cd_points[num-1-i].append(_depth_points[channel_count-i-5])
        # DYL: Here depth array must be FLOAT ARRAY in order to use the numpy.nan
        # There is no numpy.nan for INT!!!
        return num, ref_line_y, length, cd_points