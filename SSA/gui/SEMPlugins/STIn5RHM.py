# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import NormalDist
from ...analysis import channel
import numpy as np
import json
from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal

class STIn5RHM(NormalDist):
    
    data_transfer_sig = pyqtSignal(list, list, dict)
    calib_dict = {'500K' : 0.248047,
                  '100K' : 1.116503,
                  '50K' : 2.233027,
                  '20K' : 5.582623,
                  '12K' : 9.304371,
                  '10K' : 11.16579}
    name = 'STI n5 demo Remaining HM'
    _lvl_name=['RemainHM']
    information = ('RHM',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__(mode='Vertical', add_bot_lim=True)
        
        self._auto_CD = self._STIn5AutoRHM
        self.data = {}
        
        try:
            with open('STIn5RHMSetting.json', 'r') as f:
                setting_dict = json.load(f)
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
                self._mag = setting_dict['Mag']
        except:
            self._scan_to_avg = 3
            self._threshold = 80      
            self._mag = '500K'
              
        self._calib = self.calib_dict[self._mag]       
        self._show_profile = False
        self._field = 'bright'
        
        self._extra_control_widget.append(QLabel('Field:'))
        self._choose_field = QComboBox()
        self._choose_field.addItem('dark')
        self._choose_field.addItem('bright')
        self._choose_field.setCurrentText(self._field)
        self._choose_field.activated[str].connect(self._set_field)
        self._extra_control_widget.append(self._choose_field)
        
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
        
        self._extra_control_widget.append(QLabel('Scan to Avg:'))
        self._input_scan_avg = QLineEdit()
        self._input_scan_avg.setText(str(self._scan_to_avg))
        self._input_scan_avg.editingFinished.connect(self._change_scan_avg)
        self._extra_control_widget.append(self._input_scan_avg)
        
    def clean_up(self):
        self._saveSettings('STIn5RHMSetting.json')
        super().clean_up()
    
    def _saveSettings(self, file_name):                
        setting_dict = {'Scan' : self._scan_to_avg,
                        'Threshold' : self._threshold,
                        'Mag' : self._mag}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
    
    def _update_plugin(self):
        self._on_new_image(self._full_image, same_img=True)
        super()._update_plugin()
    
    def _set_mag(self, magnification):
        self._mag = magnification
        self._calib = self.calib_dict[self._mag]
        self._update_plugin()
    
    def _set_field(self, field):
        self._field = field
        self._update_plugin()
        
    def _change_thres(self):
        try:
            thres = float(self._input_thres.text())
            if thres >= 100:
                thres = 99
                self._input_thres.setText(str(thres))
            if thres <= -100:
                thres = -99
                self._input_thres.setText(str(thres))
            self._threshold = thres           
            self._update_plugin()
        except:
            return
        
    def _change_scan_avg(self):
        try:
            self._scan_to_avg = int(self._input_scan_avg.text())
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
      
    def _STIn5AutoRHM(self, image, interface=None):

        y_lim, x_lim = image.shape
         
        if interface is None:            
            ref_range = [5, int(y_lim*0.5)]
            channel_count, reference, _depth, _depth_points, channel_center, plateau \
                            = channel.FinRHM(image, find_ref=True, 
                                               threshold=self._threshold,
                                               ref_range=ref_range, mode='up',
                                               scan=self._scan_to_avg,
                                               field=self._field)            
        else:
            channel_count, reference, _depth, _depth_points, channel_center, plateau \
                            = channel.FinRHM(image, find_ref=interface, 
                                               threshold=self._threshold, 
                                               ref_range=None, mode='up',
                                               scan=self._scan_to_avg,
                                               field=self._field)                         
        length = [[] for _ in range(channel_count)]
        cd_points = [[] for _ in range(channel_count)]
               
        for i in range(channel_count):
            length[i].append(_depth[i])
            cd_points[i].append(_depth_points[i])        
        
        return channel_count, reference, length, cd_points