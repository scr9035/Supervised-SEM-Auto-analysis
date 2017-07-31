# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import NormalDist
from ...analysis import channel
import numpy as np
import json
from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal

class STIn5Edge(NormalDist):
    
    data_transfer_sig = pyqtSignal(list, list, dict)
    calib_dict = {'500K' : 0.248047,
                  '100K' : 1.116503,
                  '50K' : 2.233027,
                  '20K' : 5.582623,
                  '12K' : 9.304371,
                  '10K' : 11.16579}
    name = 'STI n5 demo CDs'
    _lvl_name=['level1', 'level2', 'Max', 'Min', 'Spacer']
    information = ('CDs',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__(mode='Horizontal', add_bot_lim=True)
        
        self._auto_CD = self._STIn5AutoEdge
        self.data = {}
        
        try:
            with open('STIn5EdgeSetting.json', 'r') as f:
                setting_dict = json.load(f)
                self._level1 = setting_dict['level1']
                self._level2 = setting_dict['level2']
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
                self._mag = setting_dict['Mag']
        except:
            self._level1 = 5
            self._level2 = 54
            self._scan_to_avg = 5
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
        
        self._extra_control_widget.append(QLabel('Level 1 Level (nm):'))
        self._input_lvl1 = QLineEdit()
        self._input_lvl1.setText(str(self._level1))
        self._input_lvl1.editingFinished.connect(self._change_lvl1)
        self._extra_control_widget.append(self._input_lvl1)
        
        self._extra_control_widget.append(QLabel('Level 2 Level (nm):'))
        self._input_lvl2 = QLineEdit()
        self._input_lvl2.setText(str(self._level2))
        self._input_lvl2.editingFinished.connect(self._change_lvl2)
        self._extra_control_widget.append(self._input_lvl2)
        
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
        self._saveSettings('STIn5EdgeSetting.json')
        super().clean_up()
    
    def _saveSettings(self, file_name):                
        setting_dict = {'level1' : self._level1, 
                        'level2' : self._level2,
                        'Scan' : self._scan_to_avg,
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
    
    def _change_lvl1(self):
        try:
            self._level1 = int(self._input_lvl1.text())
            self._update_plugin()
        except:
            return
        
    def _change_lvl2(self):
        try:
            self._level2 = int(self._input_lvl1.text())
            self._update_plugin()
        except:
            return
        
    def _change_thres(self):
        try:
            thres = float(self._input_thres.text())
            if thres >= 100:
                thres = 99
                self._input_thres.setText(str(thres))
            if thres <= 0:
                thres = 1
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
      
    def _STIn5AutoEdge(self, image, interface=None):
        level1_lvl = int(round(self._level1/self._calib))
        level2_lvl = int(round(self._level2/self._calib))
        search_lvl = np.arange(level1_lvl+self._scan_to_avg, level2_lvl-self._scan_to_avg, 
                               self._scan_to_avg)
        search_lvl = np.concatenate([[level1_lvl], search_lvl, [level2_lvl]], 0)
        
        y_lim, x_lim = image.shape
        
        if self._field == 'bright':
            space_field = 'dark'
        else:
            space_field = 'bright'
        
        if interface is None:            
            ref_range = [5, int(y_lim*0.5)]
            channel_count, reference, lvl_cd, lvl_points, _center, _plateau \
                            = channel.FinEdge(image, search_lvl, find_ref=True, 
                                               threshold=self._threshold, 
                                               ref_range=ref_range, mode='up', 
                                               scan=self._scan_to_avg,
                                               field=self._field)                            
            space_count, reference, space_cd, space_points, space_center, space_plat \
                            = channel.FinEdge(image, np.array([0]), find_ref=True,
                                              threshold=self._threshold,
                                              ref_range=ref_range, mode='up',
                                              scan=self._scan_to_avg,
                                              field=space_field)            
        else:
            
            channel_count, reference, lvl_cd, lvl_points, _center, _plateau \
                            = channel.FinEdge(image, search_lvl, find_ref=interface, 
                                              threshold=self._threshold, 
                                              ref_range=None, mode='up',
                                              scan=self._scan_to_avg,
                                              field=self._field)
            space_count, reference, space_cd, space_points, space_center, space_plat \
                            = channel.FinEdge(image, np.array([0]), find_ref=interface,
                                              threshold=self._threshold,
                                              ref_range=None, mode='up',
                                              scan=self._scan_to_avg,
                                              field=space_field)            
        max_cd = [[] for _ in range(channel_count)]
        min_cd = [[] for _ in range(channel_count)]
        max_points = [[] for _ in range(channel_count)]
        min_points = [[] for _ in range(channel_count)]
        for i in range(channel_count):
            maxCD_idx = np.argmax(lvl_cd[i][1:-1])
            minCD_idx = np.argmin(lvl_cd[i][1:-1]) 
            
            max_cd[i].append(lvl_cd[i][maxCD_idx+1])
            min_cd[i].append(lvl_cd[i][minCD_idx+1])
            
            max_points[i].append(lvl_points[i][maxCD_idx+1])
            min_points[i].append(lvl_points[i][minCD_idx+1])
        
        if channel_count > space_count:
            diff = channel_count - space_count
            for i in range(diff):
                space_cd.append([np.nan])
                space_points.append([[[np.nan, np.nan], [np.nan, np.nan]]])
        
        lvl1_points = np.expand_dims(np.array(lvl_points)[:,0], axis=1)
        lvl2_points = np.expand_dims(np.array(lvl_points)[:,-1], axis=1)
        lvl1_cd = np.expand_dims(np.array(lvl_cd)[:,0], axis=1)
        lvl2_cd = np.expand_dims(np.array(lvl_cd)[:,-1], axis=1)   

        channel_CD = np.concatenate((lvl1_cd, lvl2_cd, max_cd, min_cd, space_cd), axis=1)
        cd_points = np.concatenate((lvl1_points, lvl2_points, max_points, min_points, space_points), axis=1).tolist()
        cd_points[-1][-1] = None       
        return channel_count, reference, channel_CD, cd_points