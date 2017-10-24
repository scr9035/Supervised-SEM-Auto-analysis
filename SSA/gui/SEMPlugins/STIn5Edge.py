# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import NormalDist
from ...analysis import channel
import numpy as np
import json
from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox, QCheckBox)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

class STIn5Edge(NormalDist):
    
    data_transfer_sig = pyqtSignal(list, list, dict)
    calib_dict = {'500K' : 0.248047,
                  '100K' : 1.116503,
                  '50K' : 2.233027,
                  '20K' : 5.582623,
                  '12K' : 9.304371,
                  '10K' : 11.16579}
    name = 'STI n5 demo CDs'
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
                self._levels = setting_dict['Levels']
                self._measure_max_min = setting_dict['TakeMaxMin']
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
                self._mag = setting_dict['Mag']
        except:
            self._levels = [5, 28, 54]
            self._measure_max_min = False
            self._scan_to_avg = 5
            self._threshold = -80    
            self._mag = '500K'
        
        max_min_check = QCheckBox('Measure Max/Min CD')
        if self._measure_max_min:
            max_min_check.toggle()
        max_min_check.stateChanged.connect(self._change_measure_max_min)
        self._extra_control_widget.append(max_min_check)
        
        self._lvl_name = ['level'+str(i+1) for i in range(len(self._levels))]
        if self._measure_max_min:
            self._lvl_name.append('Max')
            self._lvl_name.append('Min')
        self._lvl_name.append('Spacer')
        
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
        
        self._extra_control_widget.append(QLabel('Measurement Levels (nm) (eg. 1,2,3):'))
        self._input_lvl = QLineEdit()
        self._input_lvl.setText(','.join(map(str, self._levels)))
        self._input_lvl.editingFinished.connect(self._change_levels)
        self._extra_control_widget.append(self._input_lvl)
        
        
        
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
        setting_dict = {'Levels' : self._levels,
                        'TakeMaxMin' : self._measure_max_min,
                        'Scan' : self._scan_to_avg,
                        'Threshold' : self._threshold,
                        'Mag' : self._mag}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
    
    def _update_plugin(self):
        self._lvl_name = ['level'+str(i+1) for i in range(len(self._levels))]
        if self._measure_max_min:
            self._lvl_name.append('Max')
            self._lvl_name.append('Min')
        self._lvl_name.append('Spacer')
        self._on_new_image(self._full_image, same_img=True)
        super()._update_plugin()
    
    def _set_mag(self, magnification):
        self._mag = magnification
        self._calib = self.calib_dict[self._mag]
        self._update_plugin()
    
    def _set_field(self, field):
        self._field = field
        self._update_plugin()
        
    def _change_levels(self):
        try:
            self._levels = list(map(float, self._input_lvl.text().split(',')))
            self._levels.sort()
            self._update_plugin
        except:
            return
        
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
        
    def _change_measure_max_min(self, state):
        if state == Qt.Checked:
            self._measure_max_min = True
        else:
            self._measure_max_min = False
        self._update_plugin()
        
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

        levels = [int(round(i/self._calib)) for i in self._levels]
        if self._measure_max_min:
            search_lvl = np.arange(levels[0]+self._scan_to_avg, levels[-1]-self._scan_to_avg, 
                               self._scan_to_avg)
            search_lvl = np.concatenate([levels, search_lvl], axis=0)
            
        else:
            search_lvl = levels
        
        search_lvl = np.array(search_lvl)
        
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
        if channel_count > space_count:
            diff = channel_count - space_count
            for i in range(diff):
                space_cd.append([np.nan])
                space_points.append([[[np.nan, np.nan], [np.nan, np.nan]]])     

        end_points = np.array(lvl_points)[:, :len(levels)]
        end_cd = np.array(lvl_cd)[:, :len(levels)]
        if self._measure_max_min:
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
            channel_CD = np.concatenate((end_cd, max_cd, min_cd, space_cd), axis=1)
            cd_points = np.concatenate((end_points, max_points, min_points, space_points), axis=1)
        else:
            channel_CD = np.concatenate((end_cd, space_cd), axis=1)
            cd_points = np.concatenate((end_points, space_points), axis=1)
        cd_points = cd_points.tolist()
        cd_points[-1][-1] = None
        return channel_count, reference, channel_CD, cd_points