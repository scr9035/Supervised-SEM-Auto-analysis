# -*- coding: utf-8 -*-

from ..plugins import NormalDist
from ...analysis import ChannelCD, BulkCD
import numpy as np
from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal
import json
import matplotlib.pyplot as plt


class CHYCutBulk(NormalDist):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    
    calib_dict = {'100K' : 1.116503,
                  '50K' : 2.233027,
                  '22K' : 5.075224,
                  '20K' : 5.582623,
                  '12K' : 9.304371,
                  '10K' : 11.16579}
    img_mode_list = ['up',
                     'down']
    
    name = 'Channel Hole Top CD Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    special_data_transfer_sig = pyqtSignal(dict)
    _lvl_name=['BulkWidth']
    
    information = ('Measure CD/thickness of the bulk region' ,
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__()
        self._auto_CD = self.AutoCHYCut
        # DYL: dictionary to store data corresponding to each levels
        self.data = {}
        self._show_profile = False
        self._img_mode = 'up' # Is this the upper version or lower part of the bulk
        self._sampling = 100
        self._start_from = 200
        self._total_length = 5.650
        
        try:
            with open('CHYCutBulkSetting.json', 'r') as f:
                setting_dict = json.load(f)
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
                self._mag = setting_dict['Mag']
                self._sampling = setting_dict['Sampling']
                self._start_from = setting_dict['Start']
                self._total_length = setting_dict['Total_Length']
        except:
            self._scan_to_avg = 5
            self._threshold = 90
            self._mag = '22K'
                        
        self._extra_control_widget.append(QLabel('Image mode:'))
        self._choose_mode = QComboBox()
        
        for key in self.img_mode_list:
            self._choose_mode.addItem(key)
        self._choose_mode.setCurrentText(self._img_mode)
        self._choose_mode.activated[str].connect(self._set_mode)
        self._extra_control_widget.append(self._choose_mode) 
        
        self._extra_control_widget.append(QLabel(u'Total channel length (µm):'))
        self._input_total_length = QLineEdit()
        self._input_total_length.setText(str(self._total_length))
        self._input_total_length.editingFinished.connect(self._change_total_length)
        self._extra_control_widget.append(self._input_total_length)
        
        self._extra_control_widget.append(QLabel('Start from (nm):'))
        self._input_start_from = QLineEdit()
        self._input_start_from.setText(str(self._start_from))
        self._input_start_from.editingFinished.connect(self._change_start_from)
        self._extra_control_widget.append(self._input_start_from)
        
        self._extra_control_widget.append(QLabel('Sampling (nm):'))
        self._input_sampling = QLineEdit()
        self._input_sampling.setText(str(self._sampling))
        self._input_sampling.editingFinished.connect(self._change_sampling)
        self._extra_control_widget.append(self._input_sampling)
        
        self._calib = self.calib_dict[self._mag]
        self._extra_control_widget.append(QLabel('Magnification:'))
        self._choose_mag = QComboBox()
        for key in self.calib_dict.keys():
            self._choose_mag.addItem(key)
        self._choose_mag.setCurrentText(self._mag)
        self._choose_mag.activated[str].connect(self._set_mag)
        self._extra_control_widget.append(self._choose_mag)
        
        self._extra_control_widget.append(QLabel('Scan to Avg (pixel):'))
        self._input_scan_avg = QLineEdit()
        self._input_scan_avg.setText(str(self._scan_to_avg))
        self._input_scan_avg.editingFinished.connect(self._change_scan_avg)
        self._extra_control_widget.append(self._input_scan_avg)
        
        self._extra_control_widget.append(QLabel('Threshold (%):'))
        self._input_thres = QLineEdit()
        self._input_thres.setText(str(self._threshold))
        self._input_thres.editingFinished.connect(self._change_thres)
        self._extra_control_widget.append(self._input_thres)
        
      
    def AutoCHYCut(self, image, interface=None):
        y_lim, x_lim = image.shape
        sample = int(self._sampling/self._calib)
        start_from = int(self._start_from/self._calib)
        if self._img_mode == 'up':
            ref_range = [int(y_lim/50), int(y_lim/2)]
        elif self._img_mode == 'down':
            ref_range = [int(y_lim/2), y_lim]        
        
        if interface is None:
            count, ref_line, channel_cd, lvl, channel_points, channel_center, plateau = \
            BulkCD(image, sample, start_from, algo='fit', find_ref=True, 
                   ref_range=ref_range, scan=self._scan_to_avg, mode=self._img_mode)
        else:
            count, ref_line, channel_cd, lvl, channel_points, channel_center, plateau = \
            BulkCD(image, sample, start_from, algo='fit', find_ref=interface, 
                   ref_range=ref_range, scan=self._scan_to_avg, mode=self._img_mode)
        
        return count, ref_line, channel_cd, channel_points
    
    def clean_up(self):
        self._saveSettings('CHYCutBulkSetting.json')
        super().clean_up()

    def _saveSettings(self, file_name):                
        setting_dict = {'Scan' : self._scan_to_avg,
                        'Threshold' : self._threshold,
                        'Mag' : self._mag,
                        'Sampling' : self._sampling,
                        'Start' : self._start_from,
                        'Total_Length' : self._total_length}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)    
    
    def _set_mag(self, magnification):
        self._mag = magnification
        self._calib = self.calib_dict[self._mag]
        self._update_plugin()
    
    def _set_mode(self, mode):
        self._img_mode = mode
        self._update_plugin()
    
    def data_transfer(self):
        """Function override to transfer raw data to measurement data """
        num_bulk, num_lvls = np.shape(self._cd_data)
        end_points = [[] for _ in range(num_bulk)]
        
        depth = []
        bulk_cd = []
        for i in range(num_bulk):
            for j in range(num_lvls):
                cd = self._cd_data[i][j]
                line = self.cd_lines[i][j]
                if cd is not None and line is not None:
                    bulk_cd.append(cd)
                    end_points[i].append(line.end_points)
                    depth.append(line.level)
        depth = np.array(depth)
        end_points = np.array(end_points)
        left = np.array([])
        right = np.array([])
        for i in range(num_bulk):
            center = np.mean(end_points[i][:,:,0])
            left = np.concatenate((left, end_points[i][:, 0, 0] - center))
            right = np.concatenate((right, end_points[i][:, 1, 0] - center))
        ref_lvl = self._ref_line.level
        
        if self._img_mode == 'up':
            depth = (depth - ref_lvl) * self._calib
        elif self._img_mode == 'down':
            depth = self._total_length * 1000 - (ref_lvl - depth) * self._calib      
        bulk_cd = np.array(bulk_cd) * self._calib
        left = left * self._calib
        right = right * self._calib
        
#        plt.plot(left, depth, 'b.')
#        plt.plot(right, depth, 'r.')
#        plt.show()
        
        self.special_data_transfer_sig.emit({'Depth' : depth, 
                                             'Bulk_CD' : bulk_cd, 
                                             'Left' : left,
                                             'Right' : right})
#        raw_data = np.transpose(self._cd_data) * self._calib      
#        for i in range(self._lvl_count):
#            self.data[self._lvl_name[i]] = raw_data[i]
#        hori_header = ['Ch %i' %n for n in range(1,self._channel_count+1)]
#        self.data_transfer_sig.emit(self._lvl_name, hori_header, self.data)
    
    def data_merge(self, historical_data):
        pass
    
    def _update_plugin(self):
        self._on_new_image(self._full_image, same_img=True)
        super()._update_plugin()
    
    def _change_total_length(self):
        try:
            self._total_length = int(self._input_total_length.text())
            self._update_plugin()
        except:
            return
    
    def _change_start_from(self):
        try:
            self._start_from = int(self._input_start_from.text())
            self._update_plugin()
        except:
            return
    
    def _change_sampling(self):
        try:
            self._start_from = int(self._input_start_from.text())
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