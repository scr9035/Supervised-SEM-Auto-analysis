# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from PyQt5.QtWidgets import (QPushButton, QAction, QTableWidget, QTableWidgetItem,
                             QLabel, QLineEdit, QComboBox)
from PyQt5.QtCore import pyqtSignal
import numpy as np
import json

from ..plugins import NormalDist
from ...analysis.channel import IntensInterface, RecessDepth

class CHRecess(NormalDist):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'Channel Hole Recess Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
#    calib_dict = {'100K' : 1.116503,
#                  '50K' : 2.233027,
#                  '20K' : 5.582623,
#                  '12K' : 9.304371,
#                  '10K' : 11.16579}
    _lvl_name=['Recess']
    
    information = ('Measurement: Channel hole recess from reference to etch front',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__(mode='Vertical')
        self._auto_CD = self.AutoCHRecess
        # DYL: Set False so won't show profile in default
        self._show_profile = False
        self.data = {}
        
        try:
            with open('CHRecessSetting.json', 'r') as f:
                setting_dict = json.load(f)
                self._scan_to_avg = setting_dict['Scan']
#                self._mag = setting_dict['Mag']
                self._threshold = setting_dict['Threshold']
        except:
            self._scan_to_avg = 3 
#            self._mag = '50K'
            self._threshold = 100
        
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
        self._saveSettings('CHRecessSetting.json')
        super().clean_up()
    
    def _saveSettings(self, file_name):                
        setting_dict = {'Scan' : self._scan_to_avg,
                        'Threshold' : self._threshold}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
    
    def _on_new_image(self, image, calib, same_img=False):
        super()._on_new_image(image, calib)
        if self._calib is not np.nan:
            self._input_manual_calib.setEnabled(False)
            
#    def _set_mag(self, magnification):
#        self._mag = magnification
#        self._calib = self.calib_dict[self._mag]
#        self._update_plugin()

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
    
    def AutoCHRecess(self, image, interface=None):
        y_lim, x_lim = image.shape
        if interface is None:
            ref_range = [int(y_lim/2), y_lim]
            ref_line_y = IntensInterface(image, ref_range=ref_range)
        else:
            ref_line_y = int((interface[0][1] + interface[1][1])/2)  

        channel_count, _depth, _depth_points, channel_center, plateau \
                        = RecessDepth(image, ref_line_y, threshold=self._threshold,
                                       scan=self._scan_to_avg, mode='down')            

        length = [[] for _ in range(channel_count)]
        cd_points = [[] for _ in range(channel_count)]
              
        for i in range(channel_count):
            length[i].append(_depth[i])
            cd_points[i].append(_depth_points[i])
        # DYL: Here depth array must be FLOAT ARRAY in order to use the numpy.nan
        # There is no numpy.nan for INT!!!
        return channel_count, ref_line_y, length, cd_points