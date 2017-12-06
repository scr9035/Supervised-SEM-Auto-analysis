# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import HVDistance
from ...analysis.channel import ChannelCD, IntensInterface
import numpy as np
from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox, QMessageBox)
from PyQt5.QtCore import pyqtSignal
import json


class DemTopCD(HVDistance):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'DEM Top CD Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)    
    information = ('Top CD measurements, with level defined by users',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        
        super().__init__()
        self._auto_CD = self.AutoDemTopCD
        self.data = {}
        self._setting_file = self._setting_folder + 'DemTopCDSetting.json'
        try:
            with open(self._setting_file, 'r') as f:
                setting_dict = json.load(f)
                self._lvl_name = setting_dict['Level_name']
                self._lvl_pos = setting_dict['Level_pos']
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
                self._preset_ref = setting_dict['PresetRef']
        except:
            self._lvl_name = ['SSL2']
            self._lvl_pos = [400]
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
        
        self._extra_control_widget.append(QLabel('Level Names (separated by #):'))
        self._input_lvl_name = QLineEdit()
        self._input_lvl_name.setText('#'.join(self._lvl_name))
        self._input_lvl_name.editingFinished.connect(self._change_lvl_name)
        self._extra_control_widget.append(self._input_lvl_name)
        
        self._extra_control_widget.append(QLabel('Level Position (nm) (separated by #):'))
        self._input_lvl_pos = QLineEdit()
        self._input_lvl_pos.setText('#'.join(map(str, self._lvl_pos)))
        self._input_lvl_pos.editingFinished.connect(self._change_lvl_pos)
        self._extra_control_widget.append(self._input_lvl_pos)
        
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
        
    def _update_data(self):
        if len(self._lvl_pos) != len(self._lvl_name):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle('Setting Error')
            msg.setText('Level Name and Position Don\'t Match')
            msg.setInformativeText('Number of level name and number of level \
                                   position are not equal. Please check the setting \
                                   and its format.<br><br>Format Example:<br><br>\
                                      LvL1#LvL2#LvL3<br>\
                                      100#200#300')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
            return
        super()._update_data()
    
    def clean_up(self):
        self._saveSettings(self._setting_file)
        super().clean_up()
    
    def _saveSettings(self, file_name):                
        setting_dict = {'Level_name' : self._lvl_name,
                        'Level_pos' : self._lvl_pos,
                        'Scan' : self._scan_to_avg,                        
                        'Threshold' : self._threshold,
                        'PresetRef' : self._preset_ref}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)
    
    def _change_lvl_name(self):
        try:
            self._lvl_name = self._input_lvl_name.text().split('#')
            self._update_plugin()
        except:
            return
    
    def _change_lvl_pos(self):
        try:
            self._lvl_pos = list(map(float, self._input_lvl_pos.text().split('#')))
            self._update_plugin()
        except Exception as e:
            print(str(e))
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle('Setting Error')
            msg.setText('Illegal Position Setting')
            msg.setInformativeText('Please only input number separated by # <br><br>\
                                   Example:<br><br>\
                                      100#200#300')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
            return
    
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
        
        for i in range(self._lvl_count):
            self.data[self._lvl_name[i]] = raw_data[i]
        hori_header = ['Ch %i' %n for n in range(1,self._channel_count+1)]
        self.data_transfer_sig.emit(self._lvl_name, hori_header, self.data)
    
    def AutoDemTopCD(self, image, interface=None):
        if self._calib is not np.nan:
            calib = self._calib * 10**9
        else:
            calib = self._manual_calib        
        top_lvls = np.round(np.array(self._lvl_pos)/calib).astype(int)        
        
        y_lim, x_lim = image.shape             
        if interface is None:
            ref_range = [0, int(y_lim*0.8)]
            ref_line_y = IntensInterface(image, ref_range=ref_range)
        else:
            ref_line_y = int((interface[0][1] + interface[1][1])/2)
              
        channel_count, channel_CD, cd_points, _center, _plateau \
                        = ChannelCD(image, top_lvls, ref_line_y, 
                                    scan=self._scan_to_avg, threshold=self._threshold, 
                                    noise=1000, iteration=0, mode='up')
    
        line_modes = ['Horizontal' for _ in range(len(self._lvl_name))]
        return channel_count, ref_line_y, channel_CD, cd_points, line_modes
    
    
    
    
    
    
    
    
    
        