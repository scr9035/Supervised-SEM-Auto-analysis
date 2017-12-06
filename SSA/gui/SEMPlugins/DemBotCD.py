# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import HVDistance
from ...analysis.channel import ChannelCD, IntensInterface, RecessDepth
import numpy as np
from PyQt5.QtWidgets import (QLabel, QLineEdit, QMessageBox, QCheckBox)
from PyQt5.QtCore import pyqtSignal, Qt
import json

class DemBotCD(HVDistance):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    name = 'DEM Bot CD Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
#    _lvl_name=['TEOS','SSL2']
    
    information = ('Bottom CD measurements, with level defined by users and option to measure recess',
                   '')
    information = '\n'.join(information)
    
    def __init__(self):
        
        super().__init__(add_bot_lim=True)
        self._auto_CD = self.AutoDemBotCD
        self.data = {}
        self._setting_file = self._setting_folder + 'DemBotCDSetting.json'
        try:
            with open(self._setting_file, 'r') as f:
                setting_dict = json.load(f)
                self._lvl_name = setting_dict['Level_name']
                self._lvl_pos = setting_dict['Level_pos']
                self._meas_recess = setting_dict['Measure_recess']
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
                self._preset_ref = setting_dict['PresetRef']
        except:
            self._lvl_name = ['WL17', 'DMY0', 'BotCD']
            self._lvl_pos = [1280, 230, 0]
            self._meas_recess = True
            self._scan_to_avg = 5
            self._threshold = 95
            self._preset_ref = True

        self._manual_calib = 1
        self._extra_control_widget.append(QLabel('Manual Calibration (nm/pixel):'))
        self._input_manual_calib = QLineEdit()
        if self._calib is np.nan:
            self._input_manual_calib.setText(str(self._manual_calib))
        self._input_manual_calib.editingFinished.connect(self._change_manual_calib)
        self._extra_control_widget.append(self._input_manual_calib)
        
        self._recess_cb = QCheckBox('Include Recess')
        if self._meas_recess:
            self._recess_cb.toggle()
        self._recess_cb.stateChanged.connect(self._choose_measure_recess)
        self._extra_control_widget.append(self._recess_cb)
        
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
    
    def _choose_measure_recess(self, state):
        if state == Qt.Checked:
            self._meas_recess = True
            self._update_plugin()
        else:
            self._meas_recess = False
            self._update_plugin()
    
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
        
        if self._meas_recess:
            lvl_name = self._lvl_name + ['Recess']
        else:
            lvl_name = self._lvl_name
        
        for i in range(self._lvl_count):
            self.data[lvl_name[i]] = raw_data[i]
        hori_header = ['Ch %i' %n for n in range(1,self._channel_count+1)]

        self.data_transfer_sig.emit(lvl_name, hori_header, self.data)
    
    def AutoDemBotCD(self, image, interface=None):
        if self._calib is not np.nan:
            calib = self._calib * 10**9
        else:
            calib = self._manual_calib

        bot_lvls = np.round(np.array(self._lvl_pos)/calib).astype(int)        
        y_lim, x_lim = image.shape             
        if interface is None:
            ref_range = [int(y_lim/2), y_lim]
            ref_line_y = IntensInterface(image, ref_range=ref_range)
        else:
            ref_line_y = int((interface[0][1] + interface[1][1])/2)
              
        channel_count, channel_CD, cd_points, _center, _plateau \
                        = ChannelCD(image, bot_lvls, ref_line_y, 
                                    scan=self._scan_to_avg, threshold=self._threshold, 
                                    noise=1000, iteration=0, mode='down')

        line_modes = ['Horizontal' for _ in range(len(self._lvl_name))]
        
        if self._meas_recess:      
            channel_count2, recess_depth, _depth_points, _center, _plateau \
                        = RecessDepth(image, ref_line_y, threshold=self._threshold,
                                       scan=self._scan_to_avg, mode='down') 
            for i in range(channel_count):
                channel_CD[i].append(recess_depth[i])
                cd_points[i].append(_depth_points[i])
            line_modes.append('Vertical')
            
        return channel_count, ref_line_y, channel_CD, cd_points, line_modes
    
