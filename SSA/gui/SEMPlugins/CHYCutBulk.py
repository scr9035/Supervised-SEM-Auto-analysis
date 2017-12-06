# -*- coding: utf-8 -*-

from ..plugins import HVDistance
from ...analysis import ChannelCD, BulkCD
import numpy as np
from PyQt5.QtWidgets import (QLabel, QLineEdit, QComboBox, QFrame, QPushButton,
                             QCheckBox, QScrollArea, QVBoxLayout, QWidget,
                             QSizePolicy, QHBoxLayout, QGridLayout)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt
from ..utils import dialogs, new_plot
import json
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as spysig


class CHYCutBulk(HVDistance):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image 
        viewer won't be able to receive the data
    """
    img_mode_list = ['up',
                     'down',
                     'mask']
    
    name = 'Channel Hole YCut Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)
    special_data_transfer_sig = pyqtSignal(dict)
    _lvl_name=['BulkWidth']
    
    information = ('Measure center-to-center distance between the two outer holes, ' ,
                   'to present outer hole bending.')
    information = '\n'.join(information)
    
    def __init__(self):
        super().__init__()
        self._auto_CD = self.AutoCHYCut
        # DYL: dictionary to store data corresponding to each levels
        self.data = {}
        self._show_profile = False
        self._img_mode = 'up' # Is this the upper version or lower part of the bulk
        self._setting_file = self._setting_folder + 'CHYCutBulkSetting.json'
        try:
            with open(self._setting_file, 'r') as f:
                setting_dict = json.load(f)
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
                self._sampling = setting_dict['Sampling']
                self._start_from = setting_dict['Start']
                self._total_length = setting_dict['Total_Length']
                self._fit_order = setting_dict['Fit_Order']
                self._zero_position = setting_dict['Zero_Position']
                self._preset_ref = setting_dict['PresetRef']
        except:
            self._sampling = 150
            self._start_from = 200
            self._total_length = 5.650
            self._fit_order = 6
            self._zero_position = 400
            self._scan_to_avg = 5
            self._threshold = 90
            self._preset_ref = False
        
        self._manual_calib = 1
        self._extra_control_widget.append(QLabel('Manual Calibration (nm/pixel):'))
        self._input_manual_calib = QLineEdit()
        if self._calib is np.nan:
            self._input_manual_calib.setText(str(self._manual_calib))
        self._input_manual_calib.editingFinished.connect(self._change_manual_calib)
        self._extra_control_widget.append(self._input_manual_calib)
        
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
        
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        self._extra_control_widget.append(hline)
        
        self._extra_control_widget.append(QLabel('Data Analysis:'))
        
        self._extra_control_widget.append(QLabel('Order of Polynomial:'))
        self._input_order = QLineEdit()
        self._input_order.setText(str(self._fit_order))
        self._input_order.editingFinished.connect(self._change_order)
        self._extra_control_widget.append(self._input_order)
        
        self._extra_control_widget.append(QLabel('Zero Position (nm):'))
        self._input_zero_position = QLineEdit()
        self._input_zero_position.setText(str(self._zero_position))
        self._input_order.editingFinished.connect(self._change_zero)
        self._extra_control_widget.append(self._input_zero_position)
        
        upload_btn = QPushButton('Choose data', self)
        upload_btn.setToolTip('Choose the SpecialData.xlsx file generated by the measurements')
        upload_btn.clicked.connect(self._data_analysis)
        upload_btn.resize(upload_btn.sizeHint())
        self._extra_control_widget.append(upload_btn)
        
    
    def attach(self, image_viewer):
        super().attach(image_viewer)
        self._profile_cb.setEnabled(False)
      
    def AutoCHYCut(self, image, interface=None):
        if self._calib is not np.nan:
            calib = self._calib * 10**9
        else:
            calib = self._manual_calib
        
        y_lim, x_lim = image.shape
        sample = int(self._sampling/calib)
        start_from = int(self._start_from/calib)
        if self._img_mode == 'up':
            ref_range = [int(y_lim/50), int(y_lim/2)]
        elif self._img_mode == 'down' or self._img_mode == 'mask':
            ref_range = [int(y_lim/2), y_lim]        
        
        if interface is None:
            count, ref_line, channel_cd, lvl, channel_points, channel_center, plateau = \
            BulkCD(image, sample, start_from, algo='fit', find_ref=True, 
                   ref_range=ref_range, scan=self._scan_to_avg, mode=self._img_mode)
        else:
            count, ref_line, channel_cd, lvl, channel_points, channel_center, plateau = \
            BulkCD(image, sample, start_from, algo='fit', find_ref=interface, 
                   ref_range=ref_range, scan=self._scan_to_avg, mode=self._img_mode)
        
        line_modes = ['Horizontal' for _ in range(np.shape(channel_cd)[1])]
        return count, ref_line, channel_cd, channel_points, line_modes
    
    def clean_up(self):
        self._saveSettings(self._setting_file)
        super().clean_up()
    
    def _receive_calib(self, calib):
        super()._receive_calib(calib)
        if self._calib is not np.nan:
            self._input_manual_calib.setEnabled(False) 

    def _saveSettings(self, file_name):                
        setting_dict = {'Scan' : self._scan_to_avg,
                        'Threshold' : self._threshold,
                        'Sampling' : self._sampling,
                        'Start' : self._start_from,
                        'Total_Length' : self._total_length,
                        'Zero_Position' : self._zero_position,
                        'Fit_Order' : self._fit_order,
                        'PresetRef' : self._preset_ref}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)    
    
    def _set_mode(self, mode):
        self._img_mode = mode
        self._update_plugin()
    
    def _change_manual_calib(self):
        try:
            self._manual_calib = float(self._input_manual_calib)
            self._update_plugin()
        except:
            return
    
    def data_transfer(self):
        """Function override to transfer raw data to measurement data """
        if self._calib is not np.nan:
            calib = self._calib * 10**9
        else:
            calib = self._manual_calib
        
        if len(np.shape(self._cd_data)) == 2:
            num_channel, num_lvls = np.shape(self._cd_data)
            num_bulk = int(num_channel / 2)
            
            depth = []
            center_to_center = []

            for i in range(num_bulk):
                for j in range(num_lvls):
                    left_line = self.cd_lines[i*2][j]
                    right_line = self.cd_lines[i*2+1][j]
                    L_p1, L_p2 = left_line.end_points
                    R_p1, R_p2 = right_line.end_points
                    center_to_center.append((R_p1[0] + R_p2[0] - L_p1[0] - L_p2[0])/2)
                    depth.append(left_line.level)

            center_to_center = np.array(center_to_center) * calib
            depth = np.array(depth)
            ref_lvl = self._ref_line.level
            if self._img_mode == 'up':
                depth = (depth - ref_lvl) * calib
            elif self._img_mode == 'down':
                depth = self._total_length * 1000 - (ref_lvl - depth) * calib
            elif self._img_mode == 'mask':
                depth = - (ref_lvl - depth) * calib
            
            self.special_data_transfer_sig.emit({'Depth' : depth,
                                                 'CenterToCenter' : center_to_center})     
        
#        if len(np.shape(self._cd_data)) == 2:
#            num_bulk, num_lvls = np.shape(self._cd_data)
#            end_points = [[] for _ in range(num_bulk)]
#            
#            depth = []
#            bulk_cd = []
#            for i in range(num_bulk):
#                for j in range(num_lvls):
#                    cd = self._cd_data[i][j]
#                    line = self.cd_lines[i][j]
#                    if cd is not None and line is not None:
#                        bulk_cd.append(cd)
#                        end_points[i].append(line.end_points)
#                        depth.append(line.level)
#            depth = np.array(depth)
#            end_points = np.array(end_points)
#            left = np.array([])
#            right = np.array([])
#            for i in range(num_bulk):
#                center = np.mean(end_points[i][:,:,0])
#                left = np.concatenate((left, end_points[i][:, 0, 0] - center))
#                right = np.concatenate((right, end_points[i][:, 1, 0] - center))
#            ref_lvl = self._ref_line.level
            
#            if self._img_mode == 'up':
#                depth = (depth - ref_lvl) * self._calib
#            elif self._img_mode == 'down':
#                depth = self._total_length * 1000 - (ref_lvl - depth) * self._calib      
#            bulk_cd = np.array(bulk_cd) * self._calib
#            left = left * self._calib
#            right = right * self._calib
#            
#            self.special_data_transfer_sig.emit({'Depth' : depth, 
#                                                 'Bulk_CD' : bulk_cd, 
#                                                 'Left' : left,
#                                                 'Right' : right})
            
    def _data_analysis(self):
        
        data_path = dialogs.open_files_dialog()[0]
        full_data = pd.read_excel(data_path, header=[0,1], index_col=0)

        self.analysis_container = QScrollArea()
        self.analysis_container.setWidgetResizable(True) # CRITICAL
        self.analysis_container.setWindowTitle('YCut Analysis')
        self.analysis_section = QWidget()
        self.analysis_section.setLayout(QGridLayout())
        self.analysis_container.setWidget(self.analysis_section) # CRITICAL
        self.analysis_container.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        fig_width = 400
        fig_height = 800
        
        count = 0
        for chip_name in full_data.columns.levels[0]:
            data = full_data[chip_name].dropna(axis=0)
            x = data['Depth'].values
            full_cd = data['CenterToCenter'].values
            right = 0.5 * full_cd 
            N = 500
            coord = np.linspace(np.min(x), np.max(x), num=N)          
            cutoff_idx = np.argmin(np.abs(coord - self._zero_position))
            
            z_r = np.polyfit(x, right, self._fit_order)
            p_r = np.poly1d(z_r)
            outward = p_r(self._zero_position) - np.min(p_r(coord)[cutoff_idx:])
            inward = p_r(self._zero_position) - np.max(p_r(coord)[cutoff_idx:])

#            print('For chip ' + chip_name + ':')          
#            print('The outward bending is %.1f' %(p_r(cutoff) - np.min(p_r(coord)[cutoff_idx:])) )
#            print('The inward bending is %.1f'  %(p_r(cutoff) - np.max(p_r(coord)[cutoff_idx:])) )
#            f1 = plt.figure(figsize=(6,8))
#            plt.plot(right, x, 'b.', markersize=1)
#            plt.plot(p_r(coord), coord, 'r')
#            plt.plot([p_r(cutoff), p_r(cutoff)], [coord[0], coord[-1]], '--')
#            plt.xlabel('Position (nm)')
#            plt.ylabel('Depth (nm)')
#            plt.ylim([-1500, self._total_length * 1000])
#            plt.xlim([180, 210])
#            plt.gca().invert_yaxis()
#            plt.show()
            
            size_policy = QSizePolicy.Fixed
            figure, ax = new_plot(SizePolicy=size_policy)
            canvas = figure.canvas
            figure.set_figwidth(fig_width / float(figure.dpi))
            figure.set_figheight(fig_height / float(figure.dpi))
            qpalette = QPalette()
            qcolor = qpalette.color(QPalette.Window)
            bgcolor = qcolor.toRgb().value()
            if np.isscalar(bgcolor):
                bgcolor = str(bgcolor / 255.)
            figure.patch.set_facecolor(bgcolor)
            ax.plot(right, x, 'b.', markersize=1)
            ax.plot(p_r(coord), coord, 'r')
            ax.plot([p_r(self._zero_position), p_r(self._zero_position)], [coord[0], coord[-1]], '--')
            ax.set_xlabel('Position (nm)')
            ax.set_ylabel('Depth (nm)')
            ax.set_ylim([-1500, self._total_length * 1000])
            ax.set_xlim([180, 210])
            ax.invert_yaxis()
            figure.tight_layout()
            label = QLabel('The Chip ' + chip_name + ' outward/inward bending: %.1f/%.1f nm' %(outward, inward))
            label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            
            self.analysis_section.layout().addWidget(label, 0, count)
            self.analysis_section.layout().addWidget(canvas, 1, count)
            count += 1
        hsize = label.sizeHint().width() * 4
        vsize = self.analysis_section.sizeHint().height() + label.sizeHint().height()
        self.analysis_container.resize(hsize, vsize)
        self.analysis_container.show()

    
    def data_merge(self, historical_data):
        pass
    
    def _update_plugin(self):
        self._on_new_image(self._full_image, same_img=True)
        super()._update_plugin()
    
    def _change_total_length(self):
        try:
            self._total_length = float(self._input_total_length.text())
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
            self._sampling = int(self._input_sampling.text())
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
        
    def _change_order(self):
        try:
            self._fit_order = int(self._input_order.text())
            self._update_plugin()
        except:
            return
    
    def _change_zero(self):
        try:
            self._zero_position = int(self._input_zero_position.text())
            self._update_plugin()
        except:
            return