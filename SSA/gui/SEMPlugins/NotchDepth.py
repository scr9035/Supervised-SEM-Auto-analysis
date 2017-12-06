# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from ..plugins import HVDistance
from ...analysis.channel import HalfSplit
from ...analysis.GeneralProcess import (ReguIsoNonlinear, LGSemEdge, LPSemEdge,
                                        GradSlop, EdgeEnhance2D)
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QLabel, QLineEdit)
from PyQt5.QtCore import pyqtSignal
import json


class NotchDepth(HVDistance):
    """
    data_transfer_sig : pyqtSignal
        Any plugin needs to implement data trasfer sigal or otherwise the image
        viewer won't be able to receive the data
    """
    name = 'Notch Depth Measurements'
    data_transfer_sig = pyqtSignal(list, list, dict)

    information = ('Notch depth measurements',
                   '')
    information = '\n'.join(information)

    def __init__(self):

        super().__init__(add_bot_lim=False, add_top_lim=True)
        self._auto_CD = self.AutoNotchDepth
        self.data = {}
        self._setting_file = self._setting_folder + 'NotchDepthSetting.json'
        self._preset_ref = None
        try:
            with open(self._setting_file, 'r') as f:
                setting_dict = json.load(f)
                self._lvl_name = setting_dict['Level_name']
                self._scan_to_avg = setting_dict['Scan']
                self._threshold = setting_dict['Threshold']
        except Exception as e:
            self._lvl_name = ['NotchDepth']
            self._scan_to_avg = 10
            self._threshold = 85

        self._manual_calib = 1
        self._extra_control_widget.append(QLabel('Manual Calibration (nm/pixel):'))
        self._input_manual_calib = QLineEdit()
        if self._calib is np.nan:
            self._input_manual_calib.setText(str(self._manual_calib))
        self._input_manual_calib.editingFinished.connect(self._change_manual_calib)
        self._extra_control_widget.append(self._input_manual_calib)

        self._extra_control_widget.append(QLabel('Level Name:'))
        self._input_lvl_name = QLineEdit()
        self._input_lvl_name.setText(self._lvl_name[0])
        self._input_lvl_name.editingFinished.connect(self._change_lvl_name)
        self._extra_control_widget.append(self._input_lvl_name)

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
        self._saveSettings(self._setting_file)
        super().clean_up()

    def _saveSettings(self, file_name):
        setting_dict = {'Level_name': self._lvl_name,
                        'Scan': self._scan_to_avg,
                        'Threshold': self._threshold}
        with open(file_name, 'w') as f:
            json.dump(setting_dict, f)

    def _change_lvl_name(self):
        try:
            self._lvl_name = [self._input_lvl_name.text()]
            self._update_plugin()
        except Exception as e:
            return

    def _change_manual_calib(self):
        try:
            self._manual_calib = float(self._input_manual_calib.text())
            self._update_plugin()
        except Exception as e:
            return

    def _change_scan_avg(self):
        try:
            self._scan_to_avg = int(self._input_scan_avg.text())
            self._update_plugin()
        except Exception as e:
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
        except Exception as e:
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
        hori_header = ['Ch %i' % n for n in range(1, self._channel_count+1)]
        self.data_transfer_sig.emit(self._lvl_name, hori_header, self.data)

    def AutoNotchDepth(self, image, interface=None):
        y_lim, x_lim = image.shape
        split, left_lim, right_lim = HalfSplit(image, mode='Vertical', plot=True)
        left = image[:, left_lim:split]
        right = image[:, split:right_lim]
        
        smooth_left = EdgeEnhance2D(left, 300, 10, sigma=1.5, delta_t=0.1)
        smooth_right = EdgeEnhance2D(right, 300, 10, sigma=1.5, delta_t=0.1)
        
        f=plt.figure(figsize=(10,8))
        plt.imshow(left, cmap=plt.cm.gray)
        plt.show()
        
        f=plt.figure(figsize=(10,8))
        plt.imshow(smooth_left, cmap=plt.cm.gray)
        plt.show()
        
        
        profile_left = np.sum(smooth_left, axis=1)/(split-left_lim)
        profile_right = np.sum(smooth_right, axis=1)/(right_lim-split)

        up_left, down_left = GradSlop(profile_left, threshold=0.2,
                                      iteration=500, sigma=2, kappa=0.05,
                                      delta_t=0.1, plot=True)
        up_right, down_right = GradSlop(profile_right, threshold=0.2,
                                      iteration=500, sigma=2, kappa=0.05,
                                      delta_t=0.1, plot=True)
        center_left, plateau_left = self._channel_center(up_left, down_left)
        center_right, plateau_right = self._channel_center(up_right, down_right)
        
        f = plt.figure(figsize=(10,10))
        plt.imshow(left, cmap=plt.cm.gray)
        plt.hlines(center_left, 0, 500, 'r', label='center')
        plt.hlines(plateau_left, 0, 500, 'b', label='plateau')
        plt.legend(loc=2)
        plt.show()              
        f = plt.figure(figsize=(10,10))
        plt.imshow(right, cmap=plt.cm.gray)
        plt.hlines(center_right, 0, 500, 'r', label='center')
        plt.hlines(plateau_right, 0, 500, 'b', label='plateau')
        plt.legend(loc=2)
        plt.show()
        
        left_depth, left_end_pts = self._notch_depth(left, center_left, 
                                                     plateau_left, self._scan_to_avg,
                                                     threshold=85, orientation='backward')
        right_depth, right_end_pts = self._notch_depth(right, center_right, 
                                                       plateau_right, self._scan_to_avg,
                                                       threshold=85, orientation='forward')
        right_end_pts[:, :, 0] += split
        left_end_pts[:, :, 0] += left_lim
        depth = np.append(left_depth, right_depth, axis=0)
        end_pts = np.append(left_end_pts, right_end_pts, axis=0)
        count, _, _ = end_pts.shape
        depth = np.reshape(depth,(count,1))
        end_pts = np.reshape(end_pts, (count, 1, 2, 2))
        return count, None, depth, end_pts, ['Horizontal']
    
    def _channel_center(self, up, down):
        
        if len(up) == len(down):
            if up[0] < down[0]:
                center = (up + down) / 2
                plateau = (down[:-1] + up[1:]) / 2       
            else:
                center = (up[:-1] + down[1:]) / 2
                plateau = (up + down) / 2
        elif len(up) - len(down) == 1:
            center = (up[:-1] + down) / 2
            plateau = (up[1:] + down) / 2
        elif len(down) - len(up) == 1:
            center = (up + down[1:]) /2
            plateau = (up + down[:-1]) / 2
        return center.astype(int), plateau.astype(int)
        
    def _notch_depth(self, image, center, plateau, scan, threshold=85,
                     axis=0, iteration=300, sigma=5, delta_t=0.1, orientation='forward'):
        y_lim, x_lim = image.shape
        if scan % 2 == 0:
            before = scan // 2
            after = scan //2
        else:
            before = scan // 2
            after = scan // 2 + 1
        edge_center =[]
        edge_plateau = []
        for idx in plateau:
            line_seg = np.sum(image[idx-before:idx+after,:], axis=axis) / scan
            edge_plateau.append(LGSemEdge(np.arange(x_lim), line_seg, 
                                          threshold=threshold, finess=0.05, 
                                          orientation=orientation, plot=True))
        
        for idx in center:
            line_seg = np.sum(image[idx-before:idx+after,:], axis=axis) / scan
#            kappa = 0.2 * (np.max(line_seg) - np.min(line_seg))
#            line_seg = ReguIsoNonlinear(line_seg, iteration, kappa, sigma=sigma, 
#                                        delta_t=0.1)
            edge_center.append(LGSemEdge(np.arange(x_lim), line_seg, 
                                         threshold=threshold, finess=0.05, 
                                         orientation=orientation, plot=True))
#            edge_center.append(LPSemEdge(np.arange(x_lim), line_seg, 
#                                         threshold=threshold, finess=0.05, 
#                                         orientation=orientation, plot=True))
        count = len(edge_center)
        edge_center = np.array(edge_center)
        edge_plateau = np.array(edge_plateau)
        depth = np.zeros(count)
        end = np.zeros(count)
        end_points = []
        if count - len(edge_plateau) == 1:
            end[:-1] = edge_plateau
            end[-1] = edge_plateau[-1]
            depth = np.abs(edge_center - end)
            for i in range(count):
                end_points.append([[edge_center[i], center[i]], [end[i], center[i]]])
        if count == len(edge_plateau):
            depth = np.abs(edge_center - edge_plateau)
            for i in range(count):
                end_points.append([[edge_center[i], center[i]], [edge_plateau[i], center[i]]])
        if len(edge_plateau) - count == 1:
            depth = np.abs(edge_center - edge_plateau[:-1])
            for i in range(count):
                end_points.append([[edge_center[i], center[i]], [edge_plateau[i], center[i]]])
        return depth, np.array(end_points)
        




































