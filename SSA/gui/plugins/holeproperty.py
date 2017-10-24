# -*- coding: utf-8 -*-

# TODO: implement the capping count plugin

from .base import Plugin
from PyQt5 import QtGui, QtWidgets
from ..canvastools import PatchTool
from PyQt5.QtWidgets import (QWidget, QPushButton, QAction, QTableWidget, QTableWidgetItem,
                             QLabel, QLineEdit, QCheckBox)
from PyQt5.QtCore import QCoreApplication, Qt, pyqtSignal
from PyQt5.QtGui import QIcon

class HoleProperty(Plugin):
    """Base plugin for top-down hole like measurements. 
    
    Early stage development
    """
    def __init__(self, maxdist=10, height=150, width=700, limits='image', 
                 dock='right', **kwargs):
        super().__init__(height=height, width=width, dock=dock, **kwargs)
        self.maxdist = maxdist
        self._height = height
        self._width = width
        self._blit_manager = None
        self._extra_control_widget = []
        self._event_manager = None
        self._limit_type = limits
        self._new_img = False
        self._auto_holes = None 
        self.set_plugin_param()
        
        self._patch_handle_prop = dict(marker='+', markersize=7, color='w', mfc='r', 
                                 ls='none', alpha=1, visible=True)
        
        self._show_statistics = True # To implement further based on needs
        
    def set_plugin_param(self, count=0, centers=[], majors=[], minors=[], angles=[]):
        self._count = count
        self._centers = centers
        self._majors = majors
        self._minors = minors
        self._angles = angles
        self._patches = [None for _ in range(self._count)]
        
    def attach(self, image_viewer):
        """Attach the layout of the plugin
        
        Two sections are introduced: control section and plot section. More button
        and options can be added to the control or plot section.
        """
        super().attach(image_viewer)
        # Two main sections used in this plugin
        self.control_section = QWidget()        
        self.control_section.setLayout(QtWidgets.QVBoxLayout())
        self.plot_section = QWidget()
        self.plot_section.setLayout(QtWidgets.QGridLayout())
        
        control_layout = self.control_section.layout()
        
        # Add update button in control section, to update all lines if 
        # reference line is changed
        update_btn = QPushButton('Update', self)
        update_btn.setToolTip('Update CD lines after changing reference line')
        update_btn.clicked.connect(self._update_all_patches)
        update_btn.resize(update_btn.sizeHint())
        control_layout.addWidget(update_btn)
        
        # Add delete button in control section
        del_btn = QPushButton('Delete', self)
        del_btn.setToolTip('Delete selected geometry')
        del_btn.clicked.connect(self._delete_patch)
        del_btn.resize(del_btn.sizeHint())
        control_layout.addWidget(del_btn)         
        
        for widget in self._extra_control_widget:
            control_layout.addWidget(widget)
        control_layout.addStretch(1)
        
        self.layout.addWidget(self.control_section, 0, 0)
        self.layout.addWidget(self.plot_section, 0, 1)
        
    def _on_new_image(self, image):
        """Override this method to update your plugin for new images."""
        super()._on_new_image(image)
        self._new_img = True
        self._image = image
        self.set_plugin_param()
        self.reset_plugin()
    
    def _receive_calib(self, calib):
        self._calib = calib
    
    def reset_plugin(self):
        # Reset the all widgets based on the plugin information   
        self.plot_patches()
        self.data_transfer()
        
    def plot_patches(self):
        # Delete all plots in the plot section
        while self.plot_section.layout().count():
            item = self.plot_section.layout().takeAt(0)
            widget = item.widget()
            widget.deleteLater()
        
        
        for i in range(self._count):            
            self._patches[i] = PatchTool(self.image_viewer, maxdist=self.maxdist, 
                         handle_props=self._patch_handle_prop)
            self._patches[i].center = self._centers[i]
            self._patches[i].major = self._majors[i]
            self._patches[i].minor = self._minors[i]
            self._patches[i].angle = self._angles[i]
            self.artists.append(self._patches[i])
    
    def _update_all_patches(self):                
        if self._auto_holes is None:
            return      
        elif self._new_img:
            # If a new image is introduced          
            self._new_img = False
            y_lim, x_lim = self._image.shape
            count, center, major, minor, angle  = self._auto_holes(self._image)                           
            self.set_plugin_param(count=count, centers=center, majors=major,
                                  minors=minor, angles=angle)
            # Data transfer is included in the reset 
            self.reset_plugin()
    
    def _delete_patch(self):
        """delete selected line """
        pass
    
    def data_transfer(self):
        """Override by subclass. Pass CD data to the main image viewer data table"""
        pass