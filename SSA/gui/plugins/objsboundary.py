# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li


from PyQt5 import QtGui, QtWidgets
from ..canvastools import BoundaryTool
from PyQt5.QtWidgets import (QWidget, QPushButton, QAction, QTableWidget, QTableWidgetItem,
                             QLabel, QLineEdit, QCheckBox)
from PyQt5.QtCore import QCoreApplication, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
import numpy as np
from skimage.util.dtype import dtype_range
from skimage import measure

from ..utils import new_plot
from .base import Plugin
from ..canvastools import LineTool, LimLineTool

class Boundary(Plugin):
    
    def __init__(self, maxdist=10, height=150, width=700, limits='image', 
                 dock='right', mode='Horizontal', **kwargs):
        super().__init__(height=height, width=width, dock=dock, **kwargs)
        self.maxdist = maxdist
        self._height = height
        self._width = width
        self._blit_manager = None
        self._extra_control_widget = []
        self._event_manager = None
        self._limit_type = limits
        self._new_img = False
        self._mode = mode
        self._auto_CD = None # This callable needs to be specified by subclass
        self._auto_boundary = None # This callable needs to be specified by subclass   
        self.set_plugin_param()
        self._show_boundary = False
        self._show_profile = True
        
    def set_plugin_param(self, channel_count=0, lvl_count=0, end_points=[],
                         cd_data=[], ref_CDs_ends=None, obj_count=0, obj_center=[], 
                        boundaries=[], ref_high_ends=None, ref_low_ends=None):
        
        self._channel_count = channel_count        
        self._ref_CD_moved = False
        self._ref_CDs_ends = ref_CDs_ends    
        self._lvl_count = lvl_count  
        self._cdline_ends = end_points
        self._cd_data = np.array(cd_data, dtype=float)       
        self.figures  = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
        self.axs      = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
        self.canvas   = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
        self.cd_lines = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
        
        
        self._obj_count = obj_count
        self._ref_bound_moved = False
        self._ref_high_ends = ref_high_ends
        self._ref_low_ends = ref_low_ends
        self._obj_center = obj_center # DYL: Center of the objects boundary
        self._boundaries = boundaries # DYL: boundary data
        self._objects = [None for _ in range(self._obj_count)]

    def attach(self, image_viewer):
        """Attach the layout of the plugin
        
        Two sections are introduced: control section and plot section. More button
        and options can be added to the control or plot section.
        """
        super().attach(image_viewer)
        # DYL: Two main sections used in this plugin
        self.control_section = QWidget()        
        self.control_section.setLayout(QtWidgets.QVBoxLayout())
        self.plot_section = QWidget()
        self.plot_section.setLayout(QtWidgets.QGridLayout())
        if not self._show_profile:
            self.plot_section.hide()
        
        control_layout = self.control_section.layout()
        control_layout.addWidget(QLabel('Measurement Mode:'))
        mode_disp = QLineEdit()
        mode_disp.setText(self._mode)
        mode_disp.setReadOnly(True)
        control_layout.addWidget(mode_disp)
        
        cb = QCheckBox('Show Profile')
        if self._show_profile:
            cb.toggle()
        cb.stateChanged.connect(self._display_profiles)
        control_layout.addWidget(cb)
        
        cb = QCheckBox('Show Boundaries')
        if self._show_boundary:
            cb.toggle()
        cb.stateChanged.connect(self._display_boundary)
        control_layout.addWidget(cb)
        
        # DYL: Add update button in control section, to update all lines if 
        # reference line is changed
        update_btn = QPushButton('Update', self)
        update_btn.setToolTip('Update boundary after changing reference lines')
        update_btn.clicked.connect(self._update_data)
        update_btn.resize(update_btn.sizeHint())
        control_layout.addWidget(update_btn)
        
        # DYL: Add delete button in control section
        del_btn = QPushButton('Delete', self)
        del_btn.setToolTip('Delete selected objects')
        del_btn.clicked.connect(self._delete_obj)
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
    
    def _corp_img(self):
        self.remove_image_artists()
        
        y_lim, x_lim = self._full_image.shape
        self._image = self._full_image
        
        if self._add_right_lim:
            if self._right_lim_line is None:
                right_lim_handle_prop = dict(marker='<', markersize=7, color='r', mfc='r',
                                            ls='none', alpha=1, visible=True)
                self._right_lim_line = LimLineTool(self.image_viewer, mode='Vertical', 
                                            lim=x_lim, line_props=dict(linestyle='-.'),
                                            maxdist=self.maxdist, 
                                            on_move=self._lim_line_changed,
                                            handle_props=right_lim_handle_prop)
                self._right_lim_line.end_points = [[int(x_lim*0.8), 0], 
                                                   [int(x_lim*0.8), y_lim-1]]
                self._right_lim_lvl = int(x_lim*0.8)
                self._lim_artists.append(self._right_lim_line)
                self._image = self._image[:,:self._right_lim_lvl]
            else:
                self._right_lim_line.limit = x_lim
                self._right_lim_lvl = int(self._right_lim_line.level)
                self._image = self._image[:,:self._right_lim_lvl]
                                  
        if self._add_bot_lim:
            if self._bot_lim_line is None:            
                bot_lim_handle_prop = dict(marker='^', markersize=7, color='r', mfc='r',
                                            ls='none', alpha=1, visible=True)            
                self._bot_lim_line = LimLineTool(self.image_viewer, mode='Horizontal',
                                            lim=y_lim, line_props=dict(linestyle='-.'), 
                                            maxdist=self.maxdist,
                                            on_move=self._lim_line_changed,
                                            handle_props=bot_lim_handle_prop)
                self._bot_lim_line.end_points = [[0, y_lim-1], 
                                                 [x_lim-1, y_lim-1]]
                self._bot_lim_lvl = y_lim - 1
                self._lim_artists.append(self._bot_lim_line)
                self._image = self._image[:self._bot_lim_lvl,:]
            else:
                self._bot_lim_line.limit = y_lim
                self._bot_lim_lvl = int(self._bot_lim_line.level)
                self._image = self._image[:self._bot_lim_lvl,:]
            
        if self._add_top_lim:
            if self._top_lim_line is None:
                top_lim_handle_prop = dict(marker='v', markersize=7, color='r', mfc='r',
                                            ls='none', alpha=1, visible=True)
                self._top_lim_line = LimLineTool(self.image_viewer, mode='Horizontal',
                                            lim=y_lim, line_props=dict(linestyle='-.'), 
                                            maxdist=self.maxdist,
                                            on_move=self._lim_line_changed,
                                            handle_props=top_lim_handle_prop)
                
                self._top_lim_line.end_points = [[0, int(y_lim/4)],
                                                 [x_lim-1, int(y_lim/4)]]
                self._top_lim_lvl = int(y_lim/4)
                self._lim_artists.append(self._top_lim_line)
                self._image = self._image[self._top_lim_lvl:,:]
                
            else:
                self._top_lim_line.limit = y_lim
                self._top_lim_lvl = int(self._top_lim_line.level)
                if self._top_lim_lvl >= self._bot_lim_lvl:
                    raise RuntimeError
                self._image = self._image[self._top_lim_lvl:,:]
        
    def reset_plugin(self):
        # DYL: reset the all widgets based on the plugin information   
        self.plot_boundary()
        self.plot_profile()
        self.data_transfer()
        
    def plot_boundary(self):
        ref_handle_prop = dict(marker='.', markersize=7, color='r', mfc='r', ls='none',
                     alpha=1, visible=True)
        if self._ref_high_ends is not None:
            self._ref_high_line = LineTool(self.image_viewer, mode='Horizontal',
                                      line_props=dict(linestyle=':'),
                                      maxdist=self.maxdist, 
                                      on_move=self._ref_high_changed, 
                                      handle_props=ref_handle_prop)
            self._ref_high_line.set_visible(self._show_boundary)
            self._ref_high_line.end_points = self._ref_high_ends
            self.artists.append(self._ref_high_line) 
        
        if self._ref_low_ends is not None:
            self._ref_low_line = LineTool(self.image_viewer, mode='Horizontal',
                                      line_props=dict(linestyle=':'),
                                      maxdist=self.maxdist, 
                                      on_move=self._ref_low_changed, 
                                      handle_props=ref_handle_prop)
            self._ref_low_line.set_visible(self._show_boundary)
            self._ref_low_line.end_points = self._ref_low_ends
            self.artists.append(self._ref_low_line)        
        for i in range(self._obj_count):    
            self._objects[i] = BoundaryTool(self.image_viewer, maxdist=self.maxdist)
            self._objects[i].set_visible(self._show_boundary)
            self._objects[i].center = self._obj_center[i]
            self._objects[i].boundary = self._boundaries[i]    
            self.artists.append(self._objects[i])
            
    
    def plot_profile(self):
        # DYL: Delete all plots in the plot section
        while self.plot_section.layout().count():
            item = self.plot_section.layout().takeAt(0)
            widget = item.widget()
            widget.deleteLater()
        
        ref_handle_prop = dict(marker='+', markersize=15, color='g', mfc='r', ls='none',
                     alpha=1, visible=True)
        if self._ref_CDs_ends is not None:
            self._ref_CDs_line = LineTool(self.image_viewer, mode='Horizontal',
                                      line_props=dict(linestyle=':'),
                                      maxdist=self.maxdist, 
                                      on_move=self._ref_CDs_changed, 
                                      handle_props=ref_handle_prop)
            self._ref_CDs_line.end_points = self._ref_CDs_ends
            self.artists.append(self._ref_CDs_line)
            
        fig_width = self._width / (self._lvl_count + 1)
        fig_height = self._height / (self._channel_count + 1)
        for i in range(self._channel_count):
            for j in range(self._lvl_count):
                self.figures[i][j], self.axs[i][j] = new_plot()
                self.canvas[i][j] = self.figures[i][j].canvas
                self.figures[i][j].set_figwidth(fig_width / float(self.figures[i][j].dpi))
                self.figures[i][j].set_figheight(fig_height / float(self.figures[i][j].dpi))
                qpalette = QtGui.QPalette()
                qcolor = qpalette.color(QtGui.QPalette.Window)
                bgcolor = qcolor.toRgb().value()
                if np.isscalar(bgcolor):
                    bgcolor = str(bgcolor / 255.)
                self.figures[i][j].patch.set_facecolor(bgcolor)
                # DYL: Position of the plots
                self.plot_section.layout().addWidget(self.canvas[i][j], *[i,j])
        
        image = self._image       
        if self._limit_type == 'image':
            self.limits = (np.min(image), np.max(image))
        elif self._limit_type == 'dtype':
            self.limits = dtype_range[image.dtype.type]
        elif self._limit_type is None or len(self._limit_type) == 2:
            self.limits = self._limit_type
        else:
            raise ValueError("Unrecognized `limits`: %s" % self._limit_type)
        if not self._limit_type is None:
            for i in range(self._channel_count):
                for j in range(self._lvl_count):
                    self.axs[i][j].set_ylim(self.limits)
        
        for i in range(self._channel_count):
            for j in range(self._lvl_count):
                if self._cdline_ends[i][j] is not None:
                    # DYL: if don't need a cd_line, can put None in the list
                    if self._mode == 'Horizontal':
                        cd_handle_prop = dict(marker='|', markersize=7, color='r', mfc='r', ls='none',
                             alpha=1, visible=True)
                    elif self._mode == 'Vertical':
                        cd_handle_prop = dict(marker='_', markersize=7, color='r', mfc='r', ls='none',
                             alpha=1, visible=True)
                    else:
                        print('Wrong Mode')
                    self.cd_lines[i][j] = LineTool(self.image_viewer,
                                                     mode=self._mode,
                                                     maxdist=self.maxdist, 
                                                     on_move=self.cd_line_changed,
                                                     handle_props=cd_handle_prop)
                    self.cd_lines[i][j].end_points = self._cdline_ends[i][j]
                    self.artists.append(self.cd_lines[i][j])
                    self._reset_lines(self.axs[i][j], image, self.cd_lines[i][j])
                else:
                    self.cd_lines[i][j] = None
        self._autoscale_view()
    
    def _update_data(self):
        y_lim, x_lim = self._image.shape
        if self._new_img:
            self._new_img = False
            if self._auto_boundary is not None:
                obj_count, ref_high_y, ref_low_y, obj_center, obj_boundary = \
                    self._auto_boundary(self._image)                
            else:
                ref_high_y = self._ref_high_ends[0][1]
                ref_low_y = self._ref_low_ends[0][1]
                obj_count = self._obj_count
                obj_center = self._obj_center
                obj_boundary = self._boundaries
            
            if self._auto_CD is not None:
                channel_count, ref_cd_y, channel_CD, cd_points = \
                                self._auto_CD(self._image)
                lvl_count = len(channel_CD[0])                
            else:
                channel_count = self._channel_count
                lvl_count = self._lvl_count
                channel_CD = self._cd_data
                cd_points = self._cdline_ends
                if self._ref_CDs_ends is not None:
                    ref_cd_y = self._ref_CDs_ends[0][1]
                else:
                    ref_cd_y = None           
        else:
            if not self._ref_bound_moved and not self._ref_CD_moved:
                return           
            if self._ref_bound_moved:
                if self._auto_boundary is not None:
                    obj_count, ref_high_y, ref_low_y, obj_center, obj_boundary = \
                    self._auto_boundary(self._image, ref_high_pts=self._ref_high_ends, 
                                        ref_low_pts=self._ref_low_ends)
            else:
                ref_high_y = self._ref_high_ends[0][1]
                ref_low_y = self._ref_low_ends[0][1]
                obj_count = self._obj_count
                obj_center = self._obj_center
                obj_boundary = self._boundaries
                
            if self._ref_CD_moved:
                if self._auto_CD is not None:
                    channel_count, ref_cd_y, channel_CD, cd_points = \
                                self._auto_CD(self._image, interface=self._ref_CDs_ends)
                    lvl_count = len(channel_CD[0])
            else:
                channel_count = self._channel_count
                lvl_count = self._lvl_count
                channel_CD = self._cd_data
                cd_points = self._cdline_ends
                if self._ref_CDs_ends is not None:
                    ref_cd_y = self._ref_CDs_ends[0][1]
                else:
                    ref_cd_y = None
                    
        self.remove_image_artists()
        ref_high_ends = [(1, ref_high_y), (x_lim-1, ref_high_y)]
        ref_low_ends = [(1, ref_low_y), (x_lim-1, ref_low_y)]
        
        if ref_cd_y is None:
            # DYL: this plugin doesn't need reference line
            ref_CDs_ends = None 
        else:
            ref_CDs_ends = [(20, ref_cd_y), (x_lim-20, ref_cd_y)]
        self.set_plugin_param(channel_count=channel_count, lvl_count=lvl_count, 
                              end_points=cd_points, cd_data=channel_CD, 
                              ref_CDs_ends=ref_CDs_ends, obj_count=obj_count, 
                              obj_center=obj_center, boundaries=obj_boundary, 
                              ref_high_ends=ref_high_ends, ref_low_ends=ref_low_ends)
        self.reset_plugin()
        
    def _display_boundary(self, state):
        # TODO: calculate and show the boundary of the objects
        if state == Qt.Checked:
            self._show_boundary = True
            for obj in self._objects:
                obj.set_visible(True)
            self._ref_high_line.set_visible(True)
            self._ref_low_line.set_visible(True)
        else:
            self._show_boundary = False
            for obj in self._objects:
                obj.set_visible(False)
            self._ref_high_line.set_visible(False)
            self._ref_low_line.set_visible(False)
    
    def _display_profiles(self, state):
        if state == Qt.Checked:
            self._show_profile = True
            self.plot_section.show()
        else:
            self._show_profile = False
            self.plot_section.hide()
    
    def _delete_obj(self):
        """delete selected object """
        for i in range(self._count):
            if self._objects[i].is_active():
                self._objects[i].delete
                self._objects[i].redraw()
        self.data_transfer()
    
    def _ref_high_changed(self, end_points):
        self._ref_bound_moved = True
        self._ref_high_ends = self._ref_high_line.end_points
    
    def _ref_low_changed(self, end_points):
        self._ref_bound_moved = True
        self._ref_low_ends = self._ref_low_line.end_points
    
    def _ref_CDs_changed(self, end_points):
        self._ref_CD_moved = True
        self._ref_CDs_ends = self._ref_CDs_line.end_points
    
    def cd_line_changed(self, end_points):
        self._update_act_line()
        self._autoscale_view()    
        self.redraw()
    
    def _update_act_line(self):
        """
        Update the cd line plot and table entry if one cd line is manually changed.
        """
        for i in range(self._channel_count):
            for j in range(self._lvl_count):
                if self.cd_lines[i][j] is not None: # if a cd lien exists
                    if self.cd_lines[i][j].is_active() and not self.cd_lines[i][j].is_deleted():
                        cd = self._reset_lines(self.axs[i][j], self._image, self.cd_lines[i][j]) 
                        # DYL: Here the cd_data must be FLOAT ARRAY!! CANNOT be INT!!!
                        self._cd_data[i][j] = cd
                        self.data_transfer()
                        self.axs[i][j].relim()
                        
                        
    def _reset_lines(self, ax, image, line_tool, margin=15):
        # DYL: Clear lines out
        y_lim, x_lim = image.shape
        for line in ax.lines:
            ax.lines = []
        # DYL: Draw the line
        # If the line is perfectly horizontal, show the line within the markers
        # and also show some extra "margin". The position of markers are shown
        # as the red lines
        p1, p2 = line_tool.end_points        
        if self._mode == 'Horizontal' and  p1[1] == p2[1]:
            left_peak = int(min(p1[0], p2[0]))
            right_peak = int(max(p1[0], p2[0]))
            left_edge = max(left_peak-margin, 0)
            right_edge = min(right_peak+margin, x_lim-1)
            scan_data = image[int(p1[1]), left_edge:right_edge]
            ax.plot(np.arange(left_edge, right_edge), scan_data, 'k-')
            ax.plot([left_peak, left_peak], [min(scan_data), max(scan_data)], 'r-')
            ax.plot([right_peak, right_peak], [min(scan_data), max(scan_data)], 'r-') 
            return right_peak - left_peak
        elif self._mode == 'Vertical' and p1[0] == p2[0]:
            top_peak = int(min(p1[1], p2[1]))
            bot_peak = int(max(p1[1], p2[1]))
            top_edge = max(top_peak-margin, 0)
            bot_edge = min(bot_peak+margin, y_lim-1)
            scan_data = image[top_edge:bot_edge, int(p1[0])]
            ax.plot(np.arange(top_edge, bot_edge), scan_data, 'k-')
            ax.plot([top_peak, top_peak], [min(scan_data), max(scan_data)], 'r-')
            ax.plot([bot_peak, bot_peak], [min(scan_data), max(scan_data)], 'r-')
            return bot_peak - top_peak
        else:
            # DYL: Non-horizontal line. Use the extrapolation to give the line
            # profile
            scan_data = measure.profile_line(image, *line_tool.end_points[:, ::-1])
            ax.plot(scan_data, 'k-')
            return None
    
    def _autoscale_view(self):
        # DYL: Auto scale all the axis
        if self.limits is None:
            for i in range(self._channel_count):
                for j in range(self._lvl_count):
                    self.axs[i][j].autoscale_view(tight=True)
        else:
            for i in range(self._channel_count):
                for j in range(self._lvl_count):
                    self.axs[i][j].autoscale_view(scaley=False, tight=True)

    def redraw(self):
        """Redraw plot."""
        for i in range(self._channel_count):
            for j in range(self._lvl_count):
                if self.cd_lines[i][j] is not None:
                    if self.cd_lines[i][j].is_active():
                        self.canvas[i][j].draw_idle()
    
    def data_transfer(self):
        pass