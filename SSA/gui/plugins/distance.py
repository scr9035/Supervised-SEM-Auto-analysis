# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

import numpy as np

from ..canvastools import LineTool, LimLineTool
from skimage import measure
from skimage.util.dtype import dtype_range
from ..plugins import Plugin
#from ..utils.canvas import BlitManager, EventManager
from ..utils import new_plot

from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QLineEdit, QCheckBox,
                             QGridLayout, QVBoxLayout, QFrame)
from PyQt5.QtCore import Qt


class NormalDist(Plugin):
        
    name = 'Channel CD Measurement'

    def __init__(self, channel_count=0, lvl_count=0, channel_CD=[], end_points=[], 
                 ref_ends=None, mode='Horizontal', add_top_lim=False, 
                 add_bot_lim=False, add_left_lim=False, add_right_lim=False,
                 maxdist=10, height=150, width=700, limits='image', dock='right', 
                 **kwargs):
        """
        Cannot be used off the shelf. Need to be inherented for auto analysis and 
        correct data tranfer.
        
        Parameters
        ----------
        dock : string
            Dock position of the plugin relative to the imageViewer
        
        channel_count : int
            Number of total channels
        
        lvl_count : int
            Number of levels
            
        end_points : list 
            Shape of Channel_count x levels (x 2), each element is [p1, p2], which
            are left and right edge point.
            
        ref_line : [p1, p2]
            p1 is (x1, y1), p2 is (x2, y2)
            
        """
        super().__init__(height=height, width=width, dock=dock, **kwargs)
        self.maxdist = maxdist
        self._height = height
        self._width = width
        self._mode = mode
        self._blit_manager = None
        self._extra_control_widget = []
        self._event_manager = None
        self._limit_type = limits
        self._new_img = False
        self._auto_CD = None # This callable needs to be specified by subclass
        
        self._ref_line = None
        self._ref_moved = False
        
        self._add_bot_lim = add_bot_lim
        self._add_right_lim = add_right_lim
        self._add_top_lim = add_top_lim
        self._add_left_lim = add_left_lim
        self._top_lim_line = None
        self._left_lim_line = None
        self._bot_lim_line = None
        self._right_lim_line = None
        self._top_lim_lvl = None
        self._right_lim_lvl = None
        self._bot_lim_lvl = None
        
        self._lim_artists = []
        self._ref_artists = []
        self.set_plugin_param()
        self._show_profile = True # Whether to show the profile plot
            
    def set_plugin_param(self, channel_count=0, lvl_count=0, channel_CD=[], end_points=[], 
                 ref_ends=None):
        # DYL: reset related data
        
        self._ref_ends = ref_ends
        self._channel_count = channel_count    
        self._lvl_count = lvl_count        
        self._cdline_ends = end_points
        self._cd_data = np.array(channel_CD, dtype=float)         
        
        self.figures  = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
        self.axs      = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
        self.canvas   = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
        self.cd_lines = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
            
    def _on_new_image(self, image, same_img=False):
        """Override this method to update your plugin for new images."""
        super()._on_new_image(image)
        if not same_img:
            self.remove_ref_artists()
            self._ref_line = None
            self._ref_moved = False
        self._new_img = True
        self._full_image = image
        self._corp_img()
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
                self._right_lim_line.end_points = [[round(x_lim*0.8), 0], 
                                                   [round(x_lim*0.8), y_lim-1]]
                self._right_lim_lvl = round(x_lim*0.8)
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
                if self._bot_lim_lvl is not None:
                    if self._top_lim_lvl >= self._bot_lim_lvl:
                        raise RuntimeError
                self._image = self._image[self._top_lim_lvl:,:]

    def reset_plugin(self):
        # DYL: reset the all widgets based on the plugin information
        self.plot_profile()
        self.data_transfer()
    
    def data_transfer(self):
        """Override by subclass. Pass CD data to the main image viewer data table"""
        pass
    
    def attach(self, image_viewer):
        """Attach the layout of the plugin
        
        Two sections are introduced: control section and plot section. More button
        and options can be added to the control or plot section.
        """
        super().attach(image_viewer)
        # DYL: Two main sections used in this plugin
#        self.control_section = QtWidgets.QVBoxLayout()
        self.control_section = QWidget()
        self.control_section.setLayout(QVBoxLayout())
        self.plot_section = QWidget()
        self.plot_section.setLayout(QGridLayout())
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
        
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(hline)
        
        top_lim_check = QCheckBox('Top Limit Line')
        if self._add_top_lim:
            top_lim_check.toggle()
        top_lim_check.stateChanged.connect(self._use_top_lim)
        control_layout.addWidget(top_lim_check)
        
        bot_lim_check = QCheckBox('Bot Limit Line')
        if self._add_bot_lim:
            bot_lim_check.toggle()
        bot_lim_check.stateChanged.connect(self._use_bot_lim)
        control_layout.addWidget(bot_lim_check)
        
        
        right_lim_check = QCheckBox('Right Limit Line')
        if self._add_right_lim:
            right_lim_check.toggle()
        right_lim_check.stateChanged.connect(self._use_right_lim)
        control_layout.addWidget(right_lim_check)
        
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(hline)
        
        # DYL: Add update button in control section, to update all lines if 
        # reference line is changed
        update_btn = QPushButton('Update', self)
        update_btn.setToolTip('Update CD lines after changing reference line')
        update_btn.clicked.connect(self._update_data)
        update_btn.resize(update_btn.sizeHint())
        control_layout.addWidget(update_btn)
        # DYL: Add delete button in control section
        del_btn = QPushButton('Delete', self)
        del_btn.setToolTip('Delete selected CD line')
        del_btn.clicked.connect(self._delete_line)
        del_btn.resize(del_btn.sizeHint())
        control_layout.addWidget(del_btn)         
        
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(hline)
        
        for widget in self._extra_control_widget:
            control_layout.addWidget(widget)
        control_layout.addStretch(1)
        
        self.layout.addWidget(self.control_section, 0, 0)
        self.layout.addWidget(self.plot_section, 0, 1)
          
    def _display_profiles(self, state):
        if state == Qt.Checked:
            self._show_profile = True
            self.plot_section.show()
        else:
            self._show_profile = False
            self.plot_section.hide()
    
    def _use_top_lim(self, state):
        if state == Qt.Checked:
            if not self._add_top_lim:
                self._add_top_lim = True
                self._on_new_image(self._full_image)
        else:
            if self._add_top_lim:
                self._add_top_lim = False
                self._lim_artists.remove(self._top_lim_line)
                self._top_lim_line.remove()
                self._top_lim_line = None
                self._top_lim_lvl = None
                self._on_new_image(self._full_image)
                    
    def _use_bot_lim(self, state):
        if state == Qt.Checked:
            if not self._add_bot_lim:
                self._add_bot_lim = True
                self._on_new_image(self._full_image)
        else:
            if self._add_bot_lim:
                self._add_bot_lim = False
                self._lim_artists.remove(self._bot_lim_line)
                self._bot_lim_line.remove()
                self._bot_lim_line = None
                self._bot_lim_lvl = None
                self._on_new_image(self._full_image)
    
    def _use_right_lim(self, state):
        if state == Qt.Checked:
            if not self._add_right_lim:
                self._add_right_lim = True
                self._on_new_image(self._full_image)
        else:
            if self._add_right_lim:
                self._add_right_lim = False
                self._lim_artists.remove(self._right_lim_line)
                self._right_lim_line.remove()
                self._right_lim_line = None
                self._right_lim_lvl = None
                self._on_new_image(self._full_image)
    
    def plot_profile(self):
        """Add all profile plots
        """       
        # DYL: Delete all plots in the plot section
        while self.plot_section.layout().count():
            item = self.plot_section.layout().takeAt(0)
            widget = item.widget()
            widget.deleteLater()
        
        # DYL: Arrange all the canvas
        fig_width = self._width / (self._lvl_count + 1)
        fig_height = self._height / (self._channel_count + 1)
        for i in range(self._channel_count):
            for j in range(self._lvl_count):
                self.figures[i][j], self.axs[i][j] = new_plot()
                self.canvas[i][j] = self.figures[i][j].canvas
                self.figures[i][j].set_figwidth(fig_width / float(self.figures[i][j].dpi))
                self.figures[i][j].set_figheight(fig_height / float(self.figures[i][j].dpi))
                qpalette = QPalette()
                qcolor = qpalette.color(QPalette.Window)
                bgcolor = qcolor.toRgb().value()
                if np.isscalar(bgcolor):
                    bgcolor = str(bgcolor / 255.)
                self.figures[i][j].patch.set_facecolor(bgcolor)
                # DYL: Position of the plots
                self.plot_section.layout().addWidget(self.canvas[i][j], *[i,j])
                
        # DYL: Plot all CD lines and plot their profile
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
        
        # DYL: Add the reference line if the end points are given
        if self._ref_ends is not None:
            if self._ref_line is None:
                ref_handle_prop = dict(marker='.', markersize=7, color='r', mfc='r', 
                                       ls='none', alpha=1, visible=True)
                self._ref_line = LineTool(self.image_viewer, mode='Horizontal',
                                          line_props=dict(linestyle=':'),
                                          maxdist=self.maxdist, 
                                          on_move=self.ref_line_changed, 
                                          handle_props=ref_handle_prop)
                self._ref_line.end_points = self._ref_ends
                self._ref_artists.append(self._ref_line) 
        
        # DYL: Add all other lines for distance measurements
        for i in range(self._channel_count):
            for j in range(self._lvl_count):
                if self._cdline_ends[i][j] is not None:
                    # DYL: if don't need a cd_line, can put None in the list
                    if self._mode == 'Horizontal':
                        cd_handle_prop = dict(marker='|', markersize=7, color='r', 
                                              mfc='r', ls='none', alpha=1, visible=True)
                    elif self._mode == 'Vertical':
                        cd_handle_prop = dict(marker='_', markersize=7, color='r', 
                                              mfc='r', ls='none', alpha=1, visible=True)
                    else:
                        print('Wrong Mode')
                    self.cd_lines[i][j] = LineTool(self.image_viewer,
                                                     mode=self._mode,
                                                     maxdist=self.maxdist, 
                                                     on_move=self.cd_line_changed,
                                                     handle_props=cd_handle_prop)
                    # DYL: add 0.5 to move the point to the center of the pixel
                    # at here or in the line class
                    if self._add_top_lim:
                        self._cdline_ends[i][j][0][1] += self._top_lim_lvl
                        self._cdline_ends[i][j][1][1] += self._top_lim_lvl                        
                    self.cd_lines[i][j].end_points = self._cdline_ends[i][j]
                    self.artists.append(self.cd_lines[i][j])
                    self._reset_lines(self.axs[i][j], self._full_image, self.cd_lines[i][j])
                else:
                    self.cd_lines[i][j] = None
        self._autoscale_view()
        self.image_viewer.redraw()
        
    
    def cd_line_changed(self, end_points):
        self._update_act_line()
        self._autoscale_view()    
        self.redraw()
        
    def ref_line_changed(self, end_points):
        self._ref_moved = True
        self._ref_ends = self._ref_line.end_points
        
    def _lim_line_changed(self, lvl):         
        self._new_img = True
                      
    def _delete_line(self):
        """delete selected line """
        for i in range(self._channel_count):
            for j in range(self._lvl_count):
                if self.cd_lines[i][j] is not None:
                    if self.cd_lines[i][j].is_active():
                        self.cd_lines[i][j].delete
                        self.axs[i][j].clear()
                        self.canvas[i][j].draw()
                        self._cd_data[i][j] = None
        self.data_transfer()
  
    def _update_data(self):        
        if self._auto_CD is None:
            return
        if not self._ref_moved and not self._new_img:
            return
        
        try:
            self._corp_img()
        except RuntimeError:
            self.set_plugin_param()
            self.reset_plugin()
            return
        
        if self._ref_line is not None:
            self._ref_moved = False
            ends = self._ref_line.end_points
            if self._add_top_lim:
                ends[0][1] -= self._top_lim_lvl
                ends[1][1] -= self._top_lim_lvl
        else:
            ends = None
        self._new_img = False
        
        y_lim, x_lim = self._image.shape
           
        try:
            channel_count, ref_line_y, channel_CD, cd_points = self._auto_CD(self._image, 
                                                                interface=ends)
            lvl_count = len(channel_CD[0])
        except:
            ref_line_y = int(y_lim/2)
            channel_count = 0
            lvl_count = 0
            channel_CD = []
            cd_points = []
        
        if ref_line_y is None: # DYL: when plugin doesn't need reference line
            ref_ends = None 
        else:
            if self._add_top_lim:
                ref_line_y += self._top_lim_lvl
            if x_lim > 10:
                ref_ends = [(10, ref_line_y), (x_lim-10, ref_line_y)]
            else:
                ref_ends = [(0, ref_line_y), (x_lim-1, ref_line_y)]      
        
        self.set_plugin_param(channel_count=channel_count, lvl_count=lvl_count, 
                              channel_CD=channel_CD, end_points=cd_points, ref_ends=ref_ends)
        self.reset_plugin()
        
#        elif self._ref_moved:
#            # DYL: Calculate the new cd lines and values for the new reference line
#            self._corp_img()
#            channel_count, ref_line_y, channel_CD, cd_points = self._auto_CD(self._image, 
#                                                                interface=self._ref_ends)
#            
#            lvl_count = len(channel_CD[0])
#            # DYL: remove the previous cd lines drawn on the figure
##            self.remove_image_artists()
#            self.set_plugin_param(channel_count=channel_count, lvl_count=lvl_count, 
#                              channel_CD=channel_CD, end_points=cd_points, ref_ends=self._ref_ends)
#            self.reset_plugin()
#        elif self._new_img:
#            # DYL: If a new image is introduced          
#            self._new_img = False
#            self._corp_img()
#            y_lim, x_lim = self._image.shape          
#            channel_count, ref_line_y, channel_CD, cd_points = self._auto_CD(self._image, 
#                                                                interface=None)            
#            lvl_count = len(channel_CD[0])
#            if ref_line_y is None:
#                # DYL: this plugin doesn't need reference line
#                ref_ends = None 
#            else:
#                ref_ends = [(10, ref_line_y), (x_lim-10, ref_line_y)]                 
#            self.set_plugin_param(channel_count=channel_count, lvl_count=lvl_count, 
#                              channel_CD=channel_CD, end_points=cd_points, ref_ends=ref_ends)
#            # DYL: Data transfer is included in the reset 
#            self.reset_plugin()

    def _update_act_line(self):
        """
        Update the cd line plot and table entry if one cd line is manually changed.
        """
        for i in range(self._channel_count):
            for j in range(self._lvl_count):
                if self.cd_lines[i][j] is not None: # if a cd lien exists
                    if self.cd_lines[i][j].is_active() and not self.cd_lines[i][j].is_deleted():
                        cd = self._reset_lines(self.axs[i][j], self._full_image, self.cd_lines[i][j]) 
                        # DYL: Here the cd_data must be FLOAT ARRAY!! CANNOT be INT!!!
                        self._cdline_ends[i][j] = self.cd_lines[i][j].end_points
                        self._cd_data[i][j] = cd
                        self.data_transfer()
                        self.axs[i][j].relim() 
    
    def _reset_lines(self, ax, image, line_tool, margin=5):
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
            left_peak = int(round(min(p1[0], p2[0])))
            right_peak = int(round(max(p1[0], p2[0])))
            left_edge = max(left_peak-margin, 0)
            right_edge = min(right_peak+margin, x_lim-1)
            scan_data = image[int(round(p1[1])), left_edge:right_edge]
            ax.plot(np.arange(left_edge, right_edge), scan_data, 'k-')
            ax.plot([left_peak, left_peak], [min(scan_data), max(scan_data)], 'r-')
            ax.plot([right_peak, right_peak], [min(scan_data), max(scan_data)], 'r-')                
            return right_peak - left_peak
        elif self._mode == 'Vertical' and p1[0] == p2[0]:
            top_peak = int(round(min(p1[1], p2[1])))
            bot_peak = int(round(max(p1[1], p2[1])))
            top_edge = max(top_peak-margin, 0)
            bot_edge = min(bot_peak+margin, y_lim-1)
            scan_data = image[top_edge:bot_edge, int(round(p1[0]))]
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
    
    def remove_lim_artists(self):
        """Remove artists that are connected to the image viewer."""
        for a in self._lim_artists:
            a.remove()
        self._lim_artists = []
    
    def remove_ref_artists(self):
        """Remove artists that are connected to the image viewer."""   
        for a in self._ref_artists:
            a.remove()
        self._ref_artists = []
    
    def clean_up(self):
        self.remove_lim_artists()
        self.remove_ref_artists()
        super().clean_up()
        
        
    def help(self):
        helpstr = ("Plugin to measure normal distance",
                   " ")
        return '\n'.join(helpstr)      