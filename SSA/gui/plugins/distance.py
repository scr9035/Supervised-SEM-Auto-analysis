# -*- coding: utf-8 -*-
import os
import numpy as np
from skimage import measure
from skimage.util.dtype import dtype_range
from ..canvastools import LineTool, LimLineTool
from ..plugins import Plugin
#from ..utils.canvas import BlitManager, EventManager
from ..utils import new_plot

from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QLineEdit, QCheckBox,
                             QGridLayout, QVBoxLayout, QHBoxLayout, QFrame, QScrollArea,
                             QTabWidget, QSizePolicy, QLayout)
from PyQt5.QtCore import Qt


class HVDistance(Plugin):
    """Base (backend level) plugin for CD measurements
    
    Can hardly be used off the shelf. Need to be inherented by frontend plugins
    for auto analysis and correct data tranfer.
    
    Effective measurements are limited to horizontal or vertical lines only.
    """
        
    name = 'Channel CD Measurement'

    def __init__(self, channel_count=0, lvl_count=0, channel_CD=[], end_points=[], 
                 ref_ends=None, add_top_lim=False, add_bot_lim=False, 
                 add_left_lim=False, add_right_lim=False, maxdist=10, 
                 preset_ref=False, limits='image', dock='right', 
                 **kwargs):
        """
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
        super().__init__(layout=QHBoxLayout, dock=dock, **kwargs)
        self.maxdist = maxdist
        self._blit_manager = None
        self._extra_control_widget = []
        self._event_manager = None
        self._limit_type = limits
        self._new_img = False
        # This callable needs to be specified by subclass in order to do auto-analysis
        self._auto_CD = None
        
        self._image = None
        self._full_image = None
        
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
        self._preset_ref = preset_ref
        
        self._lim_artists = []
        self._ref_artists = []
        self.set_plugin_param()
        
        self._calib = np.nan
        self._show_profile = False
        self._setting_folder = 'Settings\\'
        if not os.path.exists(self._setting_folder):
            os.makedirs(self._setting_folder)

            
    def set_plugin_param(self, channel_count=0, lvl_count=0, channel_CD=[], 
                         end_points=[], line_modes=[], ref_ends=None):
        """Reset all the stored data
        
        Reset happens when plugin is initiated, a new image is loaded, an auto-analysis
        is made, or any high level parameters are changed which can affect all measurements.
        """
        
        self._ref_ends = ref_ends
        self._channel_count = channel_count    
        self._lvl_count = lvl_count        
        self._cdline_ends = end_points
        self._cd_data = np.array(channel_CD, dtype=float)         
        self._line_modes = line_modes
        
        self.figures  = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
        self.axs      = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
        self.canvas   = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
        self.cd_lines = [[None for _ in range(self._lvl_count)] for _ in range(self._channel_count)]
            
    def _on_new_image(self, image, same_img=False):
        """Override this method to specify requirements of any frontend plugins 
        when new images are loaded.
        """
        super()._on_new_image(image)
        if not same_img:
            self.remove_ref_artists()
            self._ref_line = None
            self._ref_moved = False
        self._new_img = True
        self._full_image = image
        self._crop_img()
        self.set_plugin_param()
        self._set_pre_ref()            
        self.reset_plugin()
    
    def _receive_calib(self, calib):
        self._calib = calib

    def reset_plugin(self):
        # Reset the all widgets based on the plugin information
        self.display_measurements()
        self.data_transfer()
    
    def display_measurements(self):
        self._plot_artists()
        if self._show_profile:
            self._draw_profiles()
        
    def data_transfer(self):
        """Override by subclass. Pass data to the main image viewer data table"""
        pass
    
    def attach(self, image_viewer):
        """Attach the layout of the plugin
        
        Two sections are introduced: control section and plot section. More button
        and options can be added to the control or plot section.
        """
        super().attach(image_viewer)
        # Two main sections used in this plugin
        self.control_section = QWidget()
        self.control_section.setLayout(QVBoxLayout())
        self.control_section.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        
        self.plot_container = QScrollArea()        
        self.plot_container.setFrameShape(QFrame.NoFrame)
        self.plot_container.setWidgetResizable(True)
        self.plot_container.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_section = QWidget()
        self.plot_container.setWidget(self.plot_section)
        self.plot_section.setLayout(QGridLayout())
        if not self._show_profile:
            self.plot_container.hide() 
            
        control_layout = self.control_section.layout()
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(hline)
        
        self._profile_cb = QCheckBox('Show Profile')
        if self._show_profile:
            self._profile_cb.toggle()
        self._profile_cb.stateChanged.connect(self._display_plot_section)
        control_layout.addWidget(self._profile_cb)    
        
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(hline)
        
        # User can add limit line
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
        
        # Add update button in control section, to update all lines if 
        # reference line is changed
        update_btn = QPushButton('Update', self)
        update_btn.setToolTip('Update CD lines after changing reference line')
        update_btn.clicked.connect(self._update_data)
        control_layout.addWidget(update_btn)
        # Add delete button in control section
        del_btn = QPushButton('Delete', self)
        del_btn.setToolTip('Delete selected CD line')
        del_btn.clicked.connect(self._delete_line)
        control_layout.addWidget(del_btn)         
        
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(hline)
        
        preset_ref_check = QCheckBox('Preset Reference')
        if self._preset_ref is None:
            preset_ref_check.setDisabled(True)
        elif self._preset_ref:
            preset_ref_check.toggle()
        preset_ref_check.stateChanged.connect(self._use_preset_ref)
        control_layout.addWidget(preset_ref_check)
        
        for widget in self._extra_control_widget:
            control_layout.addWidget(widget)
        control_layout.addStretch(1)
        
        self.layout.addWidget(self.control_section)
        self.layout.addWidget(self.plot_container)

    def _display_plot_section(self, state):
        if state == Qt.Checked:
            self._show_profile = True
            self.plot_container.show()
            self._draw_profiles()
        else:
            self._show_profile = False
            self.plot_container.hide()
            while self.plot_section.layout().count():
                item = self.plot_section.layout().takeAt(0)
                widget = item.widget()
                widget.deleteLater()

    def _draw_profiles(self):
        # Delete all plots in the plot section
        while self.plot_section.layout().count():
            item = self.plot_section.layout().takeAt(0)
            widget = item.widget()
            widget.deleteLater()
            
        fig_width = 100
        self.plot_container.setMinimumWidth(self._lvl_count * fig_width * 2)
        if self._channel_count > 15:
            fig_height = 100
            size_policy = QSizePolicy.MinimumExpanding
        else:
            size_policy = QSizePolicy.Expanding
            fig_height = self.height() / (self._channel_count + 1)
                
        for i in range(self._channel_count):
            for j in range(self._lvl_count):
                self.figures[i][j], self.axs[i][j] = new_plot(SizePolicy=size_policy)
                self.canvas[i][j] = self.figures[i][j].canvas
                self.figures[i][j].set_figwidth(fig_width / float(self.figures[i][j].dpi))
                self.figures[i][j].set_figheight(fig_height / float(self.figures[i][j].dpi))
                qpalette = QPalette()
                qcolor = qpalette.color(QPalette.Window)
                bgcolor = qcolor.toRgb().value()
                if np.isscalar(bgcolor):
                    bgcolor = str(bgcolor / 255.)
                self.figures[i][j].patch.set_facecolor(bgcolor)
                self.plot_section.layout().addWidget(self.canvas[i][j], *[i,j])
        
        if self._image is not None:
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
                        self._reset_lines(self.axs[i][j], self._full_image, self.cd_lines[i][j])
            self._autoscale_view()
    
    def _plot_ref_line(self):
        # Add the reference line if the end points are given
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
        self.image_viewer.redraw()
    
    def _plot_artists(self):
        self._plot_ref_line()        
        # Add all other lines for distance measurements
        for i in range(self._channel_count):
            for j in range(self._lvl_count):
                if self._cdline_ends[i][j] is not None:
                    mode = self._line_modes[j]
                    # If no cd_line, can put None in the list
                    if mode == 'Horizontal':
                        cd_handle_prop = dict(marker='|', markersize=7, color='r', 
                                              mfc='r', ls='none', alpha=1, visible=True)
                    elif mode == 'Vertical':
                        cd_handle_prop = dict(marker='_', markersize=7, color='r', 
                                              mfc='r', ls='none', alpha=1, visible=True)
                    else:
                        print('Wrong Mode')
                    self.cd_lines[i][j] = LineTool(self.image_viewer,
                                                     mode=mode,
                                                     maxdist=self.maxdist, 
                                                     on_move=self.cd_line_changed,
                                                     handle_props=cd_handle_prop)
                    # Add 0.5 to move the point to the center of the pixel
                    # at here or in the line class
                    if self._add_top_lim:
                        self._cdline_ends[i][j][0][1] += self._top_lim_lvl
                        self._cdline_ends[i][j][1][1] += self._top_lim_lvl                        
                    self.cd_lines[i][j].end_points = self._cdline_ends[i][j]
                    self.artists.append(self.cd_lines[i][j])
                else:
                    self.cd_lines[i][j] = None
        self.image_viewer.redraw()
        
    
    def cd_line_changed(self, end_points):
        self._update_act_line()
        if self._show_profile:
            self._autoscale_view()    
            self.redraw()
            
    def _autoscale_view(self):
        # Auto scale all the axis
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
                        self._cd_data[i][j] = None
                        if self._show_profile:
                            self.axs[i][j].clear()
                            self.canvas[i][j].draw()
        self.data_transfer()
  
    def _update_data(self):
        """callback of the "update" button
        
        Let auto-analysis algorithm do everything first. Full auto-analysis is 
        performed when a new image is loaded, or the reference line is manually
        moved
        """
        if self._auto_CD is None:
            return                
        if not self._ref_moved and not self._new_img:
            return
        try:
            self._crop_img()
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
            channel_count, ref_line_y, channel_CD, cd_points, line_modes = self._auto_CD(self._image, 
                                                                                         interface=ends)
            lvl_count = len(channel_CD[0])
        except Exception as e:
#            print(str(e))
            ref_line_y = int(y_lim/2)
            channel_count = 0
            lvl_count = 0
            channel_CD = []
            cd_points = []
            line_modes = []
        
        if ref_line_y is None: # when plugin doesn't need reference line
            ref_ends = None 
        else:
            if self._add_top_lim:
                ref_line_y += self._top_lim_lvl
            if x_lim > 10:
                ref_ends = [(10, ref_line_y), (x_lim-10, ref_line_y)]
            else:
                ref_ends = [(0, ref_line_y), (x_lim-1, ref_line_y)]      
        
        self.set_plugin_param(channel_count=channel_count, lvl_count=lvl_count, 
                              channel_CD=channel_CD, end_points=cd_points, 
                              line_modes=line_modes, ref_ends=ref_ends)
        self.reset_plugin()
    

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
                        
    
    def _reset_lines(self, ax, image, line_tool, margin=5):
        # Clear lines out
        y_lim, x_lim = image.shape
        if ax is not None:
            for line in ax.lines:
                ax.lines = []
        # Draw the line
        # If the line is perfectly horizontal, show the line within the markers
        # and also show some extra "margin". The position of markers are shown
        # as the red lines
        p1, p2 = line_tool.end_points
        if line_tool.mode == 'Horizontal' and  p1[1] == p2[1]:
            left_peak = int(round(min(p1[0], p2[0])))
            right_peak = int(round(max(p1[0], p2[0])))
            left_edge = max(left_peak-margin, 0)
            right_edge = min(right_peak+margin, x_lim-1)
            scan_data = image[int(round(p1[1])), left_edge:right_edge]
            if ax is not None:
                ax.plot(np.arange(left_edge, right_edge), scan_data, 'k-')
                ax.plot([left_peak, left_peak], [min(scan_data), max(scan_data)], 'r-')
                ax.plot([right_peak, right_peak], [min(scan_data), max(scan_data)], 'r-')
                ax.relim()
            return right_peak - left_peak
        elif line_tool.mode == 'Vertical' and p1[0] == p2[0]:
            top_peak = int(round(min(p1[1], p2[1])))
            bot_peak = int(round(max(p1[1], p2[1])))
            top_edge = max(top_peak-margin, 0)
            bot_edge = min(bot_peak+margin, y_lim-1)
            scan_data = image[top_edge:bot_edge, int(round(p1[0]))]
            if ax is not None:
                ax.plot(np.arange(top_edge, bot_edge), scan_data, 'k-')
                ax.plot([top_peak, top_peak], [min(scan_data), max(scan_data)], 'r-')
                ax.plot([bot_peak, bot_peak], [min(scan_data), max(scan_data)], 'r-')
                ax.relim()
            return bot_peak - top_peak
        else:
            # Non-horizontal line. Use the extrapolation to give the line
            # profile
            scan_data = measure.profile_line(image, *line_tool.end_points[:, ::-1])
            if ax is not None:
                ax.plot(scan_data, 'k-')
                ax.relim()
            return None

    def _use_top_lim(self, state):
        if self._full_image is not None:
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
        else:
            if state == Qt.Checked:
                self._add_top_lim = True
            else:
                self._add_top_lim = False
                    
    def _use_bot_lim(self, state):
        if self._full_image is not None:
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
        else:
            if state == Qt.Checked:
                self._add_bot_lim = True
            else:
                self._add_bot_lim = False
    
    def _use_right_lim(self, state):
        if self._full_image is not None:
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
        else:
            if state == Qt.Checked:
                self._add_right_lim = True
            else:
                self._add_right_lim = False
                
    def _use_preset_ref(self, state):
        if state == Qt.Checked:
            self._preset_ref = True
            self._set_pre_ref()
        else:
            self._preset_ref = False
    
    def _set_pre_ref(self):
        if self._image is not None:
            if self._preset_ref is not None and self._preset_ref:
                if self._ref_ends is None:
                    y_lim, x_lim = self._image.shape              
                    self._ref_ends = [(min(10, x_lim), int(y_lim/2)), (min(x_lim, np.abs(x_lim-10)), int(y_lim/2))]
                    self._plot_ref_line()
                
    def _crop_img(self):
        """Crop the raw image using the limiting lines
        
        This is to modify the image conveniently to facilitate the auto-analysis
        Only top, bottom, and right limitting lines are implemented. Pay extra 
        attention to top and left (in the future) limiting line since they can
        change the measurements.
        """
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
                # Default right limitting line position
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
                # Default bottom limitting line position. This won't be at the very
                # bottom since the label region is usually excluded first.
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
                # Default top limiting line postion.
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