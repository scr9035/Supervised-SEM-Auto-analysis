# -*- coding: utf-8 -*-

import os
import sys
import time
import inspect
import numpy as np
from skimage import img_as_float
from skimage.util.dtype import dtype_range
from skimage.exposure import rescale_intensity
import skimage.external.tifffile as read_tiff
from matplotlib.backends.backend_pdf import PdfPages

#from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
#from ..qt import QtWidgets, Qt
from PyQt5.QtWidgets import (QMainWindow, QLabel, QPushButton, QCheckBox, 
                             QComboBox, QLineEdit, QApplication, QFrame,
                             QMessageBox, QMenu, QWidget, QGridLayout, 
                             QHBoxLayout, QVBoxLayout, QDockWidget, QSplashScreen)
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5 import (QtGui, QtCore)
from ..utils import (dialogs, init_qtapp, figimage, start_qtapp,
                     update_axes_image)
from ..utils.canvas import BlitManager, EventManager
from ...gui import SEMPlugins
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT

__all__ = ['ImageViewer']

def mpl_image_to_rgba(mpl_image):
    """Return RGB image from the given matplotlib image object.

    Each image in a matplotlib figure has its own colormap and normalization
    function. Return RGBA (RGB + alpha channel) image with float dtype.

    Parameters
    ----------
    mpl_image : matplotlib.image.AxesImage object
        The image being converted.

    Returns
    -------
    img : array of float, shape (M, N, 4)
        An image of float values in [0, 1].
    """
    image = mpl_image.get_array()
    if image.ndim == 2:
        input_range = (mpl_image.norm.vmin, mpl_image.norm.vmax)
        image = rescale_intensity(image, in_range=input_range)
        # cmap complains on bool arrays
        image = mpl_image.cmap(img_as_float(image))
    elif image.ndim == 3 and image.shape[2] == 3:
        # add alpha channel if it's missing
        image = np.dstack((image, np.ones_like(image)))
    return img_as_float(image)


class NavigationToolbar(NavigationToolbar2QT):
    """Customized navigation tool bar from mpl
    
    Only choose certain functionality from the original toolbar. Also modify
    the interaction with mouse for better user experience
    
    """
    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in 
                 ('Home', 'Back', 'Forward', 'Pan', 'Zoom')]

    def __init__(self, canvas, parent, coordinates):
        super().__init__(canvas, parent, coordinates=coordinates)
    
    def release(self, event):
        """
        If right mouse button is clicked and PAN/ZOOM is active, toggle them to
        close.
        """
        super().release(event)
        if event.button == 3:
            if self._active == 'PAN':
                self.pan()
                QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        elif event.button == 1:
            if self._active == 'ZOOM':
                self.zoom()
                QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
                
    def resetHistory(self):
        """
        Clear the history so it can be used for different image size together. 
        SEM or STEM images from different tool can possibly have different size.
        """
        self._views.clear()
        self._positions.clear()
    
class ImageViewer(QMainWindow):
    """Backend ImageViewer 

    This viewer is a simple container object that holds a Matplotlib axes
    for showing images. `ImageViewer` doesn't subclass the Matplotlib axes (or
    figure) because of the high probability of name collisions.

    Parameters
    ----------
    image : array
        Image being viewed.

    Attributes
    ----------
    canvas, fig, ax : Matplotlib canvas, figure, and axes
        Matplotlib canvas, figure, and axes used to display image.
    image : array
        Image being viewed. Setting this value will update the displayed frame.
    
    
    Signals:
    ----------
    new_image : PyQtSignal
        Signal that emits the new image when a new image is loaded or when 
        the image is a new image for the current main plugin.
                
    Slot:
    ----------
    _close_main_plugin : connected to the closing signal sent by the main plugin
                override by the derived class to specify the action

    """

    dock_areas = {'top': Qt.TopDockWidgetArea,
                  'bottom': Qt.BottomDockWidgetArea,
                  'left': Qt.LeftDockWidgetArea,
                  'right': Qt.RightDockWidgetArea}

    
#    original_image_changed = Signal(np.ndarray)
    # Signal that the original image has been changed
    new_image = pyqtSignal(np.ndarray)
    self_calib = pyqtSignal(float)
    
# TODO: Add splash screen

    def __init__(self, useblit=True, label_lvl=690):
        '''
        label_lvl : int
            Number of row, from which the image is excluded to exclude the SEM 
            label. It can be adjusted in the UI
        '''
        # Start main loop
        init_qtapp()
        super().__init__()

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.file_menu = QMenu('&File', self)
        self.file_menu.addAction('Open tiff images', self.open_imgs,
                                 Qt.CTRL + Qt.Key_O)
#        self.file_menu.addAction('Save to file', self.save_to_file,
#                                 Qt.CTRL + Qt.Key_S)
        self.file_menu.addAction('Quit', self.close,
                                 Qt.CTRL + Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)
        
        self.help_menu = QMenu('&Help', self)
        self.help_menu.addAction('About SSA', self._about_window, Qt.CTRL + Qt.Key_A)
        self.menuBar().addMenu(self.help_menu)
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        # Boolean to tell if the data is saved
        self._data_saved = None
        # Four attributes to manage multiple images that are loaded together
        self._has_img = False
        self._path_list = None
        self._img_names = None
        self._img_idx = None
        
        # List of tools to include all artists
        self._tools = []

        image = np.ones([768, 1024])            
        self.fig, self.ax = figimage(image)     
        self.useblit = useblit
        if useblit:
            self._blit_manager = BlitManager(self.ax)            
        self.canvas = self.fig.canvas
        self.canvas.setParent(self)
        self.ax.autoscale(enable=False)
        self._image_plot = self.ax.images[0]
        self._update_original_image(image)
        
        # List of plugins
        self.main_plugin = None
        self.assist_plugins = []

        # The key to manage events. All action need to be on the canvas
        self._event_manager = EventManager(self.ax)
        # Add navigation bar from matplotlib, without using coordinates
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=False)
        # The right side panel to include more information and choices
        self.right_panel = QGridLayout()
        # More options in the viewer. 
                
        self._calib = np.nan
        calib_label = QLabel('Auto Calibration: ')
        self._disp_calib = QLabel(str(self._calib * 10**9) + ' nm/pixel')
        
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setFrameShadow(QFrame.Sunken)

        self._label = QCheckBox('Label Level:', self)
        self._label.toggle()
        self._label.stateChanged.connect(self._has_label)
        
        self._label_lvl = label_lvl
        self._enter_label_lvl = QLineEdit()
        self._enter_label_lvl.setText(str(self._label_lvl))
        self._enter_label_lvl.editingFinished.connect(self._change_label_lvl)
        
        vline2 = QFrame()
        vline2.setFrameShape(QFrame.VLine)
        vline2.setFrameShadow(QFrame.Sunken)
        
        self.save_btn = QPushButton('Save Data', self)
        self.save_btn.clicked.connect(self.save_data)
        self.save_btn.resize(self.save_btn.sizeHint())
        self.save_btn.setEnabled(False)
        self.next_btn = QPushButton('Next Image', self)
        self.next_btn.clicked.connect(self._next_img)
        self.next_btn.resize(self.next_btn.sizeHint())
        self.next_btn.setEnabled(False)
        
        self.pdf_record = QCheckBox('Save to PDF', self) # Not yet implemented
        vline3 = QFrame()
        vline3.setFrameShape(QFrame.VLine)
        vline3.setFrameShadow(QFrame.Sunken)
        
        # Create drop box to choose different plugins
        plug_name = QLabel('Choose Plugin:')
        self._choose_plugin = QComboBox()
        self._choose_plugin.addItem('None')
        self._plugin_dict = {}
        all_plugins = inspect.getmembers(SEMPlugins, inspect.isclass)
        for plugin_name, plug_class in all_plugins:
            self._plugin_dict[plugin_name] = plug_class
            self._choose_plugin.addItem(plugin_name)
        if getattr(sys, 'frozen', False):
            try:
                print(sys._MEIPASS)
#                sys.path.append(sys._MEIPASS + '\Plugins')
                print('Try to find extra plugin!!')
                import Plugins
                print('imported')
                extra_plugins = inspect.getmembers(Plugins, inspect.isclass)
                print('The extra plugins are:')
                print(extra_plugins)
                for plugin_name, plug_class in extra_plugins:
                    self._plugin_dict[plugin_name] = plug_class
                    self._choose_plugin.addItem(plugin_name)
            except Exception as e: 
                print(e)
        self._choose_plugin.activated[str].connect(self._choose_main_plugin)  
        
        # The first row of widget, including the tool bar
        first_line = QHBoxLayout()
        first_line.addWidget(self.toolbar)
        first_line.addStretch(1)
        first_line.addLayout(self.right_panel)
        
        # The second row of widget
        sec_line = QHBoxLayout()
        
        sec_line.addWidget(calib_label)
        sec_line.addWidget(self._disp_calib)
        sec_line.addWidget(vline)
        sec_line.addWidget(self._label)
        sec_line.addWidget(self._enter_label_lvl)
        sec_line.addWidget(vline2)
        sec_line.addWidget(self.save_btn)
        sec_line.addWidget(self.next_btn)
        sec_line.addWidget(vline3)
        sec_line.addWidget(self.pdf_record)
        sec_line.addStretch(1)
        sec_line.addWidget(plug_name)
        sec_line.addWidget(self._choose_plugin)
        
        # layout of the main window
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.addLayout(first_line)        
        self.layout.addLayout(sec_line)
        self.layout.addWidget(self.canvas)

        status_bar = self.statusBar()
        self.status_message = status_bar.showMessage
        sb_size = status_bar.sizeHint()
        cs_size = self.canvas.sizeHint()
        self.resize(cs_size.width(), cs_size.height() + sb_size.height())

        self.connect_event('motion_notify_event', self._update_status_bar)

    def _choose_main_plugin(self, plugin_name):
        """
        Choose the proper plugin using the drop down manu
        """ 
        if self.main_plugin is not None:
            self.new_image.disconnect(self.main_plugin._on_new_image)
            self.self_calib.disconnect(self.main_plugin._receive_calib)          
            self.main_plugin.close()
            self.main_plugin = None
        if plugin_name != 'None':
            width = self.width()
            height = self.height()
            new_plug = self._plugin_dict[plugin_name]()
            self += new_plug
            self.resize(width, height)
            self.main_plugin = new_plug
            self._main_plugin_info(new_plug)
            
            # Connect signal and slots between main plugin and image viewer
            self.new_image.connect(self.main_plugin._on_new_image)
            self.self_calib.connect(self.main_plugin._receive_calib)
            
            self.main_plugin.plugin_closed.connect(self._close_main_plugin)
            self.main_plugin.plugin_updated.connect(self._plugin_updated)
            self._refresh()
           
    def __add__(self, plugin):
        """Add plugin to ImageViewer"""
        plugin.attach(self)
        if plugin.dock:
            location = self.dock_areas[plugin.dock]
            dock_location = Qt.DockWidgetArea(location)
            dock_widget = QDockWidget()
            dock_widget.setAttribute(Qt.WA_DeleteOnClose)
            dock_widget.setWidget(plugin)
            dock_widget.setWindowTitle(plugin.name)
            # When plugin is closed, send signal to its corresponding dock
            # to close the dock area
            plugin.plugin_closed.connect(dock_widget.close)
            self.addDockWidget(dock_location, dock_widget)
            horiz = (self.dock_areas['left'], self.dock_areas['right'])
            dimension = 'width' if location in horiz else 'height'
            self._add_widget_size(plugin, dimension=dimension)
        return self
        
    def _add_widget_size(self, widget, dimension='width'):
        widget_size = widget.sizeHint()
        viewer_size = self.frameGeometry()
        dx = dy = 0
        if dimension == 'width':
            dx = widget_size.width()
        elif dimension == 'height':
            dy = widget_size.height()
        w = viewer_size.width()
        h = viewer_size.height()
        self.resize(w + dx, h + dy)

    def open_imgs(self):
        """Open image file and display in viewer.
        When multiple images are chosen, the path and image name are stored in 
        the list.
        """
        path_list = dialogs.open_files_dialog()  
        if path_list is None:
            return
        else:
            self._path_list = path_list
            self._has_img = True
            self._img_idx = 0
            self._img_names = []
            for path in self._path_list:
                self._img_names.append(os.path.basename(path))    
            self.toolbar.resetHistory()
            self._refresh()
            self._show_current_img()
    
    def _next_img(self):
        if self._img_idx is not None and self._img_idx < len(self._img_names)-1:
            self._img_idx += 1
            self.toolbar.resetHistory()
            self._refresh()
               
    def _refresh(self):
        """
        Refresh the image viewer if a new image is loaded into the image 
        viewer or if a new plugin is added.
        """
        # If an image exist when refreshed, emit new_image signal
        if self._has_img:
            img = self._show_current_img()
            self.new_image.emit(img)
            self.self_calib.emit(self._calib)
            self._data_saved = False
            self.save_btn.setEnabled(True)
            if self._img_idx < len(self._img_names) - 1:
                self.next_btn.setEnabled(True)
            else:
                self.next_btn.setEnabled(False)
    
    def _has_label(self, state):
        self._refresh()
    
    def _change_label_lvl(self):
        new_lvl = int(self._enter_label_lvl.text())
        if new_lvl == self._label_lvl:
            return
        else:
            self._label_lvl = new_lvl
            self._refresh()

    def save_data(self):
        """For subclass to implement more on how to save data"""
        self._data_saved = True
        self.save_btn.setEnabled(False)
        if self.pdf_record.isChecked():
            self.save_pdf_record()
    
    def save_pdf_record(self):
        if not os.path.exists('pdf_record'):
            os.makedirs('pdf_record')
        with PdfPages('pdf_record\\' + self._img_names[self._img_idx] + '.pdf') as pdf:
            pdf.savefig(self.fig)
        
    
    def _show_current_img(self):
        self_calib = False
        with read_tiff.TiffFile(self._path_list[self._img_idx]) as tif:
            for page in tif:
                for tag in page.tags.values():
                    if tag.name == 'helios_metadata':
                        if 'Scan' in tag.value:
                            self._calib = tag.value['Scan']['PixelHeight']
                            self_calib = True
                            break
                    if tag.name == 'sem_metadata':
                        if '' in tag.value:
                            self._calib = tag.value[''][3]
                            self_calib = True
                            break
        if not self_calib:
            self._calib = np.nan
        self._disp_calib.setText('%.3f nm/pixel'%(self._calib * 10**9))
        
        image = read_tiff.imread(self._path_list[self._img_idx])
        
        # Some weird incompatibility between Python3.5 and 3.6 
        if len(image.shape) == 3:
            x, y, z = image.shape
            if x == 3:
                image = image[0,:,:]
            elif y == 3:
                image = image[:,0,:]
            elif z == 3:
                image = image[:,:,0]
        if self._label.isChecked():
            lvl = int(self._label_lvl)
            partial = image[:lvl,:].copy()
        else:
            partial = image.copy()
        self._update_original_image(image)
        self._extra_img_info()
        return partial

    def _update_original_image(self, image):
        self.original_image = image       # update displayed image
        self.image = image.copy()
 
#    def save_to_file(self, filename=None):
#        """Save current image to file.
#        """
#        if filename is None:
#            filename = dialogs.save_file_dialog()
#        if filename is None:
#            return
#        if len(self.ax.images) == 1:
#            io.imsave(filename, self.image)
#        else:
#            underlay = mpl_image_to_rgba(self.ax.images[0])
#            overlay = mpl_image_to_rgba(self.ax.images[1])
#            alpha = overlay[:, :, 3]
#
#            # alpha can be set by channel of array or by a scalar value.
#            # Prefer the alpha channel, but fall back to scalar value.
#            if np.all(alpha == 1):
#                alpha = np.ones_like(alpha) * self.ax.images[1].get_alpha()
#
#            alpha = alpha[:, :, np.newaxis]
#            composite = (overlay[:, :, :3] * alpha +
#                         underlay[:, :, :3] * (1 - alpha))
#            io.imsave(filename, composite)

    def closeEvent(self, event):
        self.close()
        
    def _show(self, x=0):
        self.move(x, 0)     
        super().show()
        self.activateWindow()
        self.raise_()

    def show(self, main_window=True):
        """Show ImageViewer and attached plugins.

        This behaves much like `matplotlib.pyplot.show` and `QWidget.show`.
        """
        splash_pix = QPixmap('welcome.png')
        splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.show()
        QApplication.processEvents()
        time.sleep(2)
        self._show()
        if main_window:
            splash.finish(self)
            start_qtapp()
            

    def redraw(self):
        if self.useblit:
            self._blit_manager.redraw()
        else:
            self.canvas.draw_idle()

    @property
    def image(self):
        return self._img

    @image.setter
    def image(self, image):
        """Draw at the same time while setting"""
        self._img = image
        update_axes_image(self._image_plot, image)

        # update display (otherwise image doesn't fill the canvas)
        h, w = image.shape[:2]
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

        # update color range
        clim = dtype_range[image.dtype.type]
        if clim[0] < 0 and image.min() >= 0:
            clim = (0, clim[1])
        self._image_plot.set_clim(clim)

        if self.useblit:
            self._blit_manager.background = None
        self.redraw()

    def reset_image(self):   
        self.image = self.original_image.copy()

    def connect_event(self, event, callback):
        """Connect callback function to matplotlib event and return id."""
        cid = self.canvas.mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id):
        """Disconnect callback by its id (returned by `connect_event`)."""
        self.canvas.mpl_disconnect(callback_id)

    def _update_status_bar(self, event):
        if event.inaxes and event.inaxes.get_navigate():
            self.status_message(self._format_coord(event.xdata, event.ydata))
        else:
            self.status_message('')

    def add_tool(self, tool):
        if self.useblit:
            self._blit_manager.add_artists(tool.artists)
        self._tools.append(tool)
        self._event_manager.attach(tool)

    def remove_tool(self, tool):
        if tool not in self._tools:
            return
        if self.useblit:
            self._blit_manager.remove_artists(tool.artists)
        self._tools.remove(tool)
        self._event_manager.detach(tool)

    def _format_coord(self, x, y):
        # callback function to format coordinate display in status bar
#        x = int(x + 0.5)
#        y = int(y + 0.5)
        try:
            return "[%.1f, %.1f]" % (x, y)
#            return "%4s @ [%4s, %4s]" % (self.image[y, x], x, y)
        except IndexError:
            return ""

    def _main_plugin_info(self, plugin):
        """Override by subclass to include more information about plugin, such 
            as more signals
        """
        pass
    
    def _about_window(self):
        about = QMessageBox()
        about.setIcon(QMessageBox.Information)
        about.setTextFormat(QtCore.Qt.RichText)
        info = ['<br> SSA version Beta 0.11 <br>',
                'The Supervised SEM/STEM Auto-analysis software',
                'Copyright &copy; SSA project contributors',
                'Licensed under the terms of MIT License',
                '',
                'Created by Dongyao Li',
                'Developed and maintained by the SSA project contributors',
                '',
                'The software is under continuous development. For more' 
                ' information please contact: ' +
                "<a href='mailto:{0}'>{0}</a><br><br>".format('dongyao.li@lamresearch.com') ]
        about.setText('<br>'.join(info))
        about.setWindowTitle('About Supervised SEM Auto-analysis')
        about.setStandardButtons(QMessageBox.Ok)
        about.exec_()
        
    
    @pyqtSlot()
    def _close_main_plugin(self):
        """Override by subclass to control reaction when main plugin is closed"""
        pass
    
    @pyqtSlot()
    def _plugin_updated(self):
        """Override by subclass to control reaction when main plugin is updated
        when any parameters in the plugin are changed, the viewer redraw
        """
        self.redraw()
    
    def _extra_img_info(self):
        """inherent by subclass to include more information about image"""
        pass