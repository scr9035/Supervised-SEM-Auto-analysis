import numpy as np
from matplotlib import lines
from .base import CanvasToolBase, ToolHandles
from PyQt5.QtWidgets import QApplication
from PyQt5 import (QtGui, QtCore)


__all__ = ['LineTool', 'ThickLineTool']


class LineTool(CanvasToolBase):
    """Widget for line selection in a plot.

    Parameters
    ----------
    manager : Viewer or PlotPlugin.
        Skimage viewer or plot plugin object.
    on_move : function
        Function called whenever a control handle is moved.
        This function must accept the end points of line as the only argument.
    on_release : function
        Function called whenever the control handle is released.
    on_enter : function
        Function called whenever the "enter" key is pressed.
    maxdist : float
        Maximum pixel distance allowed when selecting control handle.
    line_props : dict
        Properties for :class:`matplotlib.lines.Line2D`.
    handle_props : dict
        Marker properties for the handles (also see
        :class:`matplotlib.lines.Line2D`).

    Attributes
    ----------
    end_points : 2D array
        End points of line ((x1, y1), (x2, y2)).
    level : float
        Level of the line. Depends on the mode of the line
    """
                 
    def __init__(self, manager, mode='Horizontal', on_move=None, on_mouse_release=None, 
                 on_key_press=None, on_key_release=None, maxdist=20, 
                 line_props=None, handle_props=None, **kwargs):
        super().__init__(manager, on_move=on_move, on_mouse_release=on_mouse_release,
                                  on_key_press=on_key_press, 
                                  on_key_release=on_key_release,
                                  **kwargs)

        props = dict(color='w', linewidth=1, alpha=1, solid_capstyle='butt')
        props.update(line_props if line_props is not None else {})
        self.linewidth = props['linewidth']
        self.maxdist = maxdist
        self._active_pt = None
        
        # This is used to mark if this line is choosed by the user now
        self._active_line = False
        # This is used to mark if this line is deleted by the user
        self._deleted_line = False
        # Force the end point only move horizontally or vertically
        self._hori = False
        self._verti = False
        self._mode = mode

        x = (0, 0)
        y = (0, 0)
        self._end_pts = np.transpose([x, y])

        self._line = lines.Line2D(x, y, visible=False, animated=True, **props)
        self.ax.add_line(self._line)

        self._handles = ToolHandles(self.ax, x, y,
                                    marker_props=handle_props)
        self._handles.set_visible(True)
        self._line.set_visible(True)
        self.artists = [self._line, self._handles.artist]
        self.manager.add_tool(self)
    
    def remove(self):
        # Detach, not fully delete
        super().remove()
        self._line.remove()
        self._handles.remove()
              
    @property
    def end_points(self):
        return self._end_pts - 0.5
    
    @property
    def mode(self):
        return self._mode
    
    @end_points.setter
    def end_points(self, pts):
        pts = np.asarray(pts) + 0.5
        self._end_pts = pts
        self._line.set_data(np.transpose(pts))
        self._handles.set_data(np.transpose(pts))
        self._line.set_linewidth(self.linewidth)     
        # So that the line can keep moving while user changes the end points
#        self.set_visible(True)
        self.redraw()
    
    @property
    def level(self):
        if self._mode == 'Horizontal':
            return (self._end_pts[0, 1] + self._end_pts[1, 1])/2
        elif self._mode == 'Vertical':
            return (self._end_pts[0, 0] + self._end_pts[1, 0])/2
    
    @level.setter
    def level(self, l):
        pass    

    def hit_test(self, event):
        if event.button != 1 or not self.ax.in_axes(event):
            # The line is not chosen
            self._active_line = False
            return False
        idx, px_dist = self._handles.closest(event.x, event.y)
        if px_dist < self.maxdist:
            # If one of the end point is choosed, this line is chosen
            self._active_pt = idx
            self._active_line = True
            return True
        else:
            # The line is not chosen
            self._active_line = False
            self._active_pt = None
            return False
        
    def on_key_press(self, event):
        # Pass to matplotlib. This is different with Qt
#        if event.key == 'delete':
#            self.callback_on_key_press(self.delete)
        if event.key == 'shift':
            if self._mode == 'Horizontal':
                self.callback_on_key_press(self.horizontalMove)
            elif self._mode == 'Vertical':
                self.callback_on_key_press(self.verticalMove)
            
    def on_key_release(self, event):
        if event.key == 'shift':
            self.callback_on_key_release(self.freeMove)
        
    def on_mouse_press(self, event):
        if event.button != 1:
            return
        # Change cursor to cross when pressed
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        
#        self.set_visible(True)
#        if self._active_pt is not None:
#            self._active_pt = 0
#            x, y = event.xdata, event.ydata
#            self._end_pts = np.array([[x, y], [x, y]])

    def on_mouse_release(self, event):
        if event.button != 1:
            return
        # Change cursor to arrow when released
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self._active_pt = None
        self.redraw()

    def on_move(self, event):
        if event.button != 1 or self._active_pt is None:
            return
        if not self.ax.in_axes(event):
            return
        self.update(event.xdata, event.ydata)
        self.callback_on_move(self.end_points)
   
    def update(self, x=None, y=None):
        if x is not None:
        # If shift is pressed, then only move horizontally
            if self._hori and not self._verti:
                if self._active_pt == 0:
                    self._end_pts[self._active_pt, :] = x, self._end_pts[1, 1]
                else:
                    self._end_pts[self._active_pt, :] = x, self._end_pts[0, 1]
        if y is not None:
            if self._verti and not self._hori:
                if self._active_pt == 0:
                    self._end_pts[self._active_pt, :] = self._end_pts[1, 0], y
                else:
                    self._end_pts[self._active_pt, :] = self._end_pts[0, 0], y
        if x is not None and y is not None:
            if not self._verti and not self._hori:
                self._end_pts[self._active_pt, :] = x, y
        # This is to compensate the 0.5 added in the display. Here x and y are real
        # position in the canvas. This function is to handle the movement of lines
        self.end_points = self._end_pts - 0.5
    
    def is_active(self):
        return self._active_line
    
    def is_deleted(self):
        return self._deleted_line
    
    @property
    def delete(self):
        if self._active_line:
            self._deleted_line = True
            self.set_visible(False)
            self.maxdist = 0
    
    @property
    def horizontalMove(self):
        if self._active_line:    
            self._hori = True
            self._verti = False
    
    @property
    def verticalMove(self):
        if self._active_line:    
            self._hori = False
            self._verti = True

    @property
    def freeMove(self):
        if self._active_line:
            self._hori = False
            self._verti = False

class LimLineTool(CanvasToolBase):
    """Line class for limitting lines.
    
    The limiting line is controlled by one handle instead of two, since it's 
    always horizontal or vertical.
    """
    
    def __init__(self, manager, mode='Horizontal', lim=np.inf, on_move=None, on_mouse_release=None, 
                 on_key_press=None, on_key_release=None, maxdist=20, 
                 line_props=None, handle_props=None, **kwargs):
        super().__init__(manager, on_move=on_move, on_mouse_release=on_mouse_release,
                                  on_key_press=on_key_press, 
                                  on_key_release=on_key_release,
                                  **kwargs)
        
        props = dict(color='w', linewidth=1, alpha=1, solid_capstyle='butt')
        props.update(line_props if line_props is not None else {})
        self.linewidth = props['linewidth']
        self.maxdist = maxdist
        
        # This is used to mark if this line is choosed by the user now
        self._active_line = False
        # This is used to mark if this line is deleted by the user
        self._deleted_line = False
        # Force the end point only move horizontally
        self._lim = lim
        self._mode = mode

        x = (0, 0)
        y = (0, 0)
        self._level = 0
        self._end_pts = np.transpose([x, y])
        
        self._line = lines.Line2D(x, y, visible=False, animated=True, **props)
        
        self.ax.add_line(self._line)
        
        self._handles = ToolHandles(self.ax, (0,), (0,),
                                    marker_props=handle_props)
        self._handles.set_visible(True)
        self._line.set_visible(True)
        self.artists = [self._line, self._handles.artist]
        self.manager.add_tool(self)
    
    def remove(self):
        # Detach, not fully delete
        super().remove()
        self._line.remove()
        self._handles.remove()
    
    def hit_test(self, event):
        if event.button != 1 or not self.ax.in_axes(event):
            # The line is not chosen
            self._active_line = False
            return False
        idx, px_dist = self._handles.closest(event.x, event.y)
        if px_dist < self.maxdist:
            self._active_line = True
            return True
        else:
            # The line is not chosen
            self._active_line = False
            return False
    
    def on_move(self, event):
        if event.button != 1 or not self._active_line:
            return
        if not self.ax.in_axes(event):
            return
        self.update(event.xdata, event.ydata)
        self.callback_on_move(self.level)
    
    def update(self, x=None, y=None):
        if x is not None:
        # If shift is pressed, then only move horizontally
            if self._mode == 'Vertical':
                if x < self._lim:
                    self._end_pts[:,0] = [x, x]
                else:
                    self._end_pts[:,0] = [self._lim-1, self._lim-1]
        if y is not None:
            if self._mode == 'Horizontal':
                if y < self._lim:
                    self._end_pts[:,1] = [y, y]
                else:
                    self._end_pts[:,1] = [self._lim-1, self._lim-1]
        # To compensate the 0.5 added in the display. Here x and y are real
        # position in the canvas. This function is to handle the movement of lines
        self.end_points = self._end_pts - 0.5
    
    @property
    def mode(self):
        return self._mode
    
    @property
    def end_points(self):
        return self._end_pts - 0.5

    @end_points.setter
    def end_points(self, pts):
        pts = np.asarray(pts) + 0.5
        p1, p2 = pts
        if self._mode == 'Horizontal':
            if p1[1] != p2[1]:
                y = (p1[1] + p2[1])/2
                p1[1] = y
                p2[1] = y
            else:
                y = p1[1]
            self._level = y
            x = (p1[0] + p2[0])/2
        elif self._mode == 'Vertical':
            if p1[0] != p2[0]:
                x = (p1[0] + p2[0])/2
                p1[0] = x
                p2[0] = x
            else:
                x = p1[0]
            self._level = x
            y = (p1[1] + p2[1])/2
        self._end_pts = pts
        self._line.set_data(np.transpose(pts))
        self._handles.set_data([[x], [y]])
        self._line.set_linewidth(self.linewidth)     
        self.redraw()
    
    @property
    def level(self):
        return self._level
    
    @level.setter
    def level(self, lvl):
        p1, p2 = self._end_pts
        if self._mode == 'Horizontal':
            y = lvl
            x = int((p1[0] + p2[0])/2)
            p1[1] = y
            p2[1] = y
        elif self._mode == 'Vertical':
            x = lvl
            y = int((p1[1] + p2[1])/2)
            p1[0] = x
            p2[0] = x
        self.end_points = [p1, p2]        
    
    @property
    def limit(self):
        return self._lim
    
    @limit.setter
    def limit(self, lim):
        self._lim = lim
        if self._level >= self._lim:
            self.level = lim - 1        
    
    def is_active(self):
        return self._active_line
    
    def is_deleted(self):
        return self._deleted_line
    
    
class ThickLineTool(LineTool):
    """Widget for line selection in a plot (Not been used currently)

    The thickness of the line can be varied using the mouse scroll wheel, or
    with the '+' and '-' keys.

    Parameters
    ----------
    manager : Viewer or PlotPlugin.
        Skimage viewer or plot plugin object.
    on_move : function
        Function called whenever a control handle is moved.
        This function must accept the end points of line as the only argument.
    on_release : function
        Function called whenever the control handle is released.
    on_enter : function
        Function called whenever the "enter" key is pressed.
    on_change : function
        Function called whenever the line thickness is changed.
    maxdist : float
        Maximum pixel distance allowed when selecting control handle.
    line_props : dict
        Properties for :class:`matplotlib.lines.Line2D`.
    handle_props : dict
        Marker properties for the handles (also see
        :class:`matplotlib.lines.Line2D`).

    Attributes
    ----------
    end_points : 2D array
        End points of line ((x1, y1), (x2, y2)).
    """

    def __init__(self, manager, on_move=None, on_enter=None, on_release=None,
                 on_change=None, maxdist=10, line_props=None, handle_props=None):
        super(ThickLineTool, self).__init__(manager,
                                            on_move=on_move,
                                            on_enter=on_enter,
                                            on_release=on_release,
                                            maxdist=maxdist,
                                            line_props=line_props,
                                            handle_props=handle_props)

        if on_change is None:
            def on_change(*args):
                pass
        self.callback_on_change = on_change

    def on_scroll(self, event):
        if not event.inaxes:
            return
        if event.button == 'up':
            self._thicken_scan_line()
        elif event.button == 'down':
            self._shrink_scan_line()

    def on_key_press(self, event):
        if event.key == '+':
            self._thicken_scan_line()
        elif event.key == '-':
            self._shrink_scan_line()

    def _thicken_scan_line(self):
        self.linewidth += 1
        self.update()
        self.callback_on_change(self.geometry)

    def _shrink_scan_line(self):
        if self.linewidth > 1:
            self.linewidth -= 1
            self.update()
            self.callback_on_change(self.geometry)