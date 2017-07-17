# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li


from .base import CanvasToolBase, ToolHandles
import numpy as np
import matplotlib as mpl

class BoundaryTool(CanvasToolBase):
    
    def __init__(self, manager, on_move=None, on_mouse_release=None, 
                 on_key_press=None, on_key_release=None, maxdist=20, 
                 bound_props=None, handle_props=None, **kwargs):
        super().__init__(manager, on_move=on_move, on_mouse_release=on_mouse_release,
                                  on_key_press=on_key_press, 
                                  on_key_release=on_key_release,
                                  **kwargs)
        self._active = False
        self._deleted = False
        props = dict(marker='+', markersize=1, color='r', mfc='w', ls='none',
                     alpha=1, visible=True)
        props.update(bound_props if bound_props is not None else {})
        self.maxdist = maxdist
        
        bx = (0,)
        by = (0,)
        self._pts = np.squeeze(np.transpose([bx, by])) 
        self._boundary = mpl.lines.Line2D(bx, by, **props)
        self.ax.add_line(self._boundary)
        
        cx = (0,)
        cy = (0,)
        h_props = dict(marker='+', markersize=10, color='r', mfc='w', ls='none',
                     alpha=1, visible=True)
        h_props.update(handle_props if handle_props is not None else {})
        self._center = np.squeeze(np.transpose([cx, cy]))
        self._handles = ToolHandles(self.ax, cx, cy,
                                    marker_props=h_props)
        self._handles.set_visible(True)
        self.artists = [self._boundary, self._handles.artist]
        self.manager.add_tool(self)
        
    def remove(self):
        super().remove()
        self._boundary.remove()
        self._handles.remove()
        
    @property
    def center(self):
        return self._center.astype(int)
    @center.setter
    def center(self, pt):
        self._center = np.asarray(pt)        
        self._handles.set_data(np.transpose(pt))
        self.redraw()
        
    @property
    def boundary(self):
        return self._boundary   
    @boundary.setter
    def boundary(self, bound):
        self._pts = np.transpose(bound)
        self._boundary.set_data(bound)
#        self._pts = np.asarray(bound)
#        self._boundary.set_data(np.transpose(bound)) 
        self.redraw()
    
    @property
    def delete(self):
        if self._active:
            self._deleted = True
            self.set_visible(False)
            self.maxdist = 0
    
    def hit_test(self, event):
        if event.button != 1 or not self.ax.in_axes(event):
            # DYL: the line is not choosed
            self._active = False
            return False
        cx, cy = self._center
        dist = np.sqrt((event.xdata - cx)**2 + (event.ydata - cy)**2)
        if dist < self.maxdist:
            # DYL: if one of the end point is choosed, this line is choosed
            self._active = True
            return True
        else:
            self._active = False
            return False
        
    def is_active(self):
        return self._active