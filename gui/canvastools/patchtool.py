# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from .base import CanvasToolBase, ToolHandles
import numpy as np
import matplotlib as mpl

class PatchTool(CanvasToolBase):
    
    def __init__(self, manager, on_move=None, on_mouse_release=None, 
                 on_key_press=None, on_key_release=None, maxdist=20, 
                 patch_props=None, handle_props=None, **kwargs):
        super().__init__(manager, on_move=on_move, on_mouse_release=on_mouse_release,
                                  on_key_press=on_key_press, 
                                  on_key_release=on_key_release,
                                  **kwargs)
                                                   
        props = dict(edgecolor='red', linewidth=1, fill=False,linestyle='-')
        props.update(patch_props if patch_props is not None else {})
        self.linewidth = props['linewidth']
        self.maxdist = maxdist
        
        # DYL: This is used to mark if this line is choosed by the user now
        self._activated = False
        # DYL: This is used to mark if this line is deleted by the user
        self._deleted = False
        
        x = (0,)
        y = (0,)
        self._major = 1
        self._minor = 1
        self._angle = 0
        self._center = np.squeeze(np.transpose([x, y]))
        self._patch = mpl.patches.Ellipse(self._center, width=self._major, 
                                                   height=self._minor, 
                                                   angle=self._angle,
                                                   **props)
        self.ax.add_patch(self._patch)
        self._handles = ToolHandles(self.ax, x, y,
                                    marker_props=handle_props)
        self._handles.set_visible(True)
        self.artists = [self._patch, self._handles.artist]
        self.manager.add_tool(self)
    
    def remove(self):
        super().remove()
        self._patch.remove()
        self._handles.remove()
          
    @property
    def center(self):
        return self._center.astype(int)
    @center.setter
    def center(self, pt):
        self._center = np.asarray(pt)        
        self._patch.center = self._center
        self._handles.set_data(np.transpose(pt))
        self.redraw()
        
    @property
    def major(self):
        return self._major   
    @major.setter
    def major(self, w):
        self._major = w
        self._patch.width = w
        self.redraw()
    
    @property
    def minor(self):
        return self._minor
    @minor.setter
    def minor(self, h):
        self._minor = h
        self._patch.height = h
        self.redraw()
    
    @property
    def angle(self):
        return self._angle
    @angle.setter
    def angle(self, a):
        self._angle = a
        self._patch.angle = a
        self.redraw()
        
    def hit_test(self, event):
        if event.button != 1 or not self.ax.in_axes(event):
            # DYL: the line is not choosed
            self._activated = False
            return False
        idx, px_dist = self._handles.closest(event.x, event.y)
        if px_dist < self.maxdist:
            # DYL: if one of the end point is choosed, this line is choosed
            self._activated = True
            return True
        else:
            # DYL: the line is not choosed
            self._activated = False
            return False
#        
#    def on_key_press(self, event):
#        pass
#            
#    def on_key_release(self, event):
#        pass
#        
#    def on_mouse_press(self, event):
#        pass
#
#    def on_mouse_release(self, event):
#        pass
#
#    def on_move(self, event):
#        pass
#   
#    def update(self, x=None, y=None):
#        pass
#    
#    def is_active(self):
#        return self._activated
#    
#    def is_deleted(self):
#        return self._deleted
#    
#    def delete_line(self):
#        if self._activated:
#            self._deleted = True
#            self.set_visible(False)
#    
#    @property
#    def geometry(self):
#        return
#   
#    @property
#    def delete(self):
#        if self._activated:
#            self._activated = True
#            self.set_visible(False)
#            self.maxdist = 0