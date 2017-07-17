# -*- coding: utf-8 -*-
#
# Copyright © 2017 Dongyao Li

from SSA.gui.viewers import XSEMViewer
import matplotlib.pyplot as plt
# DYL: turn off the interactive mode of matplotlib so figure window won't show

if __name__ == '__main__':
    plt.ioff()
    viewer = XSEMViewer()
    viewer.show()
