# -*- coding: utf-8 -*-

from SSA.gui.viewers import XSEMViewer
import matplotlib.pyplot as plt
# Turn off the interactive mode of matplotlib so figure window won't show

if __name__ == '__main__':
    plt.ioff()
    viewer = XSEMViewer()
    viewer.show()
