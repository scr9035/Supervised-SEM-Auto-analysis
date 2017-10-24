# -*- coding: utf-8 -*-

from SSA.gui.viewers import XSEMViewer
import matplotlib.pyplot as plt
# Turn off the interactive mode of matplotlib so figure window won't show

if __name__ == '__main__':
    try:
        plt.ioff()
        viewer = XSEMViewer()
        viewer.show()
    except SystemExit as e:
        print('Error!', str(e))
        input('Press ENTER to exit')