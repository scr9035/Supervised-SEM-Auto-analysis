import os

from PyQt5.QtWidgets import QFileDialog


__all__ = ['open_files_dialog', 'save_file_dialog']


def _format_filename(filename):
    if isinstance(filename, tuple):
        # Handle discrepancy between PyQt4 and PySide APIs.
        filename = filename[0]
    if len(filename) == 0:
        return None
    return filename


def open_files_dialog():
    """Return user-selected file path."""
    filename = QFileDialog.getOpenFileNames()
    filename = _format_filename(filename)
    return filename


def save_file_dialog(default_format='png'):
    """Return user-selected file path."""
    filename = QFileDialog.getSaveFileName()
    filename = _format_filename(filename)
    if filename is None:
        return None
    #TODO: io plugins should assign default image formats
    basename, ext = os.path.splitext(filename)
    if not ext:
        filename = '%s.%s' % (filename, default_format)
    return filename
