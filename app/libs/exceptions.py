"""Exceptions"""
from libs.types import JsonSerializable


class QMLError(Exception):
    """Quick ML base exception"""
    details: JsonSerializable = None

    def __init__(self, message, details: JsonSerializable = None):
        super().__init__(message)
        self.details = details


class BusyError(QMLError):
    """Job is busy"""

    def __init__(self, details: JsonSerializable = None):
        super().__init__('Too budy right now', details)


class UnexpectedFileTypeError(QMLError):
    """Unexpected file type"""

    def __init__(self, details: JsonSerializable = None):
        super().__init__('Unexpected file type found', details)
