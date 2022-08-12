"""Exceptions"""
from libs.types import JsonSerializable


class QMLError(Exception):
    """Quick ML base exception"""
    details: JsonSerializable = None

    def __init__(self, message, details: JsonSerializable = None):
        super().__init__(message)
        self.details = details
