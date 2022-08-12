"""Common response format"""
from libs.types import JsonSerializable


def to_error_response(msg: str, details: JsonSerializable = None):
    """Format to error response"""
    return {'message': msg, 'details': details}
