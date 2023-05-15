import time


def new_info(message):
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    return 'INFO: ' + message + ' [' + now + ']'


def new_warn(message):
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    return 'WARN: ' + message + ' [' + now + ']'


def new_error(message):
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    return 'ERROR: ' + message + '[' + now + ']'
