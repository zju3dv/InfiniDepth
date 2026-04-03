from training.utils.logger import Log
def monitor_process_wrapper(func):
    """The wrapper will print a log both before and after the wrapped function runned."""
    def wrapped(*args, **kwargs):
        Log.info(f'"{func.__name__}()" begin...')
        ret_value = func(*args, **kwargs)
        Log.info(f'"{func.__name__}()" end...')
        return ret_value
    return wrapped

def monitor_class_process_wrapper(func):
    """The wrapper will print a log both before and after the wrapped function runned."""
    def wrapped(self, *args, **kwargs):
        Log.info(f'"{self.__class__.__name__}.{func.__name__}()" begin...')
        ret_value = func(self, *args, **kwargs)
        Log.info(f'"{self.__class__.__name__}.{func.__name__}()" end...')
        return ret_value
    return wrapped
