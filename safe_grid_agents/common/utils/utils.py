"""Generic utils file."""


class ConfigWrapper(dict):
    """Wraps a dictionary to allow for using __getattr__ in place of __getitem__"""

    def __init__(self, dictionary):
        super(ConfigWrapper, self).__init__()
        for k, v in dictionary.items():
            self[k] = v

    def __getattribute__(self, attr):
        try:
            return self[attr]
        except TypeError, KeyError:
            return super(ConfigWrapper, self).__getattribute__(attr)
