import typing
import collections


import pandas as pd
import yaml

from esrt.engine.param import Param

class ParamTable(object):
    """
    Basic Usage:
    >>> params = ParamsTable()
    >>> params.add(Param('Wang Gang', 'Chief'))
    >>> params.add('Jay Chou', 'Singer')
    >>> params['Wang Gang']
    Chief
    >>> print(params)

    Not allow .add() method to reasign a existing Param:
    >>> params.add['Wang Gang'] = "Youtuber"
    Traceback (most recent call last):
        ...
    ValueError: ...
    """
    def __init__(self):
        self._params = {}

    def add(self, param: Param):
        if param.name in self._params:
            msg = "Cannot reasign existing value.\n"
            msg += f"param's name: \"{param.name}\"\n"
            msg += f"param's value: {param.value}\n"
            msg += f"the existing value: {self._params[param.name].value}"
            raise ValueError(msg)
        self._params[param.name] = param

    def get(self, key) -> Param:
        return self._params[key]

    def set(self, key, param: Param):
        self._params[key] = param

    def to_frame(self) -> pd.DataFrame:
        """
        Convert the param_table to the pd.DataFrame to improve readability.

        >>> params = ParamsTable()
        >>> params.add(Param('Wang Gang', 'Chief'))
        >>> params.add('Jay Chou', 'Singer')
        >>> params.to_frame()

            Name    Value   Description
        0   Wang Gang Chief
        1   Jay Chou   Singer
        """
        df = pd.DataFrame(data={
                            'Name': [p.name for p in self],
                            'Value': [p.value for p in self],
                            'Description': [p.desc for p in self]
                                })
        return df

    def __getitem__(self, key: str) -> typing.Any:
        return self._params[key].value

    def __setitem__(self, key: str, value: typing.Any):
        self._params[key].value = value

    def __str__(self):
        return '\n'.join([p.name.ljust(50) + str(p.value) for p in self])

    def __iter__(self):
        yield from self._params.values()

    def __contains__(self, key: str):
        return key in self._params

    def keys(self) -> collections.abc.KeysView:
        return self._params.keys()

    def completed(self) -> bool:
        return any([p for p in self])

    def update(self, other: dict):
        pass

    def update_from_yaml(self, fn):
        with open(fn, 'r') as f:
            tdict = yaml.load(f, Loader=yaml.SafeLoader)
            if 'hparams' in tdict:
                hdict = tdict['hparams']
                for key, value in hdict.items():
                    self.set(key, Param(key, value))

    def update_from_json(self, fn):
        pass
