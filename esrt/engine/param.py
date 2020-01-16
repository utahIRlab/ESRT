import typing
import inspect

class Param(object):
    """
    - Basic usage with name, value, desc:
    >>> param = Param('score', 100, desc='The final score of Data Structure.')
    >>> param.name
    'score'
    >>> param.value
    '100'
    >>> param.desc
    'The final score of Data Structure.'

    - Use 'verifier' to check whether the param's value is vaild:
    >>> param = Param('score', 101, verifier = lambda x: 0 <= x <= 100)
    Traceback (most recent call last):
            ...
        ValueError: Verifier not satifised.
        The verifier's definition is as follows:
        verifier=lambda x: 0 <= x <= 100

    - The param instance return True only if its value is not None:
    >>> param = Param('score')
    >>> if param:
    ...     print("OK")
    >>> param.value = 100
    >>> if param:
    ...     print("OK")
    OK
    """
    def __init__(
        self,
        name: str,
        value: typing.Any = None,
        desc: str = None,
        verifier: typing.Callable[[typing.Any], bool] = None
    ):
        self._name = name
        self._value = value
        self._desc = desc
        self._verifier = verifier

        if self._verifier:
            self.veirfiy(self._value)


    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> typing.Any:
        return self._value

    @value.setter
    def value(self, value: typing.Any):
        self.verify(value)
        self._value = value

    @property
    def desc(self) -> str:
        return self._desc

    @desc.setter
    def desc(self, desc: str):
        self._desc = desc

    @property
    def verifier(self) -> typing.Callable[[typing.Any], bool]:
        return self._verifier

    @verifier.setter
    def verifier(self, new_verifier: typing.Callable[[typing.Any], bool]):
        self._verifier = new_verifier

    def verify(self, value: typing.Any):
        if self._verifier:
            valid = self._verifier(value)
            if not valid:
                msg = 'The value is not compatible with the verifier\n'
                msg += 'The definition of the verifier is: \n'
                msg += inspect.getsource(self._verifier).strip()
                raise ValueError(msg)

    def __bool__(self):
        return self._value is not None
