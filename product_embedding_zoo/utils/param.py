import yaml

class Param():
    """
    Read data from .yaml file, access and assign the data like:
    learning_rate = params.hparams.learning_rate
    params.hparams.learning_rate = 0.1
    """
    def __init__(self,yaml_path):
        self.update(yaml_path)

    def write(self, yaml_path):
        with open(yaml_path, 'r') as f:
            yaml.dump(self.__dict__, f)

    def update(self, yaml_path):
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
            self.__dict__.update(params)
            for k, v in self.__dict__.items():
                print(k, v, type(v))
                if isinstance(v, dict):
                    self.__dict__[k] = _Objective(v)

class _Objective():
    def __init__(self, d):
        assert isinstance(d, dict)
        self.__dict__ = d


if __name__ == "__main__":
    yaml_path = "./fake_exp_settings.yaml"
    params = Params(yaml_path)
    print(params.__dict__)
    print(params.hparams.learning_rate)
    params.hparams.learning_rate = 0.1
    print(params.hparams.learning_rate)
    for k, v in params.__dict__.items():
        if isinstance(v, _Objective):
            for k2, v2 in v.__dict__.items():
                print(k, k2, v2)
