import os


class _EnvironmentVariable:
    """
    Represents an environment variable.
    """

    def __init__(self, name, type_, default):
        self.name = name
        self.type = type_
        self.default = default

    @property
    def defined(self):
        return self.name in os.environ

    def get_raw(self):
        return os.getenv(self.name)

    def set(self, value):
        os.environ[self.name] = str(value)

    def unset(self):
        os.environ.pop(self.name, None)

    def get(self, not_exists_okay=True):
        """
        Reads the value of the environment variable if it exists and converts it to the desired
        type. Otherwise, returns the default value.
        """
        if (val := self.get_raw()) is not None:
            try:
                return self.type(val)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert {val!r} to {self.type} for {self.name}: {e}"
                )
        if not_exists_okay:
            return self.default
        else:
            raise EnvironmentError(
                f"{self.name} is not set and no default value is provided."
            )

    def __str__(self):
        return f"{self.name} (default: {self.default}, type: {self.type.__name__})"

    def __repr__(self):
        return repr(self.name)

    def __format__(self, format_spec: str) -> str:
        return self.name.__format__(format_spec)


class _BooleanEnvironmentVariable(_EnvironmentVariable):
    """
    Represents a boolean environment variable.
    """

    def __init__(self, name, default):
        # `default not in [True, False, None]` doesn't work because `1 in [True]`
        # (or `0 in [False]`) returns True.
        if not (default is True or default is False or default is None):
            raise ValueError(f"{name} default value must be one of [True, False, None]")
        super().__init__(name, bool, default)

    def get(self, not_exists_okay=True):
        if not self.defined:
            if not_exists_okay:
                return self.default
            else:
                raise EnvironmentError(
                    f"{self.name} is not set and no default value is provided."
                )

        val = os.getenv(self.name)
        lowercased = val.lower()
        if lowercased not in ["true", "false", "1", "0"]:
            raise ValueError(
                f"{self.name} value must be one of ['true', 'false', '1', '0'] (case-insensitive), "
                f"but got {val}"
            )
        return lowercased in ["true", "1"]


S3_BUCKET = _EnvironmentVariable("S3_BUCKET", str, None)
S3_ACCESS_KEY = _EnvironmentVariable("S3_ACCESS_KEY", str, None)
S3_SECRET_KEY = _EnvironmentVariable("S3_SECRET_KEY", str, None)
PARAMS_FILE = _EnvironmentVariable("PARAMS_FILE", str, "params.yaml")
MLFLOW_TRACKING_URI = _EnvironmentVariable("MLFLOW_TRACKING_URI", str, None)
