"""Shared design settings used by the viewport, projects and AI proposals."""
from soundform.models.project import DesignParameters, default_parameters


class DesignState:
    def __init__(self):
        self._parameters = default_parameters()

    def parameters(self):
        return DesignParameters.from_dict(self._parameters.to_dict())

    def set_parameters(self, parameters):
        self._parameters = DesignParameters.from_dict(parameters.to_dict())
