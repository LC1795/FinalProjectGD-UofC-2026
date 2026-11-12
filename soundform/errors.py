"""Shared exceptions. UI code translates these into messages; it does not catch bare Exception silently."""


class SoundformError(Exception):
    """Base class for errors the application can explain to a user."""


class AudioLoadError(SoundformError):
    """The file could not be read or failed format checks."""


class AnalysisError(SoundformError):
    """Spatial analysis could not be completed for the current settings."""


class GeometryError(SoundformError):
    """Parameters cannot produce a coherent architectural mesh."""


class ExportError(SoundformError):
    """A mesh or project file could not be written or failed validation."""
