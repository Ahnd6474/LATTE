class VAEError(Exception):
    """Base class for VAE related errors."""


class InvalidSequenceError(VAEError):
    """Raised when a sequence contains invalid characters."""


class SequenceLengthError(VAEError):
    """Raised when a sequence is longer than the maximum allowed."""


class DeviceNotAvailableError(VAEError):
    """Raised when the requested device is not available."""


class CheckpointLoadError(VAEError):
    """Raised when a checkpoint file cannot be loaded."""

    def __init__(self, path: str, original: Exception):
        msg = f"Failed to load checkpoint '{path}': {original}"
        super().__init__(msg)
        self.path = path
        self.original = original
