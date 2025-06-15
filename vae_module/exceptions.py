class VAEError(Exception):
    """Base class for VAE related errors."""


class InvalidSequenceError(VAEError):
    """Raised when a sequence contains invalid characters."""


class SequenceLengthError(VAEError):
    """Raised when a sequence is longer than the maximum allowed."""


class DeviceNotAvailableError(VAEError):
    """Raised when the requested device is not available."""
