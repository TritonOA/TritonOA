from tritonoa.data.formats.shru import SHRUFileFormat, SHRUReader, SHRURecordFormatter
from tritonoa.data.formats.sio import SIOFileFormat, SIOReader, SIORecordFormatter
from tritonoa.data.formats.wav import WAVFileFormat, WAVReader, WAVRecordFormatter
from tritonoa.data.formats.whoi3dvha import (
    WHOI3DVHAFileFormat,
    WHOI3DVHAReader,
    WHOI3DVHARecordFormatter,
)


def get_file_format(suffix: str | None = None, file_format: str | None = None) -> str:
    """Get and validate file format from suffix or format string.

    Args:
        suffix: File suffix (e.g., '.wav', '.shru').
        file_format: File format identifier string.

    Returns:
        Validated file format string.

    Raises:
        ValueError: If neither suffix nor file_format is provided, or if
            the format cannot be recognized.
    """
    if suffix is None and file_format is None:
        raise ValueError("An argument 'suffix' or 'file_format' must be provided.")
    if file_format is not None:
        file_format = validate_file_format(file_format)
    if file_format is None:
        file_format = validate_file_format(suffix)
    return file_format


def get_formatter(suffix: str | None = None, file_format: str | None = None) -> object:
    """Factory to get record formatter for file format.

    Args:
        suffix: File suffix (e.g., '.wav', '.shru').
        file_format: File format identifier string.

    Returns:
        Record formatter instance for the specified file format.

    Raises:
        ValueError: If the file format is not recognized or if neither
            suffix nor file_format is provided.
    """
    file_format = get_file_format(suffix, file_format)
    if file_format == SHRUFileFormat.FORMAT.value:
        return SHRURecordFormatter()
    if file_format == SIOFileFormat.FORMAT.value:
        return SIORecordFormatter()
    if file_format == WAVFileFormat.FORMAT.value:
        return WAVRecordFormatter()
    if file_format == WHOI3DVHAFileFormat.FORMAT.value:
        return WHOI3DVHARecordFormatter()
    raise ValueError(f"File format {file_format} is not recognized.")


def get_reader(suffix: str | None = None, file_format: str | None = None) -> object:
    """Factory to get file reader for file format.

    Args:
        suffix: File suffix (e.g., '.wav', '.shru').
        file_format: File format identifier string.

    Returns:
        File reader instance for the specified file format.

    Raises:
        ValueError: If the file format is not recognized or if neither
            suffix nor file_format is provided.
    """
    file_format = get_file_format(suffix, file_format)
    if file_format == SHRUFileFormat.FORMAT.value:
        return SHRUReader()
    if file_format == SIOFileFormat.FORMAT.value:
        return SIOReader()
    if file_format == WAVFileFormat.FORMAT.value:
        return WAVReader()
    if file_format == WHOI3DVHAFileFormat.FORMAT.value:
        return WHOI3DVHAReader()
    raise ValueError(f"File format {file_format} is not recognized.")


def validate_file_format(desc: str) -> str:
    """Validate and normalize file format from suffix or format string.

    Args:
        desc: File suffix or format identifier string to validate.

    Returns:
        Normalized file format string.

    Raises:
        ValueError: If the file format cannot be inferred from the
            provided descriptor.
    """
    if SHRUFileFormat.is_format(desc):
        return SHRUFileFormat.FORMAT.value
    if SIOFileFormat.is_format(desc):
        return SIOFileFormat.FORMAT.value
    if WAVFileFormat.is_format(desc):
        return WAVFileFormat.FORMAT.value
    if WHOI3DVHAFileFormat.is_format(desc):
        return WHOI3DVHAFileFormat.FORMAT.value
    raise ValueError(f"File format cannot be inferred from file extension '{desc}'.")
