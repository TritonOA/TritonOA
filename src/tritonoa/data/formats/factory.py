from typing import Optional

from tritonoa.data.formats.shru import SHRUReader, SHRURecordFormatter, SHRUFileFormat
from tritonoa.data.formats.sio import SIOReader, SIOFileFormat
from tritonoa.data.formats.whoi3dvha import WHOI3DVHAReader, WHOI3DVHAFileFormat


def get_file_format(
    suffix: Optional[str] = None, file_format: Optional[str] = None
) -> str:
    if suffix is None and file_format is None:
        raise ValueError("An argument 'suffix' or 'file_format' must be provided.")
    if file_format is not None:
        file_format = validate_file_format(file_format)
    if file_format is None:
        file_format = validate_file_format(suffix)
    return file_format


def get_formatter(
    suffix: Optional[str] = None, file_format: Optional[str] = None
) -> object:
    file_format = get_file_format(suffix, file_format)
    if file_format == SHRUFileFormat.FORMAT.value:
        return SHRURecordFormatter()
    raise ValueError(f"File format {file_format} is not recognized.")


def get_reader(
    suffix: Optional[str] = None, file_format: Optional[str] = None
) -> object:
    """Factory to get header reader for file format.

    Args:
        suffix (Optional[str], optional): File suffix. Defaults to None.
        file_format (Optional[str], optional): File format. Defaults to None.

    Returns:
        tuple[Callable, FileFormat]: Header reader and file format.

    Raises:
        ValueError: If file format is not recognized.
    """
    file_format = get_file_format(suffix, file_format)
    if file_format == SHRUFileFormat.FORMAT.value:
        return SHRUReader()
    if file_format == SIOFileFormat.FORMAT.value:
        return SIOReader()
    if file_format == WHOI3DVHAFileFormat.FORMAT.value:
        return WHOI3DVHAReader()
    raise ValueError(f"File format {file_format} is not recognized.")


def validate_file_format(desc: str) -> str:
    """Get file format from suffix."""
    if SHRUFileFormat.is_format(desc):
        return SHRUFileFormat.FORMAT.value
    if SIOFileFormat.is_format(desc):
        return SIOFileFormat.FORMAT.value
    if WHOI3DVHAFileFormat.is_format(desc):
        return WHOI3DVHAFileFormat.FORMAT.value
    raise ValueError(f"File format cannot be inferred from file extension '{desc}'.")
