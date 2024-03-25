from tifffile import TiffFile


def parse_si_metadata(tiff_path):
    """
    Reads metadata from a Scanimage TIFF and return a dictionary with
    specified key values.

    Currently can only extract numerical data.

    Args:
        tiff_path: path to TIFF or directory containing tiffs

    Returns:
        dict: dictionary of SI parameters

    """
    if tiff_path.suffix != ".tif":
        tiffs = [tiff_path / tiff for tiff in sorted(tiff_path.glob("*.tif"))]
    else:
        tiffs = [
            tiff_path,
        ]
    if tiffs:
        return TiffFile(tiffs[0]).scanimage_metadata["FrameData"]
    else:
        return None
