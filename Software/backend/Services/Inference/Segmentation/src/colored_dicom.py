import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SegmentationStorage
import os

def get_dicom_tag_value(metadata, tag, default=""):
    """ Extract the value from metadata for the given DICOM tag. """
    item = metadata.get(tag, {})
    if 'Value' in item and isinstance(item['Value'], list) and len(item['Value']) > 0:
        return str(item['Value'][0])
    return default

def convert_array_to_dicom_seg(image_array, output_folder, metadata=None):
    # Set defaults if metadata is None
    if metadata is None:
        metadata = {}

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SegmentationStorage  # Use SegmentationStorage UID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    # Extract DICOM tags from metadata using the corrected function
    ds.PatientID = get_dicom_tag_value(metadata, '00100020', "DefaultPatientID")
    ds.PatientName = get_dicom_tag_value(metadata, '00100010', 'Anonymous')
    ds.StudyInstanceUID = get_dicom_tag_value(metadata, '0020000D', generate_uid())
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = get_dicom_tag_value(metadata, '00080060', 'MR')
    ds.Manufacturer = get_dicom_tag_value(metadata, '00080070', 'Unknown manufacturer')
    
    ds.SliceThickness = "1"
    ds.PatientPosition = get_dicom_tag_value(metadata, '00185100', 'HFS')
    ds.StudyID = "SLICER10001"
    ds.SeriesNumber = "301"
    ds.SamplesPerPixel = 1 if len(image_array.shape) == 3 else image_array.shape[-1]
    ds.PhotometricInterpretation = "MONOCHROME2" if ds.SamplesPerPixel == 1 else "RGB"
    ds.PixelSpacing = [1, 1]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 0

    # Set Rows and Columns based on image shape
    ds.Rows, ds.Columns = image_array.shape[1:3]

    # Iterate over slices and create DICOM SEG
    for i, slice_2d in enumerate(image_array):
        ds.InstanceNumber = i + 1
        ds.SOPInstanceUID = generate_uid()
        ds.PixelData = slice_2d.astype(np.uint8).tobytes()

        # Spatial positioning and orientation
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # Updated to a more reasonable value
        ds.ImagePositionPatient = [0, 0, i * float(ds.SliceThickness)]
        ds.SliceLocation = i * float(ds.SliceThickness)

        filename = os.path.join(output_folder, f"slice_{i + 1:03d}.dcm")
        pydicom.dcmwrite(filename, ds, write_like_original=False)
        print(f"Saved {filename}")

