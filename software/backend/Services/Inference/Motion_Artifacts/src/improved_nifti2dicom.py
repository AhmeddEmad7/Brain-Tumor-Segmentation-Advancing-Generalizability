import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
import numpy as np
import os

def get_dicom_tag_value(metadata, tag, default=""):
    """ Extract the value from metadata for the given DICOM tag. """
    item = metadata.get(tag, {})
    if 'Value' in item and isinstance(item['Value'], list) and len(item['Value']) > 0:
        return str(item['Value'][0])
    return default
def extract_image_orientation_patient(nifti_file):
    nifti_img = nib.load(nifti_file)
    affine = nifti_img.affine

    row_vector = affine[:3, 0]  # X-axis direction
    col_vector = affine[:3, 1]  # Y-axis direction

    image_orientation_patient = list(row_vector / np.linalg.norm(row_vector)) + list(col_vector / np.linalg.norm(col_vector))
    return image_orientation_patient, affine

def convert_nifti_to_dicom(nifti_file, output_folder, metadata=None):
    if metadata is None:
        metadata = {} 
    nifti_img = nib.load(nifti_file)
    nifti_data = nifti_img.get_fdata()

    image_orientation_patient, affine = extract_image_orientation_patient(nifti_file)
    origin = affine[:3, 3]
    
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    ds.PatientID = get_dicom_tag_value(metadata, '00100020', "DefaultPatientID")
    ds.PatientName = get_dicom_tag_value(metadata, '00100010', 'Anonymous')
    ds.StudyInstanceUID = get_dicom_tag_value(metadata, '0020000D') or generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = get_dicom_tag_value(metadata, '00080060') or 'MR'
    ds.Manufacturer = get_dicom_tag_value(metadata, '00080070') or 'Unknown manufacturer'
    # ds.ImagePositionPatient = get_dicom_tag_value(metadata, '00200032', [0.0, 0.0, 0.0])
    # print("ds.ImagePositionPatient",ds.ImagePositionPatient)
    
    # # ds.ImageOrientationPatient = get_dicom_tag_value(metadata, '00200037', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    # ds.SliceLocation = get_dicom_tag_value(metadata, '00201040', "0")
    # print("ds.SliceLocation",ds.SliceLocation)

    ds.SliceThickness = "1"
    ds.PatientPosition = get_dicom_tag_value(metadata, '00185100') or 'HFS'
    ds.StudyID = "SLICER10001"
    ds.SeriesNumber = "301"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [1, 1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.WindowCenter = "948"
    ds.WindowWidth = "1896"
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.RescaleType = "HU"

    nifti_data = np.swapaxes(nifti_data, 0, 2)
    nifti_data = np.flip(nifti_data, axis=(1, 2))  # Flip along X (axis 1) and Y (axis 2)

    ds.Rows, ds.Columns = nifti_data.shape [1:3]

    for i, slice_2d in enumerate(nifti_data):
        
        ds.InstanceNumber = i + 1
        ds.SOPInstanceUID = generate_uid()
        ds.PixelData = slice_2d.astype(np.int16).tobytes()
        
        # # Spatial positioning and orientation
        ds.ImagePositionPatient = list(origin + i * affine[:3, 2])
        ds.ImageOrientationPatient = image_orientation_patient
        ds.SliceLocation = i * float(ds.SliceThickness)

        filename = os.path.join(output_folder, f"slice_{i + 1:03d}.dcm")
        pydicom.filewriter.dcmwrite(filename, ds, write_like_original=False)
        print(f"Saved {filename}")
        

nifti_file = r"C:\Users\hazem\Downloads\test2\project\segmentation\sequences\BraTS-GLI-00182-000-t1c.nii"
output_folder = r"C:\Users\hazem\Downloads\hz11"
convert_nifti_to_dicom(nifti_file,output_folder)
# Example usage
# json_metadata_file = "C:\\Users\\Mina A Tayeh\\Desktop\\Dicom-nifti conv\\BraTS2021_00000_flair.nii\\mina.json"
# os.makedirs(output_folder, exist_ok=True)
# convert_nifti_to_dicom(nifti_file, output_folder, json_metadata_file)