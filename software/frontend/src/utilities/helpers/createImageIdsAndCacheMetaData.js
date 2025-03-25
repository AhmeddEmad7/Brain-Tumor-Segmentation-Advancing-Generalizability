import { api } from 'dicomweb-client';
import dcmjs from 'dcmjs';
import { calculateSUVScalingFactors } from '@cornerstonejs/calculate-suv';
import { getPTImageIdInstanceMetadata } from './getPTImageIdInstanceMetadata';
import { utilities } from '@cornerstonejs/core';
import cornerstoneDICOMImageLoader from '@cornerstonejs/dicom-image-loader';

import ptScalingMetaDataProvider from './ptScalingMetaDataProvider';
import getPixelSpacingInformation from './getPixelSpacingInformation';
import { convertMultiframeImageIds } from './convertMultiframeImageIds';
import removeInvalidTags from './removeInvalidTags';
import * as cornerstone from '@cornerstonejs/core';

const { DicomMetaDictionary } = dcmjs.data;
const { calibratedPixelSpacingMetadataProvider } = utilities;

// Helper to extract pixelRepresentation from naturalized metadata,
// defaulting to 0 if not present.
function getPixelRepresentationFromMetadata(naturalizedMetadata) {
  const value = naturalizedMetadata.PixelRepresentation;
  return typeof value === 'number' ? value : 0;
}

/**
 * Uses dicomweb-client to fetch metadata of a study, cache it in cornerstone,
 * and return a list of imageIds for the frames.
 *
 * Uses the app config to choose which study to fetch, and which
 * dicom-web server to fetch it from.
 *
 * @returns {string[]} An array of imageIds for instances in the study.
 */
export default async function createImageIdsAndCacheMetaData({
  StudyInstanceUID,
  SeriesInstanceUID,
  SOPInstanceUID = null,
  wadoRsRoot,
  client = null,
}) {
  const SOP_INSTANCE_UID = '00080018';
  const SERIES_INSTANCE_UID = '0020000E';
  const MODALITY = '00080060';

  const studySearchOptions = {
    studyInstanceUID: StudyInstanceUID,
    seriesInstanceUID: SeriesInstanceUID,
  };

  client = client || new api.DICOMwebClient({ url: wadoRsRoot });
  const instances = await client.retrieveSeriesMetadata(studySearchOptions);
  const modality = instances[0][MODALITY].Value[0];

  let imageIds = instances
    .map((instanceMetaData) => {
      const seriesInstanceUidValue =
        instanceMetaData[SERIES_INSTANCE_UID]?.Value?.[0];
      const sopInstanceUidValue =
        SOPInstanceUID || instanceMetaData[SOP_INSTANCE_UID]?.Value?.[0];

      if (!seriesInstanceUidValue || !sopInstanceUidValue) {
        console.warn('Missing required metadata for instance, skipping:', instanceMetaData);
        return null;
      }

      const prefix = 'wadors:';
      const imageId =
        prefix +
        wadoRsRoot +
        '/studies/' +
        StudyInstanceUID +
        '/series/' +
        seriesInstanceUidValue +
        '/instances/' +
        sopInstanceUidValue +
        '/frames/1';

      // Cache instance metadata
      cornerstoneDICOMImageLoader.wadors.metaDataManager.add(imageId, instanceMetaData);
      return imageId;
    })
    .filter((id) => id !== null);

  // Convert multiframe imageIds to individual frame imageIds if needed.
  imageIds = convertMultiframeImageIds(imageIds);

  // Ensure every imageId has the expected metadata for 'imagePixelModule'
  imageIds.forEach((imageId) => {
    let instanceMetaData = cornerstoneDICOMImageLoader.wadors.metaDataManager.get(imageId);
    if (instanceMetaData) {
      instanceMetaData = removeInvalidTags(instanceMetaData);
      const naturalizedMetadata = DicomMetaDictionary.naturalizeDataset(instanceMetaData);
      let pixelModule = cornerstone.metaData.get('imagePixelModule', imageId);
      if (!pixelModule) {
        pixelModule = {
          pixelRepresentation: getPixelRepresentationFromMetadata(naturalizedMetadata),
          BitsAllocated: naturalizedMetadata.BitsAllocated ?? 16,
          BitsStored: naturalizedMetadata.BitsStored ?? 16,
          HighBit: naturalizedMetadata.HighBit ?? 15,
          PhotometricInterpretation: naturalizedMetadata.PhotometricInterpretation ?? 'MONOCHROME2',
          SamplesPerPixel: naturalizedMetadata.SamplesPerPixel ?? 1,
        };
        cornerstone.metaData.add('imagePixelModule', pixelModule, imageId);
      }
      // Optionally add calibrated pixel spacing if available.
      const pixelSpacing = getPixelSpacingInformation(naturalizedMetadata);
      if (pixelSpacing) {
        calibratedPixelSpacingMetadataProvider.add(imageId, {
          rowPixelSpacing: parseFloat(pixelSpacing[0]),
          columnPixelSpacing: parseFloat(pixelSpacing[1]),
        });
      }
    } else {
      // If no instance metadata is available, add default fallback metadata.
      const fallbackPixelModule = {
        pixelRepresentation: 0,
        BitsAllocated: 16,
        BitsStored: 16,
        HighBit: 15,
        PhotometricInterpretation: 'MONOCHROME2',
        SamplesPerPixel: 1,
      };
      cornerstone.metaData.add('imagePixelModule', fallbackPixelModule, imageId);
    }
  });

  // For PT modality, calculate and add SUV scaling factors.
  if (modality === 'PT') {
    const InstanceMetadataArray = [];
    imageIds.forEach((imageId) => {
      const instanceMetadata = getPTImageIdInstanceMetadata(imageId);
      // Temporary fix if static-wado produces a string instead of an array.
      if (instanceMetadata && typeof instanceMetadata.CorrectedImage === 'string') {
        instanceMetadata.CorrectedImage = instanceMetadata.CorrectedImage.split('\\');
      }
      if (instanceMetadata) {
        InstanceMetadataArray.push(instanceMetadata);
      }
    });
    if (InstanceMetadataArray.length) {
      try {
        const suvScalingFactors = calculateSUVScalingFactors(InstanceMetadataArray);
        InstanceMetadataArray.forEach((instanceMetadata, index) => {
          ptScalingMetaDataProvider.addInstance(imageIds[index], suvScalingFactors[index]);
        });
      } catch (error) {
        console.log(error);
      }
    }
  }

  return imageIds;
}
