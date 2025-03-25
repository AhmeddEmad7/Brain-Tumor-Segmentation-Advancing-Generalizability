import { metaData } from '@cornerstonejs/core';
import * as cornerstone from '@cornerstonejs/core';

function getFrameInformation(imageId) {
  if (imageId.includes('wadors:')) {
    const frameIndex = imageId.indexOf('/frames/');
    const imageIdFrameless = frameIndex > 0 ? imageId.slice(0, frameIndex + 8) : imageId;
    return {
      frameIndex,
      imageIdFrameless
    };
  } else {
    const frameIndex = imageId.indexOf('&frame=');
    let imageIdFrameless = frameIndex > 0 ? imageId.slice(0, frameIndex + 7) : imageId;
    if (!imageIdFrameless.includes('&frame=')) {
      imageIdFrameless = imageIdFrameless + '&frame=';
    }
    return {
      frameIndex,
      imageIdFrameless
    };
  }
}

function convertMultiframeImageIds(imageIds) {
  const newImageIds = [];
  imageIds.forEach((imageId) => {
    const { imageIdFrameless } = getFrameInformation(imageId);
    const instanceMetaData = metaData.get('multiframeModule', imageId);
    if (instanceMetaData && instanceMetaData.NumberOfFrames && instanceMetaData.NumberOfFrames > 1) {
      const numberOfFrames = instanceMetaData.NumberOfFrames;
      // Get the original imagePixelModule metadata (if it exists)
      const originalPixelModule = cornerstone.metaData.get('imagePixelModule', imageId);
      for (let i = 0; i < numberOfFrames; i++) {
        const newImageId = imageIdFrameless + (i + 1);
        // Copy the original pixel module metadata to the new imageId
        if (originalPixelModule) {
          cornerstone.metaData.add('imagePixelModule', originalPixelModule, newImageId);
        }
        newImageIds.push(newImageId);
      }
    } else {
      newImageIds.push(imageId);
    }
  });
  return newImageIds;
}

export { convertMultiframeImageIds };
