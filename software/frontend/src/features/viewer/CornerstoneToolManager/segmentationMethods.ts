import store from '@/redux/store.ts';
import * as cornerstoneTools from '@cornerstonejs/tools';
import * as cornerstone from '@cornerstonejs/core';
import { viewerSliceActions } from '@features/viewer/viewer-slice.ts';
import { adaptersSEG, helpers } from '@cornerstonejs/adapters';
import * as cornerstoneDicomImageLoader from '@cornerstonejs/dicom-image-loader';
import dcmjs from 'dcmjs';
import axios from 'axios';
import { api } from 'dicomweb-client';
import { getAndSetSeriesInstances } from '../viewer-viewport-reducers';
import { Direction } from 'react-toastify/dist/utils';

const { wadouri } = cornerstoneDicomImageLoader;
const { downloadDICOMData } = helpers;
const { Cornerstone3D } = adaptersSEG;
type SegmentStats = {
    label: number;
    voxelCount: number;
    volume: number;
    center: [number, number, number];
    meanValue: number;
    stdDev: number;
    minHU: number;
    maxHU: number;
    skewness: number;
    kurtosis: number;
};
// Constants for DICOMWeb client
const DICOM_URL = import.meta.env.VITE_ORTRHANC_PROXY_URL;
const SINGLEPART = true;

/**
 * Helper function to get rendering engine and viewport dynamically
 *
 * This function is used to get the rendering engine and viewport based on the selected viewport
 *
 * @param {number} selectedViewportId - The selected viewport ID.
 * @returns {void}
 */
export const getRenderingAndViewport = (selectedViewportId: string) => {
    // Get the current application state
    const state = store.getState();
    const { segmentations, renderingEngineId, currentToolGroupId } = state.viewer;

    // Get the rendering engine and viewport using the selected viewport ID
    const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
    const viewport = renderingEngine?.getViewport(selectedViewportId) as cornerstone.Types.IVolumeViewport;

    // Return the rendering engine, viewport, current tool group ID, and selected viewport ID
    return { segmentations, renderingEngine, viewport, currentToolGroupId, selectedViewportId };
};

/**
 * Adds a segment to a specific segmentation representation.
 *
 * This function is called when the user clicks on the add segment button in the segmentation panel.
 *
 * @param {number} numberOfSegments - The number of segments to add to the segmentation.
 * @returns {void}
 */
export const addSegmentToSegmentation = (numberOfSegments: number) => {
    const { segmentations } = store.getState().viewer;

    // Get the current active segmentation
    const segmentation = segmentations.find((segmentation) => segmentation.isActive === true);

    if (segmentation) {
        // Dispatch action to add the specified number of segments to the active segmentation
        store.dispatch(viewerSliceActions.addSegment({ segmentationId: segmentation.id, numberOfSegments }));

        // Set the active segment index to the newly added segment
        cornerstoneTools.segmentation.segmentIndex.setActiveSegmentIndex(
            segmentation.id,
            segmentation.segments.length + 1
        );
    }
};

/**
 * Add a new segmentation to the viewer state
 *
 * This function is called when the user clicks on the add segmentation button in the segmentation panel
 */
export const addSegmentation = async () => {
    const state = store.getState();
    const { selectedViewportId } = state.viewer;
    const { viewport, currentToolGroupId, segmentations } = getRenderingAndViewport(selectedViewportId);

    const newSegmentationId = `SEGMENTATION_${segmentations.length}`;
    const is3DActive = true;
    addSegmentationsToState(newSegmentationId, viewport, currentToolGroupId, 1, is3DActive);

    viewport.render();
};

/**
 * Downloads the current segmentation mask as a DICOM (.dcm) file.
 *
 * This function is called when the user clicks on the download button in the segmentation panel.
 *
 * @returns {Promise<void>} A promise that resolves when the segmentation mask has been downloaded.
 */
export const showFillAndOutline = async () => {
    const state = store.getState();
    const { selectedViewportId } = state.viewer;
    cornerstoneTools.BidirectionalTool;
    // Retrieve the rendering engine and viewport using the selected viewport ID.
    const { currentToolGroupId } = getRenderingAndViewport(selectedViewportId);
    const activeSegmentationRepresentation =
        cornerstoneTools.segmentation.activeSegmentation.getActiveSegmentationRepresentation(
            currentToolGroupId
        );
    const canConvertToSurface = cornerstoneTools.segmentation.polySeg.canComputeRequestedRepresentation(
        activeSegmentationRepresentation.segmentationRepresentationUID
    );
    console.log('canConvertToSurface', canConvertToSurface);
    console.log('UID', activeSegmentationRepresentation.segmentationRepresentationUID);
    cornerstoneTools.segmentation.config.setSegmentationRepresentationSpecificConfig(
        currentToolGroupId,
        activeSegmentationRepresentation.segmentationRepresentationUID,
        {
            [cornerstoneTools.Enums.SegmentationRepresentations.Labelmap]: {
                renderFill: true,
                renderOutline: true,
                fillAlpha: 0.9,
                outlineOpacity: 1,
                outlineWidthActive: 2
            }
        }
    );
};

export const showOutlineOnly = async () => {
    const state = store.getState();
    const { selectedViewportId } = state.viewer;

    // Retrieve the rendering engine and viewport using the selected viewport ID.
    const { currentToolGroupId } = getRenderingAndViewport(selectedViewportId);
    const activeSegmentationRepresentation =
        cornerstoneTools.segmentation.activeSegmentation.getActiveSegmentationRepresentation(
            currentToolGroupId
        );

    cornerstoneTools.segmentation.config.setSegmentationRepresentationSpecificConfig(
        currentToolGroupId,
        activeSegmentationRepresentation.segmentationRepresentationUID,
        {
            [cornerstoneTools.Enums.SegmentationRepresentations.Labelmap]: {
                renderFill: false,
                renderOutline: true,
                outlineWidthActive: 2,
                fillAlpha: 0.9,
                outlineOpacity: 0.9
            }
        }
    );
};

export const setSegmentationOpacity = async (opacity: number) => {
    const state = store.getState();
    const { selectedViewportId } = state.viewer;

    // Retrieve the rendering engine and viewport using the selected viewport ID.
    const { currentToolGroupId } = getRenderingAndViewport(selectedViewportId);
    const activeSegmentationRepresentation =
        cornerstoneTools.segmentation.activeSegmentation.getActiveSegmentationRepresentation(
            currentToolGroupId
        );

    // Set the opacity for the segmentation representation
    cornerstoneTools.segmentation.config.setSegmentationRepresentationSpecificConfig(
        currentToolGroupId,
        activeSegmentationRepresentation.segmentationRepresentationUID,
        {
            [cornerstoneTools.Enums.SegmentationRepresentations.Labelmap]: {
                fillAlpha: opacity,
                outlineOpacity: opacity
            }
        }
    );
};
export const saveSegmentation = async () => {
    const state = store.getState();
    const { selectedViewportId } = state.viewer;
    const { viewport, currentToolGroupId } = getRenderingAndViewport(selectedViewportId);
    const viewportVolumeId = viewport.getActorUIDs()[0];
    const cacheVolume = cornerstone.cache.getVolume(viewportVolumeId);

    // Get the active segmentation representation
    const activeSegmentationRepresentation =
        cornerstoneTools.segmentation.activeSegmentation.getActiveSegmentationRepresentation(
            currentToolGroupId
        );

    // Get segmentation volume
    const cacheSegmentationVolume = cornerstone.cache.getVolume(
        activeSegmentationRepresentation.segmentationId
    );

    const segmentation3D = {
        data: Array.from(new Uint8Array(cacheSegmentationVolume.getScalarData())),
        dimensions: cacheSegmentationVolume.dimensions,
        spacing: cacheSegmentationVolume.spacing,
        origin: cacheSegmentationVolume.origin,
        direction: cacheSegmentationVolume.direction
    };

    // Generate label maps
    const labelmapData = Cornerstone3D.Segmentation.generateLabelMaps2DFrom3D(cacheSegmentationVolume);
    const segmentsMetadata = {};

    // Generate metadata for each segment
    labelmapData.segmentsOnLabelmap.forEach((segmentIndex) => {
        const color = cornerstoneTools.segmentation.config.color.getColorForSegmentIndex(
            currentToolGroupId,
            activeSegmentationRepresentation.segmentationRepresentationUID,
            segmentIndex
        );
        segmentsMetadata[segmentIndex] = generateMockMetadata(segmentIndex, color);
    });

    // Generate segmentation dataset
    const generatedSegmentation = Cornerstone3D.Segmentation.generateSegmentation(
        cacheVolume.imageIds,
        { ...labelmapData, metadata: segmentsMetadata },
        cornerstone.metaData
    );

    const relevantMetadata = {
        // Required DICOM Metadata
        SOPClassUID: generatedSegmentation.dataset.SOPClassUID,
        SOPInstanceUID: generatedSegmentation.dataset.SOPInstanceUID,
        Modality: generatedSegmentation.dataset.Modality,
        SeriesInstanceUID: generatedSegmentation.dataset.SeriesInstanceUID,
        StudyInstanceUID: generatedSegmentation.dataset.StudyInstanceUID,
        FrameOfReferenceUID: generatedSegmentation.dataset.FrameOfReferenceUID,

        ContentLabel: generatedSegmentation.dataset.ContentLabel,
        ContentDescription: generatedSegmentation.dataset.ContentDescription,
        SeriesDescription: generatedSegmentation.dataset.SeriesDescription,

        // Patient Info (if applicable)
        PatientName: generatedSegmentation.dataset.PatientName || 'Anonymous',
        PatientID: generatedSegmentation.dataset.PatientID || '0000',
        PatientBirthDate: generatedSegmentation.dataset.PatientBirthDate || '',
        PatientSex: generatedSegmentation.dataset.PatientSex || 'O',

        // Study Info
        StudyDate: generatedSegmentation.dataset.StudyDate || '20241217',
        StudyTime: generatedSegmentation.dataset.StudyTime || '000000',
        StudyID: generatedSegmentation.dataset.StudyID || '123456',
        AccessionNumber: generatedSegmentation.dataset.AccessionNumber || '1234',

        // Segment and Pixel Data
        SegmentSequence: generatedSegmentation.dataset.SegmentSequence,
        PerFrameFunctionalGroupsSequence: generatedSegmentation.dataset.PerFrameFunctionalGroupsSequence,
        Rows: generatedSegmentation.dataset.Rows,
        Columns: generatedSegmentation.dataset.Columns,
        NumberOfFrames: generatedSegmentation.dataset.NumberOfFrames,
        PixelData: Array.from(new Uint8Array(generatedSegmentation.dataset.PixelData)),

        // Image Metadata
        SamplesPerPixel: generatedSegmentation.dataset.SamplesPerPixel || '1',
        BitsAllocated: generatedSegmentation.dataset.BitsAllocated || '8',
        BitsStored: generatedSegmentation.dataset.BitsStored || '8',
        HighBit: generatedSegmentation.dataset.HighBit || '7',
        PixelRepresentation: generatedSegmentation.dataset.PixelRepresentation || '0',
        PhotometricInterpretation: generatedSegmentation.dataset.PhotometricInterpretation || 'MONOCHROME2'
    };
    console.log('generatedSegmentation', generatedSegmentation.dataset);

    try {
        const response = await axios.post('http://localhost:8000/inference/segmentation/uploadorthanc', {
            dataset: relevantMetadata,
            segmentationData: segmentation3D // Add Segmentation Mask Data
        });

        if (response.status === 200) {
            console.log('Segmentation uploaded successfully:', response.data.message);
            alert('Segmentation uploaded successfully!');
        } else {
            console.error('Error uploading segmentation:', response.data.message);
            alert('Failed to upload segmentation!');
        }
    } catch (error) {
        alert('An error occurred while uploading segmentation.');
    }
};
// Enable both fill and outline for segmentations on a given element
// export  function showSegmentationFillAndOutline(element) {
//     const segmentationModule = cornerstoneTools.getModule('segmentation');
//     // Turn on filled regions and outlines
//     segmentationModule.configuration.renderFill = true;
//     segmentationModule.configuration.renderOutline = true;
//     // (Optional) Adjust opacity or outline thickness if needed:
//     // segmentationModule.configuration.fillAlpha = 0.7;    // 70% opacity fill
//     // segmentationModule.configuration.outlineAlpha = 0.9; // 90% opacity outline
//     // segmentationModule.configuration.outlineWidth = 1;   // 1px outline thickness (default)
//     // Re-render the image to apply changes
//     cornerstone.updateImage(element);
//   }

export const downloadSegmentation = async () => {
    // Get the current application state and selected viewport ID.
    const state = store.getState();
    const { selectedViewportId } = state.viewer;

    // Retrieve the rendering engine and viewport using the selected viewport ID.
    const { viewport, currentToolGroupId } = getRenderingAndViewport(selectedViewportId);

    // Get the volume ID from the viewport.
    const viewportVolumeId = viewport.getActorUIDs()[0];

    // Retrieve the cached volume data.
    // Get the active segmentation representation for the current tool group.
    const activeSegmentationRepresentation =
        cornerstoneTools.segmentation.activeSegmentation.getActiveSegmentationRepresentation(
            currentToolGroupId
        );

    // Retrieve the cached segmentation volume data.
    const cacheSegmentationVolume = cornerstone.cache.getVolume(
        activeSegmentationRepresentation.segmentationId
    );

    const segmentationData = Array.from(new Uint8Array(cacheSegmentationVolume.scalarData));
    console.log('segmentationData', segmentationData);
    // Generate 2D label maps from the 3D segmentation volume.
    const labelmapData = Cornerstone3D.Segmentation.generateLabelMaps2DFrom3D(cacheSegmentationVolume);

    // Initialize metadata array for segments.
    labelmapData.metadata = [];

    // Generate metadata for each segment in the label map.
    labelmapData.segmentsOnLabelmap.forEach((segmentIndex) => {
        const color = cornerstoneTools.segmentation.config.color.getColorForSegmentIndex(
            currentToolGroupId,
            activeSegmentationRepresentation.segmentationRepresentationUID,
            segmentIndex
        );
        const segmentMetadata = generateMockMetadata(segmentIndex, color);
        labelmapData.metadata[segmentIndex] = segmentMetadata;
    });

    // Generate the segmentation dataset for DICOM export.
    const generatedSegmentation = Cornerstone3D.Segmentation.generateSegmentation(
        csImages,
        labelmapData,
        cornerstone.metaData
    );
    console.log('generatedSegmentation', generatedSegmentation.dataset);

    const fileName = `${viewportVolumeId}_SEG.dcm`;

    // Download the generated segmentation dataset as a DICOM file.
    downloadDICOMData(generatedSegmentation.dataset, fileName);
};

/**
 * Generates mock metadata for a segment based on the segment index and color.
 *
 * @param {number} segmentIndex - The index of the segment.
 * @param {any} color - The color of the segment in RGB format.
 * @returns {object} The mock metadata for the segment.
 */
function generateMockMetadata(segmentIndex: number, color: any) {
    // Convert the RGB color values to CIELab values and round them
    const RecommendedDisplayCIELabValue = dcmjs.data.Colors.rgb2DICOMLAB(
        color.slice(0, 3).map((value: number) => value / 255)
    ).map((value: number) => Math.round(value));

    // Return the mock metadata object for the segment
    return {
        SegmentedPropertyCategoryCodeSequence: {
            CodeValue: 'T-D0050',
            CodingSchemeDesignator: 'SRT',
            CodeMeaning: 'Tissue'
        },
        SegmentNumber: segmentIndex.toString(),
        SegmentLabel: 'Tissue ' + segmentIndex.toString(),
        SegmentAlgorithmType: 'SEMIAUTOMATIC',
        SegmentAlgorithmName: 'Slicer Prototype',
        RecommendedDisplayCIELabValue,
        SegmentedPropertyTypeCodeSequence: {
            CodeValue: 'T-D0050',
            CodingSchemeDesignator: 'SRT',
            CodeMeaning: 'Tissue'
        }
    };
}

/**
 * Creates an input element for file selection and triggers a click to open the file dialog.
 * This function is called when the user clicks on the upload button in the segmentation panel.
 *
 * @returns {Promise<void>} A promise that resolves when the segmentation files have been selected and processed.
 */
export const uploadSegmentation = async () => {
    // Create an input element for file selection
    const inputElement = document.createElement('input');
    inputElement.type = 'file';
    inputElement.accept = '.dcm';
    inputElement.multiple = true;

    // Add an event listener to handle file selection
    inputElement.addEventListener('change', async (event) => {
        let eventTarget = (event.target as HTMLInputElement) || null;
        if (eventTarget?.files) {
            // Process each selected file
            for (let i = 0; i < eventTarget.files.length; i++) {
                const file = eventTarget.files[i];
                await readSegmentation(file);
            }
        }
    });

    // Trigger a click on the input element to open the file dialog
    inputElement.click();
};

/**
 * Reads the segmentation input and loads it into the viewer.
 *
 * @param {File | string} input - The input segmentation data, either as a File object or a Series UID string.
 * @returns {Promise<void>} A promise that resolves when the segmentation has been loaded into the viewer.
 */
export const readSegmentation = async (input: File | string) => {
    let imageId, arrayBuffer;
    const state = store.getState();
    const { currentStudyInstanceUid, is3DActive, selectedViewportId, renderingEngineId } = state.viewer; // ✅ Get 3D flag

    if (typeof input === 'string') {
        const seriesInstanceUID = input;
        const client = new api.DICOMwebClient({ url: DICOM_URL, singlepart: SINGLEPART });

        const SOPInstanceUIDs = await getAndSetSeriesInstances(currentStudyInstanceUid, seriesInstanceUID);

        arrayBuffer = await client.retrieveInstance({
            studyInstanceUID: currentStudyInstanceUid,
            seriesInstanceUID: seriesInstanceUID,
            sopInstanceUID: SOPInstanceUIDs[0]
        });
    } else {
        imageId = wadouri.fileManager.add(input);

        const image = await cornerstone.imageLoader.loadAndCacheImage(imageId);
        const instance = cornerstone.metaData.get('instance', imageId);

        if (instance.Modality !== 'SEG' && instance.Modality !== 'seg') {
            console.error('This is not a segmentation file.');
            return;
        }

        arrayBuffer = image.data.byteArray.buffer;
    }
    // const { selectedViewportId ,renderingEngineId} = state.viewer;

    console.log('arrayBuffer', arrayBuffer);
    // const { viewport, currentToolGroupId } = getRenderingAndViewport(selectedViewportId);
    //     if(is3DActive){

    //         const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
    //         if (!renderingEngine) return;

    //         const segResult = await Cornerstone3D.Segmentation.generateToolState( viewport.getImageIds(), arrayBuffer, cornerstone.metaData);
    //         const segBuffer = segResult.labelmapBufferArray[0];  // assuming one segment
    //     const voxels = new Uint8Array(segBuffer);
    //     // console.log(`Max label value = ${Math.max(...voxels)}`);  // expect 1 or more
    //     const labelArray = new Uint8Array(segResult.labelmapBufferArray[0]);
    //     const vtkSegData = vtkDataArray.newInstance({
    //       values: labelArray,
    //       numberOfComponents: 1,
    //       name: 'SegmentMask'
    //     });
    //     // 1) Grab the primary volume ID from the viewport
    //     const primaryVolumeId = viewport.getActorUIDs()[0];
    //     console.log('primaryVolumeId', primaryVolumeId);
    //     // 2) Fetch it out of the Cornerstone cache
    //     const primaryVolume   = cornerstone.cache.getVolume(primaryVolumeId)!;
    //     console.log('primaryVolume', primaryVolume);
    //     // 3) Now pull its real geometry
    //     const dims    = primaryVolume.imageData!.getDimensions();
    //     const spacing = primaryVolume.imageData!.getSpacing();
    //     const origin  = primaryVolume.imageData!.getOrigin();

    //     // 4) Use *those* when you build your vtkImageData:
    //     const segImageData = vtkImageData.newInstance();
    //     segImageData.setOrigin(...origin);
    //     segImageData.setSpacing(...spacing);
    //     segImageData.setExtent(
    //         0, dims[0] - 1,
    //         0, dims[1] - 1,
    //         0, dims[2] - 1
    //     );

    //     segImageData.getPointData().setScalars(vtkSegData);

    //     const marcher = vtkImageMarchingCubes.newInstance({
    //         contourValue: 0.5,
    //         computeNormals: true,
    //         mergePoints: true
    //     });
    //     marcher.setInputData(segImageData);
    //     const segmentationMesh = marcher.getOutputData();
    //       const segMapper = vtkMapper.newInstance();
    //     segMapper.setInputData(segmentationMesh);
    //     const segActor = vtkActor.newInstance();
    //     segActor.setMapper(segMapper);
    //     viewport.addActor(
    //         {uid: 'SegRawVTK',actor:segActor});
    //     viewport.resetCamera();
    //     viewport.render();
    //     return;
    // }

    const stats = computeSegmentationStat();
    console.log('stats ', stats);
    if (!arrayBuffer) {
        console.error('Failed to load segmentation: No data available.');
        return;
    }

    // ✅ Load Segmentation Based on Viewport Type (3D vs 2D)
    await loadSegmentation(arrayBuffer, is3DActive);
};

export const computeSegmentationStat = async () => {
    const state = store.getState();
    const { selectedViewportId, segmentations } = state.viewer;
    console.log('segmentations volume', segmentations[0]);
    console.log('segmentations uid', segmentations[1]);
    console.log('segmentations segment index', segmentations[2]);
    cornerstoneTools.segmentation.polySeg.computeAndAddSurfaceRepresentation;
    // Get the segmentation linked to the current viewport
    const segmentation = segmentations.find((seg) => seg.isActive);

    if (!segmentation) {
        console.warn('❌ No segmentation found for selected viewport');
        return;
    }
    console.log('segmentation.segmentationVolume', segmentation.segmentationVolume);
    const segmentationVolume = cornerstone.cache.getVolume(segmentation.segmentationVolume);
    console.log('segmentationVolume123', segmentationVolume);
    const sourceVolume = cornerstone.cache.getVolume(segmentation.volumeId!);

    if (!segmentationVolume || !sourceVolume) {
        console.warn('🚫 Missing segmentation or source volume in cache');
        return;
    }
    const labelData = new Uint8Array(segmentationVolume.getScalarData());
    console.log('labelData', labelData);
    const imageData = new Float32Array(sourceVolume.getScalarData());
    const [dimX, dimY, dimZ] = segmentationVolume.dimensions;
    const [sx, sy, sz] = segmentationVolume.spacing;
    const voxelVolume = sx * sy * sz;

    const statsByLabel: Record<number, any> = {};

    for (let z = 0; z < dimZ; z++) {
        for (let y = 0; y < dimY; y++) {
            for (let x = 0; x < dimX; x++) {
                const index = z * dimY * dimX + y * dimX + x;
                const label = labelData[index];

                if (label === 0) continue;

                const value = imageData[index];

                if (!statsByLabel[label]) {
                    const segment = segmentation.segments.find((s) => s.segmentIndex === label);
                    statsByLabel[label] = {
                        labelName: segment?.label ?? `Segment ${label}`,
                        color: segment?.color ?? [255, 255, 255],
                        count: 0,
                        sum: 0,
                        values: [],
                        min: Infinity,
                        max: -Infinity,
                        sumX: 0,
                        sumY: 0,
                        sumZ: 0
                    };
                }

                const s = statsByLabel[label];

                s.count++;
                s.sum += value;
                s.values.push(value);
                s.min = Math.min(s.min, value);
                s.max = Math.max(s.max, value);
                s.sumX += x;
                s.sumY += y;
                s.sumZ += z;
            }
        }
    }

    // Final stats calculations
    for (const label in statsByLabel) {
        const s = statsByLabel[label];
        s.mean = s.sum / s.count;

        let varianceSum = 0;
        for (const val of s.values) {
            varianceSum += Math.pow(val - s.mean, 2);
        }
        s.std = Math.sqrt(varianceSum / s.count);

        s.values.sort((a, b) => a - b);
        const mid = Math.floor(s.count / 2);
        s.median = s.count % 2 === 0 ? (s.values[mid - 1] + s.values[mid]) / 2 : s.values[mid];

        let skewSum = 0;
        let kurtSum = 0;
        for (const val of s.values) {
            const diff = (val - s.mean) / s.std;
            skewSum += diff ** 3;
            kurtSum += diff ** 4;
        }
        s.skewness = skewSum / s.count;
        s.kurtosis = kurtSum / s.count - 3;

        s.volume = s.count * voxelVolume;

        const cx = s.sumX / s.count;
        const cy = s.sumY / s.count;
        const cz = s.sumZ / s.count;
        s.center = [
            segmentationVolume.origin[0] + cx * sx,
            segmentationVolume.origin[1] + cy * sy,
            segmentationVolume.origin[2] + cz * sz
        ];

        delete s.values; // optional
    }

    console.log('🧠 Segment Statistics:', statsByLabel);
    return statsByLabel;
};

/**
 * Loads a segmentation into the viewer.
 *
 * @param {ArrayBuffer} arrayBuffer - The array buffer containing the segmentation data.
 * @param {boolean} is3DActive - Whether 3D segmentation should be used.
 */
async function loadSegmentation(arrayBuffer: ArrayBuffer, is3DActive: boolean) {
    const state = store.getState();
    const { selectedViewportId, renderingEngineId } = state.viewer;
    const { viewport, currentToolGroupId } = getRenderingAndViewport(selectedViewportId);
    cornerstoneTools.segmentation;
    if (!viewport) {
        console.error('❌ Viewport not found for segmentation.');
        return;
    }

    const newSegmentationId = 'LOAD_SEGMENTATION_ID:' + cornerstone.utilities.uuidv4();
    console.log('Segmentation Volume ID in segmentation methods:', newSegmentationId);

    // ✅ Generate segmentation tool state

    ////////////////////test /////////////////////////
    const generateToolState = await Cornerstone3D.Segmentation.generateToolState(
        viewport.getImageIds(),
        arrayBuffer,
        cornerstone.metaData
    );
    console.log('🔄 Generating Segmentation Tool State:', generateToolState);

    // ✅ Add segmentation state (Labelmap for 2D, Surface for 3D)
    const derivedVolume = await addSegmentationsToState(
        newSegmentationId,
        viewport,
        currentToolGroupId,
        generateToolState.segMetadata.data.length - 1,
        is3DActive
    );
    const stats = computeSegmentationStat();
    console.log('stats ', stats);
    console.log('✅ Derived Volume Created:', derivedVolume);

    // ✅ Assign segmentation data to volume
    const scalarData = derivedVolume.getScalarData();
    const combinedBuffer = new Uint8Array(scalarData.length);
    let offset = 0;
    console.log('scalarData', scalarData);
    console.log('combinedBuffer', combinedBuffer);

    for (const slice of generateToolState.labelmapBufferArray) {
        combinedBuffer.set(new Uint8Array(slice), offset);
        offset += slice.byteLength;
    }
    console.log('  derivedVolume.imageData', derivedVolume.imageData);
    scalarData.set(combinedBuffer);
    const imageData = derivedVolume.imageData;
    imageData.getPointData().getScalars().setData(scalarData); // ⬅️ critical!
    console.log('imageData1', imageData);
    imageData.modified(); // ⬅️ refreshes internal structures
    console.log('imageData2', imageData);
    viewport.render();
}

async function ensureSurfaceSegmentation() {
    const state = store.getState();
    const { selectedViewportId } = state.viewer;
    const { currentToolGroupId } = getRenderingAndViewport(selectedViewportId);
    const activeSegmentationRepresentation =
        cornerstoneTools.segmentation.activeSegmentation.getActiveSegmentationRepresentation(
            currentToolGroupId
        );

    try {
        // Use standard surface representation
        await cornerstoneTools.segmentation.addSegmentationRepresentations(currentToolGroupId, [
            {
                segmentationId: activeSegmentationRepresentation.segmentationId,
                type: cornerstoneTools.Enums.SegmentationRepresentations.Surface,
                options: {
                    // Add any surface-specific options here
                }
            }
        ]);
    } catch (err) {
        console.error('Surface conversion failed:', err);
    }
}
/**
 * Adds segmentations to the application state and associates them with a tool group.
 *
 * @param {string} segmentationId - The unique identifier for the segmentation.
 * @param {cornerstone.Types.IVolumeViewport} viewport - The viewport where the segmentation is applied.
 * @param {string} currentToolGroupId - The ID of the tool group managing segmentation.
 * @param {number} numberOfSegments - The number of segments to add.
 * @param {boolean} is3DActive - Whether segmentation should be surface-based (3D).
 * @returns {Promise<cornerstone.Volume>} A promise that resolves to the derived segmentation volume.
 */
async function addSegmentationsToState(
    segmentationId: string,
    viewport: cornerstone.Types.IVolumeViewport,
    currentToolGroupId: string,
    numberOfSegments: number,
    is3DActive: boolean
) {
    const viewportVolumeId = viewport.getActorUIDs()[0];

    const derivedVolume = await cornerstone.volumeLoader.createAndCacheDerivedSegmentationVolume(
        viewportVolumeId,
        { volumeId: segmentationId }
    );
    const representationType = is3DActive
        ? cornerstoneTools.Enums.SegmentationRepresentations.Surface
        : cornerstoneTools.Enums.SegmentationRepresentations.Labelmap;

    await cornerstoneTools.segmentation.addSegmentations([
        {
            segmentationId,
            representation: {
                type: cornerstoneTools.Enums.SegmentationRepresentations.Labelmap,
                data: { volumeId: segmentationId }
            }
        }
    ]);

    const [uid] = await cornerstoneTools.segmentation.addSegmentationRepresentations(currentToolGroupId, [
        {
            segmentationId,
            type: cornerstoneTools.Enums.SegmentationRepresentations.Labelmap,
            options: {
                polySeg: {
                    enabled: true
                }
            }
        }
    ]);

    cornerstoneTools.segmentation.activeSegmentation.setActiveSegmentationRepresentation(
        currentToolGroupId,
        uid
    );
    console.log('segmentation bbnbossss henaaaasd');
    // 🧠 If in 3D mode, also compute Surface representation
    if (is3DActive) {
        // cornerstoneTools.segmentation.polySeg.
        console.log('🔁 Converting Labelmap to Surface...');
        await ensureSurfaceSegmentation();
    }

    store.dispatch(
        viewerSliceActions.addSegmentation({
            id: segmentationId,
            volumeId: viewportVolumeId,
            uid,
            segmentationVolume: derivedVolume.volumeId
        })
    );

    addSegmentToSegmentation(numberOfSegments);

    return derivedVolume;
}

export default getRenderingAndViewport;
