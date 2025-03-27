import { cache, Enums, getEnabledElement ,  getWebWorkerManager, triggerEvent,} from '@cornerstonejs/core';
import type { Types } from '@cornerstonejs/core';
import * as cornerstoneTools from '@cornerstonejs/tools';
import store from '@/redux/store.ts';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkCellArray from '@kitware/vtk.js/Common/Core/CellArray';

import * as cornerstone from '@cornerstonejs/core';

// ✅ Import segmentation utilities from Cornerstone
// import{SegmentationRepresentation} from '@cornerstonejs/tools/src/types/SegmentationStateTypes';
// import { getSegmentation } from '@cornerstonejs/tools/dist/esm/stateManagement/segmentation/getSegmentation';
// import { getColorLUT } from '@cornerstonejs/tools/dist/esm/stateManagement/segmentation/getColorLUT';
// import { canComputeRequestedRepresentation } from '@cornerstonejs/tools/dist/esm/stateManagement/segmentation/helpers';
// import { computeAndAddSurfaceRepresentation } from '@cornerstonejs/tools/src/stateManagement/segmentation/polySeg/Surface/computeAndAddSurfaceRepresentation';
const { ViewportType } = Enums;
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
 * Converts labelmap segmentation into 3D surface.
 * @param {string} segmentationId - The ID of the segmentation.
 */
async function convertSegmentationToSurface(segmentationId: string) {
    console.log(`🔄 Converting segmentation ${segmentationId} to 3D surface...`);
    
    const segmentation = cornerstoneTools.segmentation.state.getSegmentation(segmentationId);
    if (!segmentation || !segmentation.representationData?.Labelmap) {
        console.warn('⚠️ Only labelmap segmentations can be converted.');
        return null;
    }
    
    const segmentIndices = Object.keys(segmentation.segments);
    const surfaces = await Promise.allSettled(
        segmentIndices.map((segmentIndex) =>
            convertLabelmapToSurface(segmentation.representationData.Labelmap, segmentIndex)
    )
);

const validSurfaces = surfaces
.filter((res) => res.status === 'fulfilled')
.map((res) => res.value);

    if (!validSurfaces.length) {
        console.error('❌ No valid surfaces were generated.');
        return null;
    }
    
    console.log('✅ Surface conversion complete.');
    return validSurfaces;
}

/**
 * Handles adding a surface representation to the viewport.
 * @param {Types.IVolumeViewport} viewport - The Cornerstone 3D viewport.
 * @param {string} segmentationId - The segmentation ID.
*/
export async function renderSegmentationAsSurface(viewport: Types.IVolumeViewport, segmentationId: string) {
    console.log(`🚀 Rendering 3D surface for segmentation ${segmentationId}...`);
    
    const segmentation = cornerstoneTools.segmentation.state.getSegmentation(segmentationId);
    if (!segmentation) {
        console.error(`❌ Segmentation ID ${segmentationId} not found.`);
        return;
    }
    
    let surfaceData = segmentation.representationData[cornerstoneTools.Enums.SegmentationRepresentations.Surface];
    console.log('🔍 Checking surfaceData111:', surfaceData);
    
    surfaceData = await convertSegmentationToSurface(segmentationId);
    // if (!surfaceData) {
    //     console.log(`🔄 Computing surface representation for ${segmentationId}...`);
    //     if (!surfaceData) {
    //         console.error(`❌ Surface computation failed for ${segmentationId}.`);
    //         return;
    //     }
    // }
    console.log('🔍 Checking surfaceData:', surfaceData);

    const state = store.getState();
    const { selectedViewportId } = state.viewer;
    const { currentToolGroupId } = getRenderingAndViewport(selectedViewportId);
    
    const activeSegmentation = cornerstoneTools.segmentation.activeSegmentation.getActiveSegmentationRepresentation(
        currentToolGroupId
    );

    surfaceData.forEach((surface) => {
        const color = cornerstoneTools.segmentation.config.color.getColorForSegmentIndex(
            currentToolGroupId,
            activeSegmentation.segmentationRepresentationUID,
            surface.segmentIndex
        );
        surface.color = color.slice(0, 3) as Types.Point3;
        
        addOrUpdateSurfaceToElement(viewport.element, surface, segmentationId);
    });

    viewport.render();
}
/**
 * Adds or updates a surface segmentation representation in the viewport.
*
* @param {HTMLDivElement} element - The HTML container for the viewport.
* @param {Types.ISurface} surface - The 3D surface representation.
* @param {string} segmentationId - The unique segmentation ID.
*/
export function addOrUpdateSurfaceToElement(
    element: HTMLDivElement,
    surface: Types.ISurface,
    segmentationId: string
): void {
    const enabledElement = getEnabledElement(element);
    const { viewport } = enabledElement;
    
    const existingActor = viewport.getActors().find((actor) => actor.uid === segmentationId);
    
    if (existingActor) {
        // ✅ Update existing actor
        console.log('🔄 Updating existing surface actor...');
        const surfaceMapper = existingActor.actor.getMapper();
        const currentPolyData = surfaceMapper.getInputData();
        
        if (surface.points.length === currentPolyData.getPoints().getData().length) {
            return; // No update needed
        }
        
        const polyData = vtkPolyData.newInstance();
        polyData.getPoints().setData(surface.points, 3);
        polyData.setPolys(vtkCellArray.newInstance({ values: Float32Array.from(surface.polys) }));
        
        surfaceMapper.setInputData(polyData);
        surfaceMapper.modified();
        
        viewport.getRenderer().resetCameraClippingRange();
        return;
    }

    // ✅ Create new surface actor
    console.log('🆕 Creating new surface actor...');
    const surfacePolyData = vtkPolyData.newInstance();
    surfacePolyData.getPoints().setData(surface.points, 3);
    surfacePolyData.setPolys(vtkCellArray.newInstance({ values: Float32Array.from(surface.polys) }));
    
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(surfacePolyData);
    
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(surface.color[0] / 255, surface.color[1] / 255, surface.color[2] / 255);

    viewport.addActor({
        uid: segmentationId,
        actor,
    });
    
    viewport.resetCamera();
    viewport.getRenderer().resetCameraClippingRange();
    viewport.render();
}


/**
 * Triggers progress events while processing segmentation in a WebWorker.
*/
const triggerWorkerProgress = (eventTarget, progress, segmentIndex) => {
    triggerEvent(eventTarget, Enums.Events.WEB_WORKER_PROGRESS, {
        progress,
        type: 'Converting Labelmap to Surface',
        id: segmentIndex,
    });
};

/**
 * Converts a labelmap representation into a 3D surface representation.
 * @param {LabelmapSegmentationData} labelmapRepresentationData - The labelmap segmentation data.
 * @param {number} segmentIndex - The index of the segment to convert.
 * @returns {Promise<Types.SurfaceData>} - The computed surface data.
*/
export async function convertLabelmapToSurface(
    labelmapRepresentationData,
    segmentIndex
): Promise<Types.SurfaceData> {
    let volumeId;
    
    // Check if the labelmap is volume-based or stack-based
    if (labelmapRepresentationData.volumeId) {
        volumeId = labelmapRepresentationData.volumeId;
    } else {
        const { imageIds } = labelmapRepresentationData;
        volumeId = await cornerstoneTools.segmentation.helpers.computeVolumeLabelmapFromStack({ imageIds });
    }
    
    const volume = cache.getVolume(volumeId);
    if (!volume) {
        throw new Error(`❌ No volume found for ID: ${volumeId}`);
    }
    
    const scalarData = volume.voxelManager.getCompleteScalarDataArray();
    const { dimensions, spacing, origin, direction } = volume;
    
    const workerManager = getWebWorkerManager();

    // ✅ Trigger worker progress event
    triggerWorkerProgress(cornerstoneTools.segmentation.state.getSegmentation(segmentIndex), 0, segmentIndex);
    
    // ✅ Use WebWorker to perform segmentation conversion
    const results = await workerManager.executeTask(
        'polySeg',
        'convertLabelmapToSurface',
        {
            scalarData,
            dimensions,
            spacing,
            origin,
            direction,
            segmentIndex,
        },
        {
            callbacks: [
                (progress) => {
                    triggerWorkerProgress(cornerstoneTools.segmentation.state.getSegmentation(segmentIndex), progress, segmentIndex);
                },
            ],
        }
    );
    
    // ✅ Trigger progress complete event
    triggerWorkerProgress(cornerstoneTools.segmentation.state.getSegmentation(segmentIndex), 100, segmentIndex);
    
    return results;
}
/**
//  * Converts a labelmap segmentation into a 3D surface representation.
//  *
//  * @param {string} segmentationId - The segmentation ID.
//  * @returns {Promise<RawSurfacesData>} - The computed surface data.
//  */
// async function computeSurfaceFromLabelmap(segmentationId:string) {
//     const segmentation = getSegmentation(segmentationId);
//     if (!segmentation?.representationData?.Labelmap) {
//         console.warn('⚠️ Only labelmap segmentation can be converted to surface.');
//         return null;
//     }

//     const segmentIndices = Object.keys(segmentation.representationData.Labelmap.data);

//     const surfaces = await Promise.allSettled(
//         segmentIndices.map((segmentIndex) =>
//             convertLabelmapToSurface(segmentation.representationData.Labelmap, segmentIndex)
//         )
//     );

//     const errors = surfaces.filter((p) => p.status === 'rejected');
//     if (errors.length > 0) {
//         console.error('❌ Failed to convert labelmap to surface:', errors);
//         return null;
//     }

//     return surfaces
//         .filter((surface) => surface.status === 'fulfilled')
//         .map((surface, index) => ({
//             segmentIndex: segmentIndices[index],
//             data: surface.value,
//         }));
// }

/**
 * Adds the computed surface representation to the viewport.
 *
 * @param {string} segmentationId - The segmentation ID.
 */
// async function computeAndAddSurfaceRepresentation(segmentationId:string){
//     console.log(`🔄 Computing surface segmentation for ${segmentationId}...`);
//     const surfaces = await computeSurfaceFromLabelmap(segmentationId);

//     if (!surfaces) {
//         console.error(`❌ Failed to compute surface representation for ${segmentationId}.`);
//         return;
//     }

//     console.log(`✅ Successfully converted segmentation to surface.`);

//     surfaces.forEach(({ segmentIndex, data }) => {
//         const viewportIds = getViewportIdsWithSegmentation(segmentationId);
//         viewportIds.forEach((viewportId) => {
//             addOrUpdateSurfaceToElement(getEnabledElement(viewportId).viewport.element, data, segmentationId);
//         });
//     });
// }
