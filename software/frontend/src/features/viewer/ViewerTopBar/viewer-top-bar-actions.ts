import store from '@/redux/store.ts';
import { viewerSliceActions } from '@features/viewer/viewer-slice.ts';
import {
    CornerstoneToolManager,
    ANNOTATION_TOOLS,
    SEGMENTATION_TOOLS
} from '@/features/viewer/CornerstoneToolManager/';
// import { OrientationAxis } from '@cornerstonejs/core/src/enums';
import { Types } from '@cornerstonejs/core';
import { IStore } from '@/models';
import { getRenderingEngine, Enums, setVolumesForViewports, CONSTANTS } from '@cornerstonejs/core';
import * as cornerstone from '@cornerstonejs/core';
import { createImageIdsAndCacheMetaData } from '@utilities/helpers/index';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkColorMaps from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction/ColorMaps';

export const toggleFullScreen = () => {
    const state = store.getState();
    const { isFullScreen } = state.viewer;
    const elem = document.getElementById('root') as HTMLElement & {
        exitFullscreen(): Promise<void>;
        requestFullscreen(): Promise<void>;
    };

    if (isFullScreen) {
        document.exitFullscreen().then(() => {
            store.dispatch(viewerSliceActions.toggleFullScreen());
        });
    } else {
        elem!.requestFullscreen().then(() => {
            store.dispatch(viewerSliceActions.toggleFullScreen());
        });
    }
};

export const toggleVolumeRendering = async (forceTo2D = false) => {
    const state = store.getState();
    const { selectedViewportId, renderingEngineId, selectedSeriesInstanceUid, currentStudyInstanceUid } =
        state.viewer;
    const renderingEngine = getRenderingEngine(renderingEngineId);

    if (!renderingEngine) {
        console.error('❌ Rendering engine not found.');
        return;
    }

    // **Step 1: Get Current Viewport**
    await new Promise((resolve) => setTimeout(resolve, 200));
    let viewport = renderingEngine.getViewport(selectedViewportId) as
        | Types.IVolumeViewport
        | Types.IStackViewport;

    if (!viewport) {
        console.error(`❌ Viewport ${selectedViewportId} not found.`);
        return;
    }

    // **Step 2: Check If Already in 3D Mode**
    const isCurrently3D = viewport.type === Enums.ViewportType.VOLUME_3D;

    // ✅ **Force to 2D if `forceTo2D` is true (when changing series)**
    const newViewportType =
        forceTo2D || isCurrently3D ? Enums.ViewportType.ORTHOGRAPHIC : Enums.ViewportType.VOLUME_3D;
    console.log(
        `🔄 Switching to: ${newViewportType === Enums.ViewportType.VOLUME_3D ? '3D Volume' : '2D Stack'}`
    );

    // **Step 3: Retrieve Image IDs**
    const imageIds = await createImageIdsAndCacheMetaData({
        StudyInstanceUID: currentStudyInstanceUid,
        SeriesInstanceUID: selectedSeriesInstanceUid,
        wadoRsRoot: import.meta.env.VITE_ORTRHANC_PROXY_URL
    });

    if (!imageIds || imageIds.length === 0) {
        console.error('❌ No image IDs found for the given series.');
        return;
    }

    // **Step 4: Configure and Enable New Viewport**
    // const viewportInputArray = [
    //     {
    //         viewportId: selectedViewportId,
    //         type: newViewportType,
    //         element: viewport.element as HTMLDivElement,
    //         defaultOptions: {
    //             orientation: 'axial', // Reset to default AXIAL for 2D
    //             background: CONSTANTS.BACKGROUND_COLORS.slicer3D as [number, number, number]
    //         }
    //     }
    // ];
    // renderingEngine.setViewports(viewportInputArray);
    // renderingEngine.enableElement({
    //     viewportId: selectedViewportId,
    //     type: newViewportType,
    //     element: viewport.element as HTMLDivElement
    // });
    renderingEngine.enableElement({
        viewportId: selectedViewportId,
        type: newViewportType,
        element: viewport.element as HTMLDivElement,
        defaultOptions: {
            orientation: 'axial',
            // background: CONSTANTS.BACKGROUND_COLORS.slicer3D as [number, number, number]
        }
    });
    // **Step 5: Retrieve Updated Viewport**
    viewport = renderingEngine.getViewport(selectedViewportId) as
        | Types.IVolumeViewport
        | Types.IStackViewport;

    try {
        console.log('🔍 Loading Volume or Stack...');

        const volumeId = `cornerstoneStreamingImageVolume:${selectedSeriesInstanceUid}`;

        if (newViewportType === Enums.ViewportType.ORTHOGRAPHIC) {
            // ✅ **Load 2D Stack**
            await setVolumesForViewports(renderingEngine, [{ volumeId }], [selectedViewportId]);
            CornerstoneToolManager.setCurrentToolGroupId('CornerstoneTools2D');
            console.log('✅ Switched to 2D Stack Mode');
        } else {
            // ✅ **Load 3D Volume**
            const volume = await cornerstone.volumeLoader.createAndCacheVolume(volumeId, { imageIds });
            await volume.load();
            await setVolumesForViewports(renderingEngine, [{ volumeId }], [selectedViewportId]);
            viewport.setProperties({
                preset: 'MR-Default',
                background: CONSTANTS.BACKGROUND_COLORS.slicer3D as [number, number, number]
            });
            CornerstoneToolManager.setCurrentToolGroupId('CornerstoneTools3D');
            console.log('🎉 3D Volume Rendering Activated');
        }

        viewport.render();
    } catch (error) {
        console.error('❌ Error loading volume/stack:', error);
    }
    console.log('newViewportType', newViewportType);
    console.log('Enums.ViewportType.VOLUME_3D', Enums.ViewportType.VOLUME_3D);
    console.log('Enums.ViewportType.ORTHOGRAPHIC', Enums.ViewportType.VOLUME_3D === newViewportType);
    // **Step 6: Update Redux State**
    store.dispatch(viewerSliceActions.set3DActive(newViewportType === Enums.ViewportType.VOLUME_3D));
};

export const toggleMPRMode = async (
    renderingEngineId: string,
    selectedSeriesInstanceUid: string,
    currentStudyInstanceUid: string
) => {
    const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
    const wadoRsRoot = import.meta.env.VITE_ORTRHANC_PROXY_URL;

    if (!renderingEngine) {
        console.error('Rendering engine not found');
        return;
    }

    if (!selectedSeriesInstanceUid) {
        console.error('Invalid selectedSeriesInstanceUid provided.');
        return;
    }

    const orientations: OrientationAxis[] = ['axial', 'coronal', 'sagittal'];

    const volumeId = `cornerstoneStreamingImageVolume:${selectedSeriesInstanceUid}`;

    // Load the volume and cache metadata
    const imageIds = await createImageIdsAndCacheMetaData({
        StudyInstanceUID: currentStudyInstanceUid,
        SeriesInstanceUID: selectedSeriesInstanceUid,
        wadoRsRoot: wadoRsRoot
    });
    console.log("imageIds",imageIds )
    const volume = await cornerstone.volumeLoader.createAndCacheVolume(volumeId, {
        imageIds
    });
    await volume.load();

    // Update layout for 1 row and 3 columns
    store.dispatch(
        viewerSliceActions.toggleMPRLayout({
            orientations
        })
    );

    // Allow some time for the layout to apply
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Get all viewports
    const viewports = [0, 1, 2].map(
        (index) => renderingEngine.getViewport(`viewport-${index}`) as Types.IVolumeViewport | undefined
    );

    if (viewports.some((viewport) => !viewport)) {
        console.error('One or more viewports are not initialized.');
        return;
    }
    // Assign the volume to each viewport and set orientations
    viewports.forEach((viewport, index) => {
        if (viewport) {
            try {
                // Assign the volume
                viewport.setVolumes([{ volumeId }], true);

                // Set the orientation (Axial, Sagittal, Coronal)
                viewport.setOrientation(orientations[index]);
                viewport.render();
            } catch (error) {
                console.error(`Error configuring viewport ${index}:`, error);
            }
        }
    });
    // store.dispatch(viewerSliceActions.removeClickedSeries());
};

export const toggleViewportOverlayShown = () => {
    store.dispatch(viewerSliceActions.toggleInfoOnViewports());
};
export const applyColormapToViewport = (
    colormapName: string,
    renderingEngineId: string,
    viewportId: string,
    volumeId: string
) => {
    const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
    const viewport = renderingEngine?.getViewport(viewportId);
    if (!viewport) return;

    const actorEntry = viewport.getActor(volumeId);
    const actor = actorEntry?.actor;
    if (!actor || typeof actor.getProperty !== 'function') return;

    const cfun = vtkColorTransferFunction.newInstance();
    const preset = vtkColorMaps.getPresetByName(colormapName);
    if (!preset) return;

    cfun.applyColorMap(preset);
    cfun.updateRange();

    const property = actor.getProperty();
    property.setRGBTransferFunction(0, cfun);

    const range = viewport.getImageData()?.imageData.getPointData().getScalars().getRange();
    if (range) {
        cfun.setMappingRange(...range);
    }

    viewport.render();
};

export const handleToolClick = (toolName: string, mouseEvent: any) => {
    let clickedMouseButton = 1;
    console.log('handleToolClick', toolName, mouseEvent.button);

    switch (mouseEvent.button) {
        case 0:
            clickedMouseButton = 1;
            break;
        case 1:
            clickedMouseButton = 4;
            break;
        case 2:
            clickedMouseButton = 2;
            break;
        default:
            return;
    }

    switch (toolName) {
        case ANNOTATION_TOOLS['Window'].toolName:
            CornerstoneToolManager.setToolActive(ANNOTATION_TOOLS['Window'].toolName, clickedMouseButton);
            break;
        case ANNOTATION_TOOLS['Zoom'].toolName:
            CornerstoneToolManager.setToolActive(ANNOTATION_TOOLS['Zoom'].toolName, clickedMouseButton);
            break;
        case ANNOTATION_TOOLS['Pan'].toolName:
            CornerstoneToolManager.setToolActive(ANNOTATION_TOOLS['Pan'].toolName, clickedMouseButton);
            break;
        case 'Measurements':
            CornerstoneToolManager.setToolActive(ANNOTATION_TOOLS['Length'].toolName, clickedMouseButton);
            break;
        case 'Rotate':
            CornerstoneToolManager.setToolActive(
                ANNOTATION_TOOLS['Planar Rotate'].toolName,
                clickedMouseButton
            );
            break;
        case 'Magnify':
            CornerstoneToolManager.setToolActive(ANNOTATION_TOOLS['Magnify'].toolName, clickedMouseButton);
            break;
        case 'Scroll':
            CornerstoneToolManager.setToolActive(
                ANNOTATION_TOOLS['Stack Scroll'].toolName,
                clickedMouseButton
            );
            break;
        case 'Render':
            CornerstoneToolManager.setToolActive(
                ANNOTATION_TOOLS['TrackballRotate'].toolName,
                clickedMouseButton
            );
            break;
        case 'Cine':
            store.dispatch(viewerSliceActions.toggleCine());
            break;
        case 'Brush':
            CornerstoneToolManager.setToolActive(SEGMENTATION_TOOLS['Brush'].toolName, clickedMouseButton);
            break;
    }
};
