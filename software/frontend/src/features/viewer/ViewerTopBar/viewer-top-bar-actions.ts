import store from '@/redux/store.ts';
import { viewerSliceActions } from '@features/viewer/viewer-slice.ts';
import {
    CornerstoneToolManager,
    ANNOTATION_TOOLS,
    SEGMENTATION_TOOLS
} from '@/features/viewer/CornerstoneToolManager/';
import { OrientationAxis } from '@cornerstonejs/core/src/enums';
import { Types } from '@cornerstonejs/core';
import { useSelector, useDispatch } from 'react-redux';
import { IStore } from '@/models';
import * as cornerstone from '@cornerstonejs/core';
import { createImageIdsAndCacheMetaData } from '@utilities/helpers/index';


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

    const orientations: OrientationAxis[] = [
        OrientationAxis.AXIAL,
        OrientationAxis.SAGITTAL,
        OrientationAxis.CORONAL
    ];

    const volumeId = `cornerstoneStreamingImageVolume:${selectedSeriesInstanceUid}`;

    // Load the volume and cache metadata
    const imageIds = await createImageIdsAndCacheMetaData({
        StudyInstanceUID: currentStudyInstanceUid,
        SeriesInstanceUID: selectedSeriesInstanceUid,
        wadoRsRoot: wadoRsRoot
    });

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
    const viewports = [0, 1, 2].map((index) =>
        renderingEngine.getViewport(`viewport-${index}`) as Types.IVolumeViewport | undefined
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
    store.dispatch(viewerSliceActions.removeClickedSeries());

};


export const toggleViewportOverlayShown = () => {
    store.dispatch(viewerSliceActions.toggleInfoOnViewports());
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
        case 'Cine':
            store.dispatch(viewerSliceActions.toggleCine());
            break;
        case 'Brush':
            CornerstoneToolManager.setToolActive(SEGMENTATION_TOOLS['Brush'].toolName, clickedMouseButton);
            break;
    }
};
