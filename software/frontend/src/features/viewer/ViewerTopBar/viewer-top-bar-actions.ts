import store from '@/redux/store.ts';
import { viewerSliceActions } from '@features/viewer/viewer-slice.ts';
import {
    CornerstoneToolManager,
    ANNOTATION_TOOLS,
    SEGMENTATION_TOOLS
} from '@/features/viewer/CornerstoneToolManager/';
import { getRenderingEngine, Enums } from '@cornerstonejs/core'; // New edits

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

export const toggleVolumeRendering = async () => {
    const state = store.getState();
    const { selectedViewportId, renderingEngineId } = state.viewer;
    const renderingEngine = getRenderingEngine(renderingEngineId);
    const viewport = renderingEngine.getViewport(selectedViewportId);

    console.log('Current viewport properties:', viewport.getProperties());

    if (viewport.type === Enums.ViewportType.VOLUME_3D) {
        viewport.setProperties({ type: Enums.ViewportType.ORTHOGRAPHIC });
        console.log('Switching to 2D');
    } else {
        viewport.setProperties({ type: Enums.ViewportType.VOLUME_3D });
        console.log('Switching to 3D');
    }

    viewport.render();
    console.log('Updated viewport properties:', viewport.getProperties());
    console.log('Viewport rendered');

    // Re-initialize the viewport to ensure the changes take effect
    renderingEngine.disableElement(selectedViewportId);
    renderingEngine.enableElement({
        viewportId: selectedViewportId,
        type: viewport.type,
        element: viewport.element,
    });

    // Reload the image or volume
    const imageIds = viewport.getImageIds(); // Ensure this method retrieves image IDs correctly
    if (viewport.type === Enums.ViewportType.ORTHOGRAPHIC) {
        if (viewport.setImageIds) {
            // Use setImageIds for 2D viewports
            await viewport.setImageIds(imageIds);
        } else {
            console.error('setImageIds method is not available on the viewport object.');
        }
    } else if (viewport.type === Enums.ViewportType.VOLUME_3D) {
        if (viewport.setVolumes) {
            // Use setVolumes for 3D volumes
            await viewport.setVolumes([{ volumeId: imageIds[0] }]);
        } else {
            console.error('setVolumes method is not available on the viewport object.');
        }
    }

    // Avoid re-initializing tools that have already been added
    if (!CornerstoneToolManager.isToolInitialized('Angle')) {
        CornerstoneToolManager.initCornerstoneAnnotationTool();
    }
    if (!CornerstoneToolManager.isToolInitialized('Brush')) {
        CornerstoneToolManager.initCornerstoneSegmentationTool();
    }

    viewport.render();
    console.log('Viewport re-initialized, image reloaded, and tools re-initialized');
};

