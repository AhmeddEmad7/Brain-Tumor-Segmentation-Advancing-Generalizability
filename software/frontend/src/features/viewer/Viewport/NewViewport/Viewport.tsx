import * as cornerstone from '@cornerstonejs/core';
import { useEffect, useRef, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { viewerSliceActions } from '@features/viewer/viewer-slice.ts';
import { Types } from '@cornerstonejs/core';
import { DicomUtil } from '@/utilities';
import { IStore } from '@/models';
import useResizeObserver from '@hooks/useResizeObserver.tsx';
import ViewportOverlay from '@features/viewer/Viewport/ViewportOverlay/ViewportOverlay.tsx';
import CinePlayer from '@features/viewer/Viewport/CinePlayer/CinePlayer.tsx';
import { detectCineHeight } from '@features/viewer/Viewport/CinePlayer/detectCineHeight';
import { createImageIdsAndCacheMetaData } from '@utilities/helpers/index';
import { readSegmentation } from '../../CornerstoneToolManager/segmentationMethods';
import { getSeriesModality } from '@features/viewer/viewer-viewport-reducers';
import { toggleMPRMode ,toggleVolumeRendering } from '@features/viewer/ViewerTopBar/viewer-top-bar-actions'; // Import your toggleMPRMode function
import CornerstoneToolManager from '@/features/viewer/CornerstoneToolManager/CornerstoneToolManager';
import * as cornerstoneTools from '@cornerstonejs/tools';

const wadoRsRoot = import.meta.env.VITE_ORTRHANC_PROXY_URL;

type TViewportProps = {
    onClick?: (idx: string) => void;
    selectedViewportId?: number | string | null;
    id: string;
    ref?: HTMLDivElement | null;
    vNeighbours?: number;
    hNeighbours?: number;
};

const Viewport = ({ onClick, id, vNeighbours }: TViewportProps) => {
    const [currentImageId, setCurrentImageId] = useState<string>('');
    const [thisViewport, setThisViewport] = useState<Types.IVolumeViewport | null>(null);
    const [thisViewportImageIds, setThisViewportImageIds] = useState<string[]>([]);
    const [hasCinePlayer, setHasCinePlayer] = useState<boolean>(false);

    const viewportRef = useRef<HTMLDivElement>(null);
    const cineRef = useRef<HTMLDivElement>(null);
    const {
        selectedSeriesInstanceUid,
        selectedViewportId,
        renderingEngineId,
        viewportsWithCinePlayer,
        currentStudyInstanceUid,
        isMPRActive,
        is3DActive
    } = useSelector((store: IStore) => store.viewer);

    const dispatch = useDispatch();

    // handleViewportClick is a function that takes an id and dispatches an action to the viewerSlice
    const handleViewportClick = (id: string) => {
        dispatch(viewerSliceActions.setSelectedViewport(id));
        if (onClick) {
            onClick(id);
        }
    }; 
    useEffect(() => {
        if (selectedSeriesInstanceUid&&is3DActive) {
            console.log('🔄 Switching to 2D due to series change...');
            toggleVolumeRendering(true); // **Force switching to 2D**
        }
    }, [selectedSeriesInstanceUid]);
    
        // 🔹 Ensure correct tools are applied when mode changes
        useEffect(() => {
            const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
            if (!renderingEngine) return;
    
            const viewport = renderingEngine.getViewport(selectedViewportId);
            if (!viewport) return;
    
            let toolGroupId = 'CornerstoneTools2D';
             if (is3DActive) {
                toolGroupId = 'CornerstoneTools3D';
            }
            // 🟢 **Ensure the tool group exists**
            const toolGroup = cornerstoneTools.ToolGroupManager.getToolGroup(toolGroupId);
            if (!toolGroup) {
                console.error(`❌ Tool Group ${toolGroupId} not found.`);
                return;
            }
    
            // 🟢 **Add the viewport to the correct tool group**
            toolGroup.addViewport(selectedViewportId, renderingEngineId);
            // 🟢 **Activate the correct tool group**
            console.log(`✅ Applying tool group: ${toolGroupId}`);
            CornerstoneToolManager.setCurrentToolGroupId(toolGroupId);
    
            // 🔄 **Ensure viewport is refreshed after tool change**
            viewport.render();
        }, [isMPRActive, is3DActive, selectedViewportId,selectedSeriesInstanceUid]);
    
    useEffect(() => {

        if (selectedSeriesInstanceUid&&isMPRActive) {
            // Trigger MPR mode whenever the series changes
            toggleMPRMode(renderingEngineId, selectedSeriesInstanceUid, currentStudyInstanceUid);
        }
    }, [selectedSeriesInstanceUid, renderingEngineId, currentStudyInstanceUid]);

    useEffect(() => {
        const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);

        const updateViewport = async () => {
            try {
                if (selectedViewportId === id && selectedSeriesInstanceUid && renderingEngine) {
                    const viewport: Types.IVolumeViewport = renderingEngine!.getViewport(
                        selectedViewportId
                    ) as Types.IVolumeViewport;

                    if (
                        (await getSeriesModality(currentStudyInstanceUid, selectedSeriesInstanceUid)) ===
                        'SEG'
                    ) {
                        readSegmentation(selectedSeriesInstanceUid);
                        return;
                    }

                    const volumeId = `cornerstoneStreamingImageVolume:${selectedSeriesInstanceUid}`;
                    console.log('volumeId inside update', volumeId);
                    const imageIds = await createImageIdsAndCacheMetaData({
                        StudyInstanceUID: currentStudyInstanceUid,
                        SeriesInstanceUID: selectedSeriesInstanceUid,
                        wadoRsRoot: wadoRsRoot
                    });

                    const volume = await cornerstone.volumeLoader.createAndCacheVolume(volumeId, {
                        imageIds
                    });

                    await volume.load();

                    await viewport.setVolumes([{ volumeId }], true);

                    const direction = viewport.getImageData()?.imageData.getDirection() as number[];
                    const orientation = DicomUtil.detectImageOrientation(
                        direction ? direction.slice(0, 6) : [1, 0, 0, 0, 1, 0]
                    );
                    if (is3DActive) {   
                        // Set the orientation of the viewport
                        // Render the viewport
                        viewport.resetCamera();
                    }
                    else {
                        viewport.setOrientation(orientation);
                    }
                    
                    viewport.render();
                    // Set the current viewport and imageIds
                    setThisViewport(viewport);
                    setThisViewportImageIds(viewport.getImageIds());
                    // dispatch(viewerSliceActions.setClickedSeries(selectedSeriesInstanceUid));
                    // dispatch(viewerSliceActions.removeClickedSeries());
                }
            } catch (error) {
                console.error('Error setting viewport', error);
            }
        };

        updateViewport();
    }, [selectedSeriesInstanceUid, selectedViewportId]);

    useEffect(() => {
        if (thisViewport) {
            setCurrentImageId(thisViewportImageIds[thisViewport.getCurrentImageIdIndex()]);
        }
    }, [thisViewportImageIds]);

    // add event listener for slice scrolling
    useEffect(() => {
        const handleSliceScroll = () => {
            if (thisViewport) {
                setCurrentImageId(thisViewportImageIds[thisViewport.getCurrentImageIdIndex()]);
            }
        };

        viewportRef.current?.addEventListener('wheel', handleSliceScroll);

        return () => {
            viewportRef.current?.removeEventListener('wheel', handleSliceScroll);
        };
    }, []);

    useEffect(() => {
        if (viewportsWithCinePlayer.includes(id)) {
            setHasCinePlayer(true);
        } else {
            setHasCinePlayer(false);
        }
    }, [viewportsWithCinePlayer]);

    const handleResize = (_: number, __: number) => {
        const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
        const viewport = renderingEngine?.getViewport(id) as Types.IVolumeViewport;

        if (viewport) {
            viewport.resetCamera(true, true, true, true);
            renderingEngine?.resize(true, false);
        }
    };

    useResizeObserver(viewportRef, handleResize);

    const cineHeight = detectCineHeight(vNeighbours || 1);

    return (
        <div className={'flex-col'}>
            <div
                id={id}
                ref={viewportRef}
                onClick={() => handleViewportClick(id)}
                className={`${hasCinePlayer ? `${cineHeight[0]}` : 'h-full'} w-full relative bg-black ${selectedViewportId === id ? 'border-2 border-AAPrimary' : ''}`}
            >
                <ViewportOverlay viewport={thisViewport} currentImageId={currentImageId} />
            </div>
            {hasCinePlayer && (
                <div ref={cineRef} className={`${cineHeight[1]}`}>
                    <CinePlayer viewportElementRef={viewportRef} />
                </div>
            )}
        </div>
    );
};

export default Viewport;
