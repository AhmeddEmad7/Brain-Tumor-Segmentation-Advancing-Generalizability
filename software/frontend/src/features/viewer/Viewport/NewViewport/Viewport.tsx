import * as cornerstone from '@cornerstonejs/core';
import makeVolumeMetada from '@cornerstonejs/core/dist/esm/utilities/makeVolumeMetadata';
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
import { toggleMPRMode, toggleVolumeRendering } from '@features/viewer/ViewerTopBar/viewer-top-bar-actions'; // Import your toggleMPRMode function
import CornerstoneToolManager from '@/features/viewer/CornerstoneToolManager/CornerstoneToolManager';
import * as cornerstoneTools from '@cornerstonejs/tools';
// const { isCrosshairActive } = useSelector((store: IStore) => store.viewer);

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
        is3DActive,
        isCrosshairActive
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
        if (selectedSeriesInstanceUid && is3DActive) {
            getSeriesModality(currentStudyInstanceUid, selectedSeriesInstanceUid).then((modality) => {
                if (modality !== 'SEG') {
                    console.log('🔄 Switching to 2D due to series change...');
                    toggleVolumeRendering(true); // **Force switching to 2D**
                } else {
                    console.log('📦 Keeping 3D Mode for Segmentation');
                }
            });
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
    }, [isMPRActive, is3DActive, selectedViewportId, selectedSeriesInstanceUid]);

    useEffect(() => {
        if (selectedSeriesInstanceUid && isMPRActive) {
            // Trigger MPR mode whenever the series changes
            toggleMPRMode(renderingEngineId, selectedSeriesInstanceUid, currentStudyInstanceUid);
        }
    }, [selectedSeriesInstanceUid, renderingEngineId, currentStudyInstanceUid]);

    useEffect(() => {
        if (isMPRActive && isCrosshairActive) {
            const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
            if (!renderingEngine) return;

            const viewport = renderingEngine.getViewport(selectedViewportId);
            if (!viewport) return;

            CornerstoneToolManager.setToolActive(
                cornerstoneTools.CrosshairsTool.toolName,
                cornerstoneTools.Enums.MouseBindings.Primary
            );
        }
    }, [isMPRActive, isCrosshairActive]);

    useEffect(() => {
        const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
    
        const updateViewport = async () => {
            try {
                if (!selectedSeriesInstanceUid || !selectedViewportId || !renderingEngine) {
                    console.warn('⚠️ Missing required values to update viewport.');
                    return;
                }
    
                const viewport = renderingEngine.getViewport(selectedViewportId) as Types.IVolumeViewport;
    
                if (!viewport) {
                    console.error(`❌ Viewport ${selectedViewportId} not found.`);
                    return;
                }
    
                console.log('🔄 Updating viewport for series:', selectedSeriesInstanceUid);
    
                // ✅ **Check if the series is a segmentation (`SEG`)**
                const modality = await getSeriesModality(currentStudyInstanceUid, selectedSeriesInstanceUid);
                if (modality === 'SEG') {
                    console.log('🧩 Detected segmentation series, reading segmentation...');
                    await readSegmentation(selectedSeriesInstanceUid);
                    return;
                }
    
                // ✅ **Generate Volume ID**
                const volumeId = `cornerstoneStreamingImageVolume:${selectedSeriesInstanceUid}`;
                console.log('🆔 Volume ID:', volumeId);
    
                // ✅ **Retrieve Image IDs**
                const imageIds = await createImageIdsAndCacheMetaData({
                    StudyInstanceUID: currentStudyInstanceUid,
                    SeriesInstanceUID: selectedSeriesInstanceUid,
                    wadoRsRoot: wadoRsRoot
                });
    
                if (!imageIds || imageIds.length === 0) {
                    console.error('❌ No image IDs found for this series.');
                    return;
                }
    
                console.log(`📸 Found ${imageIds.length} image IDs.`);
    
                // ✅ **Ensure metadata exists before proceeding**
                const volumeMetada = cornerstone.utilities.makeVolumeMetadata(imageIds);
                console.log('📦 Volume metadata:', volumeMetada.PixelRepresentation);
                for (const imageId of imageIds) {
                    // Try to get the metadata; if it doesn't exist, create a default object.
                    let metadata = cornerstone.metaData.get('imagePixelModule', imageId);
                    console.log('📦 Metadata pixel:', metadata.pixelRepresentation);
                    if (!metadata) {
                      console.warn(`Metadata missing for imageId ${imageId}. Using default metadata.`);
                      metadata = {
                        pixelRepresentation: 1, // default value
                        BitsAllocated: 16,        // provide a fallback or an appropriate value
                        BitsStored: 16,
                        HighBit: 15,
                        PhotometricInterpretation: 'MONOCHROME2',
                        SamplesPerPixel: 1,
                      };
                    } else {
                      // Map the PascalCase keys to the lowercase keys expected by the volume loader
                      // Here, you can override pixelRepresentation if needed.
                      metadata.pixelRepresentation = 1; // or any value you choose
                      metadata.bitsAllocated = metadata.BitsAllocated;
                      metadata.bitsStored = metadata.BitsStored;
                      metadata.highBit = metadata.HighBit;
                      metadata.photometricInterpretation = metadata.PhotometricInterpretation;
                      metadata.samplesPerPixel = metadata.SamplesPerPixel;
                    }
                  }
                  
                // ✅ **Create & Load Volume**
                console.log('🔄 Creating & caching volume...');
                const volume = await cornerstone.volumeLoader.createAndCacheVolume(volumeId, { imageIds });
                await volume.load();
    
                // ✅ **Set Volume in Viewport**
                console.log('📡 Setting volume in viewport...');
                await viewport.setVolumes([{ volumeId }], true);
            
                // ✅ **Set Orientation**
                const direction = viewport.getImageData()?.imageData.getDirection() as number[];
                const orientation = DicomUtil.detectImageOrientation(
                    direction ? direction.slice(0, 6) : [1, 0, 0, 0, 1, 0]
                );
    
                if (is3DActive) {
                    console.log('🖥️ Setting 3D mode...');
                    viewport.resetCamera();
                } else {
                    console.log('🖼️ Setting 2D orientation:', orientation);
                    viewport.setOrientation(orientation);
                }
    
                // ✅ **Render Viewport**
                viewport.render();
    
                // ✅ **Update State**
                setThisViewport(viewport);
                setThisViewportImageIds(viewport.getImageIds());
    
                dispatch(viewerSliceActions.removeClickedSeries());
                dispatch(viewerSliceActions.setClickedSeries(selectedSeriesInstanceUid));
    
                console.log('✅ Viewport updated successfully!');
            } catch (error) {
                console.error('❌ Error setting viewport:', error);
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
