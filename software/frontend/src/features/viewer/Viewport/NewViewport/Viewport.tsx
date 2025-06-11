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
import { toggleMPRMode, toggleVolumeRendering } from '@features/viewer/ViewerTopBar/viewer-top-bar-actions'; // Import your toggleMPRMode function
import CornerstoneToolManager from '@/features/viewer/CornerstoneToolManager/CornerstoneToolManager';
import * as cornerstoneTools from '@cornerstonejs/tools';
import { cornerstoneNiftiImageVolumeLoader } from '@cornerstonejs/nifti-volume-loader';
import { Volume } from 'lucide-react';
import { OrientationMenu } from '@features/viewer/components/OrientationMenu';

// import * as cornerstone3 from '@cornerstonejs/core/src/loaders/';
// const { isCrosshairActive } = useSelector((store: IStore) => store.viewer);

const wadoRsRoot = import.meta.env.VITE_ORTRHANC_PROXY_URL;
const NIFTI_DOMAIN = import.meta.env.VITE_NIFTI_DOMAIN;
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
    const segmentationVolume = useSelector(
        (s: IStore) => s.viewer.segmentations.find((seg) => seg.isActive)?.segmentationVolume
    );
    console.log('segmentationVolume', segmentationVolume);
    // handleViewportClick is a function that takes an id and dispatches an action to the viewerSlice
    const handleViewportClick = (id: string) => {
        dispatch(viewerSliceActions.setSelectedViewport(id));
        if (onClick) {
            onClick(id);
        }
    };
    //     const addSegmentationMesh = async (
    //         viewport: Types.IVolumeViewport,
    //         segmentationVolumeId: string
    //       ) => {
    //         if (hasMeshBeenAdded.current) return;
    //         hasMeshBeenAdded.current = true;

    //         const segVol = cornerstone.cache.getVolume(segmentationVolumeId);
    //         console.log('segVol', segVol);
    //         if (!segVol) {
    //           console.warn('Segmentation volume not found.');
    //           return;
    //         }

    //         // const segImageData = segVol.imageData!;
    //         const segImageData = segVol.imageData!;
    //         const buffer = segImageData.getPointData().getScalars().getData();
    //         const dims = segImageData.getDimensions();
    //         const spacing = segImageData.getSpacing();
    //         const origin = segImageData.getOrigin();

    //         const vtkSeg = vtkImageData.newInstance({ dimensions: dims, spacing, origin });
    //         vtkSeg.getPointData().setScalars(
    //           vtkDataArray.newInstance({ values: buffer, numberOfComponents: 1 })
    //         );

    //         const mc = vtkImageMarchingCubes.newInstance({
    //           contourValue: 0.5,
    //           computeNormals: true,
    //           mergePoints: true,
    //         });
    //         mc.setInputData(vtkSeg);
    //         mc.update();
    //         const polydata = mc.getOutputData();

    //         const mapper = vtkMapper.newInstance();
    //         mapper.setInputData(polydata);

    //         const actor = vtkActor.newInstance();
    //         actor.setMapper(mapper);
    //         actor.getProperty().setColor(1, 0, 0);
    //         console.log('actorsasdsdasdasdasd');
    //         actor.getProperty().setOpacity(0.5);
    //         // console.log('📦 Segmentation buffer min/max:', Math.min(...buffer), Math.max(...buffer));
    //         const min = Array.from(buffer).reduce((a, b) => Math.min(a, b), Infinity);
    //         const max = Array.from(buffer).reduce((a, b) => Math.max(a, b), -Infinity);
    //         console.log('📦 Segmentation buffer min/max:', min, max);
    //         console.log('🧠 segmentationVolume ID:', segmentationVolume);

    //         vtkMatrixBuilder
    //   .buildFromDegree()
    //   .translate(50, 10, -10) // 👉 try (100, 0, 0), (-100, 0, 0), (0, 100, 0) etc. to shift position
    //   .apply(actor.getUserMatrix());
    //         viewport.addActor({ uid: 'SegMesh', actor });
    //         viewport.resetCamera();
    //         viewport.render();
    //       };
    //     const hasMeshBeenAdded = useRef(false);
    useEffect(() => {
        if (!thisViewport || !thisViewport.getImageIds) {
            console.warn('🛑 Viewport not ready for overlay:', thisViewport);
        }
    }, [thisViewport]);
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
        const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);

        const updateViewportNifti = async () => {
            try {
                if (!selectedViewportId || !renderingEngine) {
                    console.warn('⚠ Missing required values to update viewport.');
                    return;
                }

                const viewport = renderingEngine.getViewport(selectedViewportId) as Types.IVolumeViewport;

                console.log('currentStudyInstanceUid', currentStudyInstanceUid);

                if (!viewport) {
                    console.error(`❌ Viewport ${selectedViewportId} not found.`);
                    return;
                }
                if (currentStudyInstanceUid.endsWith('.nii') || currentStudyInstanceUid.endsWith('.gz')) {
                    console.log('File name ends with .nii or .gz');
                } else {
                    console.log('File NOt name ends with .nii or .gz');
                    return;
                }

                const niftiURL = `${NIFTI_DOMAIN}/${currentStudyInstanceUid}`;
                // const niftiURL = 'nifti/00000057_brain_flair.nii';
                console.log('niftiURL', niftiURL);
                const volumeId = 'nifti:' + niftiURL;
                console.log('🆔 Volume ID:', volumeId);

                cornerstone.volumeLoader.registerVolumeLoader('nifti', cornerstoneNiftiImageVolumeLoader);

                const volume = await cornerstone.volumeLoader.createAndCacheVolume(volumeId);

                cornerstone.setVolumesForViewports(renderingEngine, [{ volumeId }], [selectedViewportId]);

                console.log('📡 Setting volume in viewport...');
                await viewport.setVolumes([{ volumeId }], true);
                // or any supported VTK preset

                viewport.resetCamera();
                viewport.render();

                viewport.render();

                setThisViewport(viewport);
                setThisViewportImageIds(viewport.getImageIds());
                dispatch(viewerSliceActions.removeClickedSeries());

                console.log('✅ Viewport updated successfully!');
            } catch (error) {
                console.error('❌ Error setting viewport:', error);
            }
        };
        // add if Nifti here
        updateViewportNifti();
    }, [selectedViewportId]);
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

                if (currentStudyInstanceUid.endsWith('.nii') || currentStudyInstanceUid.endsWith('.gz')) {
                    console.log('File name ends with .nii or .gz');
                } else {
                    console.log('File NOt name ends with .nii or .gz');
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

                // ✅ **Create & Load Volume**
                console.log('🔄 Creating & caching volume...');

                //   const volume = await cornerstone.volumeLoader.createAndCacheVolume(volumeId);
                const volume = await cornerstone.volumeLoader.createAndCacheVolume(volumeId, { imageIds });
                await volume.load();

                // ✅ **Set Volume in Viewport**
                console.log('📡 Setting volume in viewport...');
                await viewport.setVolumes([{ volumeId }], true);
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

    // Add new useEffect to monitor viewport properties
    useEffect(() => {
        if (!thisViewport) return;

        // Get initial properties
        const initialProps = thisViewport.getProperties();

        // Create a function to check for property changes
        const checkProperties = () => {
            const currentProps = thisViewport.getProperties();

            // Compare properties that should trigger a re-render
            if (
                JSON.stringify(currentProps.colormap) !== JSON.stringify(initialProps.colormap) ||
                JSON.stringify(currentProps.voiRange) !== JSON.stringify(initialProps.voiRange) ||
                currentProps.invert !== initialProps.invert
            ) {
                thisViewport.render();
                // Update initial props for next comparison
                Object.assign(initialProps, currentProps);
            }
        };

        // Set up an interval to check for changes
        const intervalId = setInterval(checkProperties, 100); // Check every 100ms

        // Cleanup interval on unmount
        return () => {
            clearInterval(intervalId);
        };
    }, [thisViewport]); // Only re-run if thisViewport changes

    return (
        <div className={'flex-col'}>
            <div
                id={id}
                ref={viewportRef}
                onClick={() => handleViewportClick(id)}
                className={`${hasCinePlayer ? `${cineHeight[0]}` : 'h-full'} w-full relative bg-black ${selectedViewportId === id ? 'border-2 border-x-blue-200' : ''}`}
            >
                  {selectedViewportId === id && (
                <div className="absolute top-1 left-1 z-50">
                <OrientationMenu viewportId={id} />
                </div>
            )}

                <ViewportOverlay
                    viewport={thisViewport && thisViewport.getImageIds.length ? thisViewport : null}
                    currentImageId={currentImageId}
                />
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
