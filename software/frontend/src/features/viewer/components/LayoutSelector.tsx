import { useState } from 'react';
import PropTypes from 'prop-types';
import { useDispatch } from 'react-redux';
import { TAppDispatch } from '@/redux/store.ts';
import { viewerSliceActions } from '@features/viewer/viewer-slice.ts';
import { ILayout } from '@/models';
import { GridView as LayoutIcon } from '@mui/icons-material';
import * as cornerstone from '@cornerstonejs/core';
import { Types } from '@cornerstonejs/core';
import store from '@/redux/store.ts';
import { toggleVolumeRendering } from '@features/viewer/ViewerTopBar/viewer-top-bar-actions.ts';

interface LayoutSelectorProps {
    columns?: number;
    rows?: number;
    is3DPrimaryLayoutActive?: boolean;
    isAxialPrimaryLayoutActive?: boolean;
}

function LayoutSelector({
    columns = 4,
    rows = 4,
    is3DPrimaryLayoutActive = false,
    isAxialPrimaryLayoutActive = false
}: LayoutSelectorProps) {
    const [hoveredIndex, setHoveredIndex] = useState<number>();
    const dispatch = useDispatch<TAppDispatch>();

    const hoverX = hoveredIndex! % columns;
    const hoverY = Math.floor(hoveredIndex! / columns);
    const isHovered = (index: number) => {
        const x = index % columns;
        const y = Math.floor(index / columns);
        return x <= hoverX && y <= hoverY;
    };

    const handleOnSelect = ({ numRows, numCols }: ILayout) => {
        const layout: ILayout = { numRows, numCols };

        dispatch(viewerSliceActions.removeClickedSeries());

        // Special 3D Primary Layout logic
        if (is3DPrimaryLayoutActive && numRows === 2 && numCols === 2) {
            dispatch(viewerSliceActions.setMPRActive(true));
            dispatch(viewerSliceActions.set3DActive(true));
            dispatch(viewerSliceActions.set3DPrimaryLayoutActive(true));
        } else if (isAxialPrimaryLayoutActive && numRows === 3 && numCols === 2) {
            dispatch(viewerSliceActions.setMPRActive(true));
            dispatch(viewerSliceActions.setAxialPrimaryLayoutActive(true));
        } else {
            dispatch(viewerSliceActions.setMPRActive(false));
            dispatch(viewerSliceActions.set3DActive(false));
            dispatch(viewerSliceActions.set3DPrimaryLayoutActive(false));
            dispatch(viewerSliceActions.setAxialPrimaryLayoutActive(false));
        }

        dispatch(viewerSliceActions.changeViewerLayout(layout));
    };

    const handle3DPrimaryClick = async () => {
        const { renderingEngineId, selectedSeriesInstanceUid } = store.getState().viewer;

        // Only activate if not already active
        if (store.getState().viewer.is3DPrimaryLayoutActive) {
            return;
        }

        // Activate 3D Primary layout
        store.dispatch(viewerSliceActions.set3DPrimaryLayoutActive(true));
        store.dispatch(viewerSliceActions.setAxialPrimaryLayoutActive(false));

        // Set 3x2 layout and wait for it to be applied
        store.dispatch(
            viewerSliceActions.changeViewerLayout({
                numRows: 3,
                numCols: 2,
                is3DPrimaryLayoutActive: true
            })
        );
        store.dispatch(viewerSliceActions.set3DPrimaryLayoutActive(true));
        await toggleVolumeRendering();
        // Wait a bit for the layout to be applied
        await new Promise((resolve) => setTimeout(resolve, 100));

        const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
        if (!renderingEngine) {
            console.error('Rendering engine not found');
            return;
        }

        const volumeId = `cornerstoneStreamingImageVolume:${selectedSeriesInstanceUid}`;

        // Set up MPR viewports (viewport-1, viewport-2, viewport-3)
        const mprViewports = [1, 2, 3].map((index) => {
            const viewport = renderingEngine.getViewport(`viewport-${index}`);
            if (!viewport) {
                console.warn(`Viewport viewport-${index} not found`);
                return undefined;
            }
            return viewport as Types.IVolumeViewport;
        });

        const orientations = [
            cornerstone.Enums.OrientationAxis.AXIAL,
            cornerstone.Enums.OrientationAxis.CORONAL,
            cornerstone.Enums.OrientationAxis.SAGITTAL
        ];

        // Configure MPR viewports
        mprViewports.forEach((viewport, index) => {
            if (viewport) {
                try {
                    viewport.setVolumes([{ volumeId }], true);
                    viewport.setOrientation(orientations[index]);
                    viewport.render();
                } catch (error) {
                    console.error('Error setting up MPR viewport:', error);
                }
            }
        });
    };

    const handleAxialPrimaryClick = async () => {
        const { renderingEngineId, selectedSeriesInstanceUid } = store.getState().viewer;

        // Only activate if not already active
        if (store.getState().viewer.isAxialPrimaryLayoutActive) {
            return;
        }
        await toggleVolumeRendering(true);

        // Activate Axial Primary layout
        store.dispatch(viewerSliceActions.setAxialPrimaryLayoutActive(true));
        store.dispatch(viewerSliceActions.set3DPrimaryLayoutActive(false));

        // Set 3x2 layout and wait for it to be applied
        store.dispatch(
            viewerSliceActions.changeViewerLayout({
                numRows: 2,
                numCols: 2,
                isAxialPrimaryLayoutActive: true
            })
        );
        // Wait a bit for the layout to be applied
        await new Promise((resolve) => setTimeout(resolve, 100));

        const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
        if (!renderingEngine) {
            console.error('Rendering engine not found');
            return;
        }

        const volumeId = `cornerstoneStreamingImageVolume:${selectedSeriesInstanceUid}`;

        // Set up MPR viewports (viewport-1, viewport-2, viewport-3)
        const mprViewports = [0, 1, 2].map((index) => {
            const viewport = renderingEngine.getViewport(`viewport-${index}`);
            if (!viewport) {
                console.warn(`Viewport viewport-${index} not found`);
                return undefined;
            }
            return viewport as Types.IVolumeViewport;
        });

        const orientations = [
            cornerstone.Enums.OrientationAxis.AXIAL,
            cornerstone.Enums.OrientationAxis.CORONAL,
            cornerstone.Enums.OrientationAxis.SAGITTAL
        ];

        // Configure MPR viewports
        mprViewports.forEach((viewport, index) => {
            if (viewport) {
                try {
                    viewport.setVolumes([{ volumeId }], true);
                    viewport.setOrientation(orientations[index]);
                    viewport.render();
                } catch (error) {
                    console.error('Error setting up MPR viewport:', error);
                }
            }
        });
    };

    const gridSize = '20px ';
    const isSpecial3DLayout = is3DPrimaryLayoutActive && rows === 2 && columns === 2;
    const isSpecialAxialLayout = isAxialPrimaryLayoutActive && rows === 3 && columns === 2;

    const gridStyle = {
        display: 'grid',
        gridTemplateColumns: gridSize.repeat(columns),
        gridTemplateRows: gridSize.repeat(rows)
    };

    return (
        <div className="flex gap-4">
            <div className="flex flex-col gap-2 w-33">
                <button
                    onClick={handle3DPrimaryClick}
                    className={`flex items-center text-sm text-white rounded hover:bg-gray-700 px-2 py-1 w-full ${
                        store.getState().viewer.is3DPrimaryLayoutActive ? 'bg-gray-700' : ''
                    }`}
                >
                    <LayoutIcon className="w-4 mr-1" />
                    <span>3D Primary</span>
                </button>
                <button
                    onClick={handleAxialPrimaryClick}
                    className={`flex items-center mt-2 text-sm text-white rounded hover:bg-gray-700 px-2 py-1 w-full ${
                        store.getState().viewer.isAxialPrimaryLayoutActive ? 'bg-gray-700' : ''
                    }`}
                >
                    <LayoutIcon className="w-5 mr-1" />
                    <span>Axial Primary</span>
                </button>
            </div>
            <div style={gridStyle}>
                {Array.from({ length: rows * columns }, (_, index) => {
                    const x = index % columns;
                    const y = Math.floor(index / columns);
                    const areaName = isSpecial3DLayout
                        ? `viewport${index}`
                        : isSpecialAxialLayout
                          ? `viewport${index}`
                          : undefined;

                    return (
                        <div
                            key={index}
                            style={{
                                border: '1px solid white',
                                backgroundColor: isHovered(index) ? '#10a9e2' : '',
                                ...(areaName ? { gridArea: areaName } : {})
                            }}
                            data-cy={`Layout-${x}-${y}`}
                            className="cursor-pointer"
                            onClick={() => {
                                handleOnSelect({ numRows: y + 1, numCols: x + 1 });
                            }}
                            onMouseEnter={() => setHoveredIndex(index)}
                            onMouseLeave={() => setHoveredIndex(-1)}
                        />
                    );
                })}
            </div>
        </div>
    );
}

LayoutSelector.defaultProps = {
    columns: 3,
    rows: 3,
    is3DPrimaryLayoutActive: false,
    isAxialPrimaryLayoutActive: false
};

LayoutSelector.propTypes = {
    columns: PropTypes.number.isRequired,
    rows: PropTypes.number.isRequired,
    is3DPrimaryLayoutActive: PropTypes.bool,
    isAxialPrimaryLayoutActive: PropTypes.bool
};

export default LayoutSelector;
