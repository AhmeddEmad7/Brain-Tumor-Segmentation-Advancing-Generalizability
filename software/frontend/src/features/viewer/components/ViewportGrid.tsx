import React from 'react';
import { useSelector } from 'react-redux';
import Viewport from './Viewport';
import getGridClassnames from '../Viewport/ViewportGrid/gridClassnames';

const ViewportGrid: React.FC = () => {
    const store = useSelector((state) => state.viewer);

    const { numRows, numCols, is3DPrimaryLayoutActive, isAxialPrimaryLayoutActive } = store.getState().viewer;

    // Get grid classnames based on layout
    const gridClassName = getGridClassnames(
        numRows,
        numCols,
        is3DPrimaryLayoutActive,
        isAxialPrimaryLayoutActive
    );

    // Create viewport IDs based on layout
    const viewportIds = [];
    for (let i = 0; i < numRows * numCols; i++) {
        viewportIds.push(`viewport-${i}`);
    }

    // Handle special layouts
    if (is3DPrimaryLayoutActive || isAxialPrimaryLayoutActive) {
        return (
            <div className={`${gridClassName} w-full h-full gap-1`}>
                <div className="w-full h-full">
                    <Viewport viewportId="viewport-0" />
                </div>
                <div className="w-full h-full">
                    <Viewport viewportId="viewport-1" />
                </div>
                <div className="w-full h-full">
                    <Viewport viewportId="viewport-2" />
                </div>
                <div className="w-full h-full">
                    <Viewport viewportId="viewport-3" />
                </div>
            </div>
        );
    }

    // Regular grid layout
    return (
        <div className={`${gridClassName} w-full h-full gap-1`}>
            {viewportIds.map((viewportId) => (
                <div key={viewportId} className="w-full h-full">
                    <Viewport viewportId={viewportId} />
                </div>
            ))}
        </div>
    );
};

export default ViewportGrid; 