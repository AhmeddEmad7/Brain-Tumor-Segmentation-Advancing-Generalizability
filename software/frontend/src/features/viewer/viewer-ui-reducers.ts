import { PayloadAction } from '@reduxjs/toolkit';
import { IStoreViewerSlice, ILayout } from '@/models';
import { OrientationAxis } from '@cornerstonejs/core/src/enums';

const viewerUiReducer = {
    changeViewerLayout: (state: IStoreViewerSlice, action: PayloadAction<ILayout>) => {
        state.layout = action.payload;
    },
    toggleFullScreen: (state: IStoreViewerSlice) => {
        state.isFullScreen = !state.isFullScreen;
    },
    toggleStudiesPanel: (state: IStoreViewerSlice) => {
        state.isStudiesPanelOpen = !state.isStudiesPanelOpen;
    },
    toggleLeftPanel: (state: IStoreViewerSlice) => {
        state.isRightPanelOpen = !state.isRightPanelOpen;
    },
    toggleInfoOnViewports: (state: IStoreViewerSlice) => {
        state.isInfoOnViewportsShown = !state.isInfoOnViewportsShown;
    },
    setMPRActive: (state: IStoreViewerSlice, action: PayloadAction<boolean>) => {
        state.isMPRActive = action.payload;
    },
    toggleMPRLayout: (
        state: IStoreViewerSlice,
        action: PayloadAction<{ orientations: OrientationAxis[] }>
    ) => {
        state.layout = {
            numRows: 1,
            numCols: 3
        };

    }
};

export default viewerUiReducer;
