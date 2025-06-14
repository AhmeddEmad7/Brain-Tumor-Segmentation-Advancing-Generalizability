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
    set3DActive: (state: IStoreViewerSlice, action: PayloadAction<boolean>) => {
        state.is3DActive = action.payload;
    },
    setAxialPrimaryLayoutActive: (state: IStoreViewerSlice, action: PayloadAction<boolean>) => {
        state.isAxialPrimaryLayoutActive = action.payload;
    },
    set3DPrimaryLayoutActive: (state: IStoreViewerSlice, action: PayloadAction<boolean>) => {
        state.is3DPrimaryLayoutActive = action.payload;
    },
    toggleMPRLayout: (
        state: IStoreViewerSlice,
        action: PayloadAction<{ orientations: OrientationAxis[] }>
    ) => {
        state.layout = {
            numRows: 1,
            numCols: 3
        };
    },
    resetViewerLayout: (state: IStoreViewerSlice) => {
        state.layout = {
            numRows: 1,
            numCols: 1
        };
    },
    setColorBarVisible: (state: IStoreViewerSlice, action: PayloadAction<boolean>) => {
        state.isColorBarVisible = action.payload;
    }
};

export default viewerUiReducer;
