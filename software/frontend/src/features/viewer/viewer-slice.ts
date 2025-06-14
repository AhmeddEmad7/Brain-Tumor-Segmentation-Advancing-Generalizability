import { createSlice } from '@reduxjs/toolkit';
import { IStoreViewerSlice } from '@models/store.ts';
import viewerUiReducer from '@features/viewer/viewer-ui-reducers.ts';
import viewerViewportReducer from '@features/viewer/viewer-viewport-reducers.ts';
import viewerAnnotationReducer from './viewer-annotation-reducers';
import viewerSegmentationReducer from './viewer-segmentation-reducers';

const initialState: IStoreViewerSlice = {
    // ui
    isFullScreen: false,
    isMPRActive: false, // Default state for MPR
    isCrosshairActive: false, // Default state for Crosshair
    is3DActive: false, // Default state for 3D
    isAxialPrimaryLayoutActive: false, // Default state for Axial Primary
    isColorBarVisible: false,
    layout: {
        numRows: 1,
        numCols: 1
    },
    isRightPanelOpen: true,
    isStudiesPanelOpen: false,
    isInfoOnViewportsShown: true,

    // viewport
    viewports: [],
    renderingEngineId: 'myRenderingEngine',
    selectedViewportId: '',
    selectedSeriesInstanceUid: '',
    studyData: null,
    annotationToolGroupIds: [],
    currentToolGroupId: '',
    currentStudyInstanceUid: '',
    selectedCornerstoneTools: [],
    viewportsWithCinePlayer: [],
    segmentations: [],
    selectedStudyReports: []
};

const viewportsSlice = createSlice({
    name: 'viewer',
    initialState,
    reducers: {
        ...viewerUiReducer,
        ...viewerViewportReducer,
        ...viewerAnnotationReducer,
        ...viewerSegmentationReducer,
        setCrosshairActive: (state, action) => {
            state.isCrosshairActive = action.payload; //  Define setCrosshairActive
        },
        setColorBarVisible: (state, action) => {
            state.isColorBarVisible = action.payload;
        }
    }
});

export const viewerSliceActions = viewportsSlice.actions;

export default viewportsSlice;
