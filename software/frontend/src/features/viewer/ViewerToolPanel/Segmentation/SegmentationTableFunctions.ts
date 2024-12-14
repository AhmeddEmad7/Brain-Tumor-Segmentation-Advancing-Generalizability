import { viewerSliceActions } from '@features/viewer/viewer-slice';
import * as cornerstoneTools from '@cornerstonejs/tools';
import store from '@/redux/store.ts';
import { removeSegmentationAndUpdateActiveSegmentation } from '@features/viewer/viewer-segmentation-reducers';

// ---------------------- Segmentation Table Functions ---------------------- //

const handleSegmentationVisibilityToggle = (segmentationId: string) => {
    store.dispatch(viewerSliceActions.handleSegmentationVisibility({ segmentationId }));
};
const handleSegmentVisibilityToggle = (segmentationId: string, segmentIndex: number)=>{
    store.dispatch(
        viewerSliceActions.handleSegmentVisibilityToggle({
            segmentationId,
            segmentIndex
        })
    );
    console.log('Segment toggle', segmentationId, segmentIndex);

};

const handleSegmentLockToggle = (segmentationId: string, segmentIndex: number) => {
    const currentLock = cornerstoneTools.segmentation.segmentLocking.isSegmentIndexLocked(
        segmentationId,
        segmentIndex
    );

    cornerstoneTools.segmentation.segmentLocking.setSegmentIndexLocked(
        segmentationId,
        segmentIndex,
        !currentLock
    );
};

const handleSegmentationDelete = (segmentationId: string) => {
    try {
        store.dispatch(removeSegmentationAndUpdateActiveSegmentation(segmentationId));
        cornerstoneTools.segmentation.state.removeSegmentation(segmentationId);

        const currentToolGroupId = store.getState().viewer.currentToolGroupId;
        const segmentationRepresentation =
            cornerstoneTools.segmentation.state.getSegmentationIdRepresentations(segmentationId);

        cornerstoneTools.segmentation.removeSegmentationsFromToolGroup(
            currentToolGroupId,
            [segmentationRepresentation[0].segmentationRepresentationUID],
            true
        );
    } catch (error) {
        console.error('Error removing segmentation:', error);
    }
};

const handleSegmentDelete = (segmentationId: string , segmentIndex:number) => {
    store.dispatch(viewerSliceActions.removeSegment({segmentationId, segmentIndex}));
    console.log('Segment Deleted', segmentationId, segmentIndex);
};
const handleSegmentClick = (segmentationId: string, segmentIndex: number) => {
    store.dispatch(viewerSliceActions.handleSegmentClick({ segmentationId, segmentIndex }));
    // console.log('Segment Clicked', segmentationId, segmentIndex);
};

const onSegmentationClick = (segmentationId: string) => {
    store.dispatch(viewerSliceActions.onSegmentationClick({ segmentationId }));
};

export {
    handleSegmentationVisibilityToggle,
    handleSegmentVisibilityToggle,
    handleSegmentLockToggle,
    handleSegmentationDelete,
    handleSegmentClick,
    handleSegmentDelete,
    onSegmentationClick
};
