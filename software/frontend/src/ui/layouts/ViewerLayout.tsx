import { Outlet } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import ViewerTopBar from '@features/viewer/ViewerTopBar/ViewerTopBar.tsx';
import ViewerSidebar from '@features/viewer/StudySidebar/ViewerSidebar.tsx';
import ViewerToolPanel from '@features/viewer/ViewerToolPanel/ViewerToolPanel.tsx';

const ViewerLayout = () => {
    return (
        <div className={'w-full h-screen overflow-hidden'}>
            <Helmet>
                <title>MMM.AI Viewer</title>
                <meta
                    name="description"
                    content="Multimodal Medical Viewer for brain tumor segmentation and MRI Motion Artifacts Correction."
                />
            </Helmet>

            <ViewerTopBar />
            <div className={'flex h-[92.5vh]'}>
                <div className={'h-full w-3/12 max-w-52 '}>
                    <ViewerSidebar className={'h-full'} />
                </div>
                <div className={'h-full flex-grow w-11/12'} onContextMenu={(e) => e.preventDefault()}>
                    <Outlet />
                </div>
                <div className={'h-full'} onContextMenu={(e) => e.preventDefault()}>
                    <ViewerToolPanel />
                </div>
            </div>
        </div>
    );
};

export default ViewerLayout;
