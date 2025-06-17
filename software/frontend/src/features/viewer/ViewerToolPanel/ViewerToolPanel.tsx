import { useState } from 'react';
import { SidePanel } from '@ui/library';
import SegmentationTab from './Segmentation/SegmentationTab.tsx';
import MeasurementsTab from '@features/viewer/ViewerToolPanel/MeasurementsTab.tsx';
import SequenceSynthesisTab from './Synthesis/SequenceSynthesisTab.tsx';
import MotionArtifactsCorrectionTab from './MotionArtifactsCorrection/MotionArtifactsCorrectionTab.tsx';
import ReportingTab from './Reporting/ReportingTab.tsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
    faRuler,
    faBrush,
    faBriefcaseMedical,
    faPlus,
    faFile,
    faBars
} from '@fortawesome/free-solid-svg-icons';
import { Box, Menu, MenuItem, IconButton, Typography } from '@mui/material';

const ViewerToolPanel = () => {
    const [activeTabIndex, setActiveTabIndex] = useState<number>(0);
    const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
    const open = Boolean(menuAnchor);

    const handleMenuClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        setMenuAnchor(event.currentTarget);
    };

    const handleMenuClose = (index: number) => {
        setActiveTabIndex(index);
        setMenuAnchor(null);
    };

    return (
        <Box className="h-full">
            <SidePanel title={'Tools'} >
                <Box className="flex flex-col h-full">
                    <Box className="flex justify-between items-center p-2">
                        <Box className="flex gap-4">
                            <Box className="flex flex-col items-center">
                                <IconButton onClick={() => setActiveTabIndex(0)}>
                                    <FontAwesomeIcon icon={faBrush} />
                                </IconButton>
                                <Typography variant="caption">Segmentation</Typography>
                            </Box>
                            <Box className="flex flex-col items-center">
                                <IconButton onClick={() => setActiveTabIndex(1)}>
                                    <FontAwesomeIcon icon={faRuler} />
                                </IconButton>
                                <Typography variant="caption">Measurements</Typography>
                            </Box>
                            <Box className="flex flex-col items-center">
                                <IconButton onClick={() => setActiveTabIndex(4)}>
                                    <FontAwesomeIcon icon={faFile} className="mr-2" />
                                </IconButton>
                                <Typography variant="caption">Reporting</Typography>
                            </Box>
                        </Box>
                        <IconButton onClick={handleMenuClick}>
                            <FontAwesomeIcon icon={faBars} />
                        </IconButton>
                        <Menu
                            anchorEl={menuAnchor}
                            open={open}
                            onClose={() => setMenuAnchor(null)}
                            anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
                            transformOrigin={{ vertical: 'top', horizontal: 'left' }}
                        >
                            <MenuItem onClick={() => handleMenuClose(3)}>
                                <FontAwesomeIcon icon={faPlus} className="mr-2" /> Sequence Synthesis
                            </MenuItem>
                            <MenuItem onClick={() => handleMenuClose(2)}>
                                <FontAwesomeIcon icon={faBriefcaseMedical} className="mr-2" /> Motion
                                Correction
                            </MenuItem>
                        </Menu>
                    </Box>
                    <Box className="flex-1 overflow-y-auto overflow-x-hidden custom-scrollbar">
                        <div className="w-full p-4">
                            {activeTabIndex === 0 && <SegmentationTab />}
                            {activeTabIndex === 1 && <MeasurementsTab />}
                            {activeTabIndex === 2 && <MotionArtifactsCorrectionTab />}
                            {activeTabIndex === 3 && <SequenceSynthesisTab />}
                            {activeTabIndex === 4 && <ReportingTab />}
                        </div>
                    </Box>
                </Box>
            </SidePanel>
        </Box>
    );
};

export default ViewerToolPanel;
