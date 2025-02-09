import { useTheme } from '@mui/material/styles';
import StudyCard from './StudyCard.tsx';
import { useDispatch, useSelector } from 'react-redux';
import { IStore } from '@models/store.ts';
import { useEffect, useState } from 'react';
import { fetchDicomStudyByIdThunk } from '@features/studies-table/dicom-studies-table/dicom-studies-actions.ts';
import { TAppDispatch } from '@/redux/store.ts';
import { Box, Button } from '@mui/material';
import { fetchStudyReportByIdThunk } from '@features/report/report-actions.ts';

const ViewerSidebar = ({ className }: { className?: string }) => {
    const theme = useTheme();
    const dispatch = useDispatch<TAppDispatch>();
    
    const [isSidebarVisible, setIsSidebarVisible] = useState(true); // Toggle visibility

    useEffect(() => {
        const urlParams = new URLSearchParams(location.search);
        const studyInstanceUID = urlParams.get('StudyInstanceUID');

        if (studyInstanceUID) {
            dispatch(fetchDicomStudyByIdThunk(studyInstanceUID));
            dispatch(fetchStudyReportByIdThunk(studyInstanceUID));
        }
    }, []);

    const { selectedDicomStudy } = useSelector((store: IStore) => store.studies);

    return (
        <Box
            className={`${className}`}
            sx={{
                // backgroundColor: theme.palette.primary.dark
            }}
            onContextMenu={(e) => e.preventDefault()}
        >
            {/* Toggle Button */}
            <Button
                variant="contained"
                color="secondary"
                size="small"
                className="m-2"
                onClick={() => setIsSidebarVisible(!isSidebarVisible)}
            >
                {isSidebarVisible ? 'Hide Studies' : 'Show Studies'}
            </Button>

            {/* Show or Hide StudyCard */}
            {isSidebarVisible && selectedDicomStudy && <StudyCard studyData={selectedDicomStudy} />}
        </Box>
    );
};

export default ViewerSidebar;
