import { useEffect, useState } from 'react';
import SeriesCard from './SeriesCard.js';
import PermMediaIcon from '@mui/icons-material/PermMedia';
import { useDispatch } from 'react-redux';
import { viewerSliceActions } from '@features/viewer/viewer-slice.ts';
import { IDicomSeriesData, IDicomStudyData } from '@/models';
import { useTheme } from '@mui/material/styles';
import { Box, Button, Typography } from '@mui/material';

const StudyCard = ({ studyData }: { studyData: IDicomStudyData }) => {
    const theme = useTheme();
    const isDarkMode = theme.palette.mode === 'dark';
    
    const [isStudyOpen, setIsStudyOpen] = useState(true);
    const [selectedSeries, setSelectedSeries] = useState(0);
    const [isAllSeriesHidden, setIsAllSeriesHidden] = useState(false);

    const dispatch = useDispatch();

    useEffect(() => {
        dispatch(viewerSliceActions.setStudyData(studyData.series));
    }, [studyData]);

    const seriesSelectHandler = (index: number, seriesInstanceUid: string) => {
        dispatch(viewerSliceActions.setClickedSeries(seriesInstanceUid));
        setSelectedSeries(index);
    };

    return (
        <Box
            className="rounded-lg p-4 transition-all duration-300 shadow-md"
            sx={{
                background: isDarkMode 
                    ? `linear-gradient(to right, #0A192F, #112D4E)`
                    : `linear-gradient(to right, #F8F9FA, #E9ECEF)`,
                color: isDarkMode ? 'white' : 'black',
                width: isAllSeriesHidden ? '250px' : '100%', 
                transition: 'width 0.3s ease-in-out, background 0.3s ease-in-out',
                border: isDarkMode ? '1px solid #2C3E50' : '1px solid #D1D9E6',
            }}
        >
            {/* Study Card Header */}
            <Box className="flex justify-between text-base mb-3">
                <Typography variant="h6" fontWeight="bold" className="truncate" title={studyData.patientName}>
                    {studyData.patientName}
                </Typography>
            </Box>

            {/* Study Middle Part */}
            <Box className="flex justify-between text-base mb-4">
                <Box>
                    <Typography variant="h6" fontWeight="bold">
                        {studyData.modality}
                    </Typography>
                    <Typography variant="body2" className="opacity-80" title={studyData.studyDate}>
                        {studyData.studyDate}
                    </Typography>
                </Box>
                <Box className="flex items-center gap-2">
                    <PermMediaIcon fontSize="small" />
                    <Typography variant="body2">{studyData.studyTotalInstances}</Typography>
                </Box>
            </Box>

            {/* Toggle All Series Visibility */}
            <Button
                variant="contained"
                size="small"
                onClick={() => setIsAllSeriesHidden(!isAllSeriesHidden)}
                sx={{
                    transition: 'background-color 0.3s ease-in-out',
                    backgroundColor: isDarkMode 
                        ? isAllSeriesHidden ? '#005B8C' : '#007BB5'
                        : isAllSeriesHidden ? '#E0E0E0' : '#6FB3D2',
                    color: isDarkMode ? 'white' : 'black',
                    '&:hover': { 
                        backgroundColor: isDarkMode 
                            ? isAllSeriesHidden ? '#003F6B' : '#005B8C'
                            : isAllSeriesHidden ? '#D6D6D6' : '#5A99B1' 
                    },
                }}
            >
                {isAllSeriesHidden ? "Show All Series" : "Hide All Series"}
            </Button>

            {/* Render Series Only If Not Hidden */}
            {isStudyOpen && !isAllSeriesHidden && (
                <Box className="max-h-[84vh] overflow-y-auto mt-4">
                    {studyData.series.map((series: IDicomSeriesData, index: number) => (
                        <SeriesCard
                            key={index}
                            seriesIndex={index}
                            selectedIndex={selectedSeries}
                            seriesData={series}
                            onSelectedSeriesChange={seriesSelectHandler}
                        />
                    ))}
                </Box>
            )}
        </Box>
    );
};

export default StudyCard;
