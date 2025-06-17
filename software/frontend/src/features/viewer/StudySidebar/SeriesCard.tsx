import { useTheme } from '@mui/material/styles';
import { Image } from '@ui/library';
import { IDicomSeriesData } from '@/models';
import * as ContextMenu from '@radix-ui/react-context-menu';
import { useState } from 'react';
import DeleteSeriesModal from './DeleteSeriesModal';
import { Box, Typography, IconButton } from '@mui/material';
import { DeleteOutline } from '@mui/icons-material';

const gatewayUrl = import.meta.env.VITE_SERVER_URL;

interface SeriesCardProps {
    seriesData: IDicomSeriesData;
    seriesIndex: number;
    selectedIndex: number;
    onSelectedSeriesChange: (seriesIndex: number, seriesInstanceUid: string) => void;
}

const SeriesCard = ({ seriesData, seriesIndex, selectedIndex, onSelectedSeriesChange }: SeriesCardProps) => {
    const theme = useTheme();
    const [isDeleting, setIsDeleting] = useState(false);

    const isDarkMode = theme.palette.mode === 'dark';

    const backgroundGradient = isDarkMode
        ? 'linear-gradient(to right, #0A192F, #112D4E)' // Dark mode colors
        : 'linear-gradient(to right, #F3F4F6, #E5E7EB)'; // Light mode colors

    const borderColor =
        selectedIndex === seriesIndex
            ? isDarkMode
                ? '#00A8E8' // Highlight color in dark mode
                : '#1E40AF' // Highlight color in light mode
            : 'none';

    return (
        <>
            <ContextMenu.Root>
                <ContextMenu.Trigger>
                    <Box
                        className="flex flex-col rounded-lg p-3 transition-all duration-300 shadow-md cursor-pointer hover:shadow-lg"
                        sx={{
                            background: backgroundGradient,
                            color: isDarkMode ? 'white' : '#1F2937', // Text color
                            border: `2px solid ${borderColor}`
                        }}
                    >
                        <Box className="flex justify-between items-center mb-2">
                            <Typography variant="body2" fontWeight="bold" color="inherit">
                                {/* {seriesData.seriesModality} */}
                                {'MR'}
                            </Typography>
                            <Typography variant="body2" color="inherit">
                                {seriesData.numberOfInstances} images
                            </Typography>
                        </Box>

                        <Image
                            width={150}
                            height={150}
                            className="rounded-lg transition-all duration-300 hover:scale-105"
                            src={`${gatewayUrl}/dicom/studies/${seriesData.studyInstanceUid}/series/${seriesData.seriesInstanceUid}/image`}
                            alt="Series Thumbnail"
                            onDoubleClick={() =>
                                onSelectedSeriesChange(seriesIndex, seriesData.seriesInstanceUid)
                            }
                        />

                        <Typography
                            variant="body2"
                            color="inherit"
                            className="mt-2 truncate w-full"
                            title={seriesData.seriesDescription}
                        >
                            {seriesData.seriesDescription}
                        </Typography>
                    </Box>
                </ContextMenu.Trigger>
                <ContextMenu.Content
                    className={`z-50 min-w-[8rem] overflow-hidden rounded-lg p-2 shadow-md ${
                        isDarkMode ? 'bg-blue-900 text-white' : 'bg-gray-200 text-gray-800'
                    }`}
                >
                    <ContextMenu.Item
                        onClick={() => setIsDeleting(true)}
                        className={`flex items-center gap-2 transition-all hover:opacity-60 cursor-pointer ${
                            isDarkMode ? 'text-red-500' : 'text-red-600'
                        }`}
                    >
                        <DeleteOutline fontSize="medium" />
                        Delete {seriesData.seriesDescription}
                    </ContextMenu.Item>
                </ContextMenu.Content>
            </ContextMenu.Root>

            {isDeleting && (
                <DeleteSeriesModal
                    isOpen={isDeleting}
                    onClose={() => setIsDeleting(false)}
                    seriesId={seriesData.seriesInstanceUid}
                    seriesTitle={seriesData.seriesDescription}
                />
            )}
        </>
    );
};

export default SeriesCard;
