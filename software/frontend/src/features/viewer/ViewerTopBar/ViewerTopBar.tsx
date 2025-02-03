import { useTheme, Box } from '@mui/material';
import { Logo } from '@ui/library';
import { Link } from 'react-router-dom';
import {
    VIEWER_TOOLS_BUTTONS,
    VIEWER_OPTION_BUTTONS
} from '@features/viewer/ViewerTopBar/viewer-buttons.tsx';
import ViewerToolButton from '@features/viewer/ViewerTopBar/ViewerToolButton.tsx';
import CustomButton from '@features/top-bars/components/CustomButton.tsx';
import { Tooltip } from '@mui/material';

const ViewerTopBar = () => {
    const theme = useTheme();

    return (
        <Box
            className={'flex justify-between w-full'}
            sx={{
                backgroundColor: theme.palette.primary.dark,
                height: '4.1rem'
            }}
            onContextMenu={(e) => e.preventDefault()}
        >
            <Box className={'ml-3 flex'}>
                <Box className={'w-44 bg-transparent p-2'}>
                    <Link to={'/'}>
                        <Logo />
                    </Link>
                </Box>

                <Box className={'ml-2 flex'} sx={{ gap: 0.1 }}>
                    {VIEWER_TOOLS_BUTTONS.map((button, index) => (
                        <Tooltip key={index} title={button.title} arrow>
                            <div>
                                <ViewerToolButton
                                    onClick={button.onClick}
                                    icon={button.icon}
                                    menuComponent={button.menuComponent}
                                    title={button.title }
                                    disabled={button.disabled}
                                    sx={{ width: '60px ', minWidth: 'auto' ,cursor: button.disabled ? 'not-allowed' : 'pointer' }}
                                />
                            </div>
                        </Tooltip>
                    ))}
                </Box>
            </Box>

            <Box className={'flex items-center mr-1'}>
                <Box className={'space-x-1'}>
                    {VIEWER_OPTION_BUTTONS.map((button, index) => (
                        <CustomButton
                            key={index}
                            onClick={button.onClick}
                            icon={button.icon}
                            menuItems={button.menuItems}
                            menuComponent={button.menuComponent}
                        />
                    ))}
                </Box>
            </Box>
        </Box>
    );
};

export default ViewerTopBar;
