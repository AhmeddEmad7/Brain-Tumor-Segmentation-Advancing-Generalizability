import { useTheme, Box, IconButton, Menu, MenuItem } from '@mui/material';
import { Logo } from '@ui/library';
import { Link } from 'react-router-dom';
import {
    VIEWER_TOOLS_BUTTONS,
    VIEWER_OPTION_BUTTONS,
    type ViewerButtonType
} from '@features/viewer/ViewerTopBar/viewer-buttons.tsx';
import ViewerToolButton from '@features/viewer/ViewerTopBar/ViewerToolButton.tsx';
import CustomButton from '@features/top-bars/components/CustomButton.tsx';
import { Tooltip } from '@mui/material';
import { useSelector } from 'react-redux';
import { IStore } from '@/models';
import { MoreVert as MenuIcon } from '@mui/icons-material';
import { useState } from 'react';
import { ExpandCircleDown as ExpandCircleDownIcon } from '@mui/icons-material';

const ViewerTopBar = () => {
    const theme = useTheme();
    const is3DActive = useSelector((state: IStore) => state.viewer.is3DActive);
    const isDarkMode = theme.palette.mode === 'dark';
    const tools = VIEWER_TOOLS_BUTTONS(is3DActive);
    const primaryTools = tools.slice(0, 6);
    const otherTools = tools.slice(6);

    const [anchorEl, setAnchorEl] = useState(null);
    const [selectedTool, setSelectedTool] = useState<ViewerButtonType | null>(null);

    const handleMenuOpen = (event) => {
        setAnchorEl(event.currentTarget);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
    };

    const handleToolSelect = (tool) => {
        setSelectedTool(tool);
        tool.onClick();
        handleMenuClose();
    };

    return (
        <Box
            className={'flex justify-between items-center w-full'}
            sx={{
                background: isDarkMode ? '' : 'linear-gradient(to right, #E0E0E0, #FFFFFF)',
                color: isDarkMode ? '#00A8E8' : '#004E64',
                height: '4.1rem',
                display: 'flex',
                alignItems: 'center',
                padding: '0 1rem'
            }}
            onContextMenu={(e) => e.preventDefault()}
        >
            <Box className={'flex items-center'}>
                <Box className={'w-44 bg-transparent p-2'}>
                    <Link to={'/'}>
                        <Logo />
                    </Link>
                </Box>
            </Box>

            <Box className={'flex justify-center items-center'} sx={{ gap: 1 }}>
                {primaryTools.map((button, index) => (
                    <Tooltip
                        key={index}
                        title={button.disabled ? 'Not available on the current viewport' : button.title}
                        arrow
                    >
                        <div>
                            <ViewerToolButton
                                onClick={button.onClick}
                                icon={button.icon}
                                menuComponent={button.menuComponent}
                                title={button.title}
                                disabled={button.disabled}
                                sx={{
                                    width: '60px',
                                    minWidth: 'auto',
                                    cursor: button.disabled ? 'not-allowed' : 'pointer',
                                    color: isDarkMode ? '#FFFFFF' : '#000000'
                                }}
                            />
                        </div>
                    </Tooltip>
                ))}
                <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={handleMenuClose} >
                    {otherTools.map((tool, index) => (
                        <MenuItem key={index} onClick={() => handleToolSelect(tool)}>
                        <Box component="span" sx={{  marginRight: 1 ,}}>{tool.icon}</Box> 
                        {tool.title}
                    </MenuItem>
                    ))}
                </Menu>
                {/* New Dynamic Button */}
                {selectedTool && (
                    <Tooltip title={selectedTool.title} arrow>
                        <ViewerToolButton
                            onClick={selectedTool.onClick}
                            icon={selectedTool.icon}
                            title={selectedTool.title}
                            disabled={selectedTool.disabled}
                            sx={{
                                width: '60px',
                                minWidth: 'auto',
                                cursor: selectedTool.disabled ? 'not-allowed' : 'pointer',
                                color: isDarkMode ? '#FFFFFF' : '#000000'
                            }}
                        />
                    </Tooltip>
                )}
                <Tooltip title={'More Tools'} arrow>
                    <IconButton onClick={handleMenuOpen}>
                        <ExpandCircleDownIcon />
                    </IconButton>
                </Tooltip>
            </Box>

            <Box className={'flex items-center'}>
                <Box className={'space-x-1'}>
                    {VIEWER_OPTION_BUTTONS.map((button, index) => (
                        <CustomButton
                            key={index}
                            onClick={button.onClick}
                            icon={button.icon}
                            menuItems={button.menuItems}
                            menuComponent={button.menuComponent}
                            sx={{ color: isDarkMode ? '#FFFFFF' : '#000000' }}
                        />
                    ))}
                </Box>
            </Box>
        </Box>
    );
};

export default ViewerTopBar;
