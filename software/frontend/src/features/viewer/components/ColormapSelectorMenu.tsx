import { useState } from 'react';
import { Menu, MenuItem, IconButton, Tooltip } from '@mui/material';
import PaletteIcon from '@mui/icons-material/Palette';

type Props = {
    applyColormap: (colormapName: string) => void;
};

// Clinical-friendly display names mapped to VTK colormap presets
const colormapPresets: Record<string, string> = {
    Grayscale: 'Grayscale',
    'X Ray': 'X Ray',
    // HSV: 'Rainbow Blended White', // or fallback like 'Rainbow Blended White'
    'Hot Iron': 'Black-Body Radiation',
    'Red Hot': 'Inferno (matplotlib)',
    'S PET': 'Spectral_lowBlue',
    Perfusion: 'Cool to Warm',
    Rainbow: 'Rainbow Blended Grey',
    // SUV: 'Hot',
    // 'GE 256': 'Black-Body Radiation',
    // GE: 'Cool to Warm',
    Siemens: 'Red to Blue Rainbow'
};

const clinicalNames = Object.keys(colormapPresets);

const ColormapSelectorMenu = ({ applyColormap }: Props) => {
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const open = Boolean(anchorEl);

    const handleClick = (event: React.MouseEvent<HTMLElement>) => {
        setAnchorEl(event.currentTarget);
    };

    const handleClose = (displayName?: string) => {
        if (displayName) {
            const vtkPresetName = colormapPresets[displayName];
            applyColormap(vtkPresetName);
        }
        setAnchorEl(null);
    };

    return (
        <>
            <Tooltip title="Colormap">
                <IconButton onClick={handleClick} size="small" sx={{ ml: 1 }}>
                    <PaletteIcon fontSize="medium" />
                </IconButton>
            </Tooltip>

            <Menu
                anchorEl={anchorEl}
                open={open}
                onClose={() => handleClose()}
                PaperProps={{
                    style: {
                        maxHeight: 300,
                        width: '250px'
                    }
                }}
            >
                {clinicalNames.map((name) => (
                    <MenuItem key={name} onClick={() => handleClose(name)}>
                        {name}
                    </MenuItem>
                ))}
            </Menu>
        </>
    );
};

export default ColormapSelectorMenu;
