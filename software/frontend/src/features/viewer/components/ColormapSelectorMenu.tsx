import { MenuItem } from '@mui/material';

type Props = {
  applyColormap: (colormapName: string) => void;
};

const colormapPresets: Record<string, string> = {
  Grayscale: 'Grayscale',
  'X Ray': 'X Ray',
  'Hot Iron': 'Black-Body Radiation',
  'Red Hot': 'Inferno (matplotlib)',
  'S PET': 'Spectral_lowBlue',
  Perfusion: 'Cool to Warm',
  Rainbow: 'Rainbow Blended Grey',
  Siemens: 'Red to Blue Rainbow'
};

const clinicalNames = Object.keys(colormapPresets);

const ColormapSelectorMenu = ({ applyColormap }: Props) => {
  return (
    <>
      {clinicalNames.map((name) => (
        <MenuItem
          key={name}
          onClick={() => applyColormap(colormapPresets[name])}
        >
          {name}
        </MenuItem>
      ))}
    </>
  );
};

export default ColormapSelectorMenu;
