import { PaletteColor } from '@mui/material/styles';
declare module 'dcmjs';
declare module '@cornerstonejs/dicom-image-loader';

declare module '@mui/material/styles' {
    interface PaletteColor {
        lighter: string;
    }
    interface Palette {
        neutral: PaletteColor;
    }
}
