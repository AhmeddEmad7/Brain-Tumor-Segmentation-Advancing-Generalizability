import {
    InvertColors as InvertTool,
    Sync as SyncTool,
    Palette as ColorBarIcon,
    Settings as SettingsIcon
} from '@mui/icons-material';
import { TViewerButtonItems } from '../../components/ViewerButtonMenu';
import store from '@/redux/store';
import { getRenderingEngine, Types, utilities } from '@cornerstonejs/core';
import { viewerSliceActions } from '@features/viewer/viewer-slice.ts';
import { utilities as cstUtils } from '@cornerstonejs/tools';
import vtkColormaps from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction/ColorMaps';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import React, { ReactNode } from 'react';
import * as ContextMenu from '@radix-ui/react-context-menu';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { applyColormapToViewport } from '../viewer-top-bar-actions';
import ColormapSelectorMenu from '../../components/ColormapSelectorMenu';
import { SvgIcon } from '@mui/material';

const { ViewportColorbar, Enums: ColorBarEnums } = cstUtils.voi.colorbar;
const ColorLUTIcon = (props: any) => (
    <SvgIcon {...props} viewBox="0 0 64 64">
      <path
        d="M32 2C20 16 8 28 8 40c0 13.255 10.745 24 24 24s24-10.745 24-24C56 28 44 16 32 2z"
        fill="#fff"
        stroke="#000"
        strokeWidth="2"
      />
    </SvgIcon>
  );
export type TViewerButtonItems = {
    icon: ReactNode;
    label: string;
    onClick?: (e: React.MouseEvent) => void;
    divider?: boolean;
    menuItems?: Array<{
        label: string;
        onClick: () => void;
    }>;
    component?: React.ReactNode;
};
let viewportColorbar: any = null;

let colorbarContainer: HTMLDivElement | null = null;
const CT_PRESETS = [
    { label: 'Soft Tissue', ww: 400, wc: 40 },
    { label: 'Lung', ww: 1500, wc: -600 },
    { label: 'Liver', ww: 150, wc: 90 },
    { label: 'Bone', ww: 2500, wc: 480 },
    { label: 'Brain', ww: 80, wc: 40 }
];

const invert = () => {
    const { selectedViewportId, renderingEngineId } = store.getState().viewer;
    const renderingEngine = getRenderingEngine(renderingEngineId);
    if (!renderingEngine) return;

    const viewport = renderingEngine.getViewport(selectedViewportId) as Types.IVolumeViewport;
    const { invert } = viewport.getProperties();
    viewport.setProperties({ invert: !invert });
    viewport.render();
};

const toggleColorBar = () => {
    const { selectedViewportId, renderingEngineId, isColorBarVisible } = store.getState().viewer;
    const renderingEngine = getRenderingEngine(renderingEngineId);

    if (!renderingEngine) return;

    const viewport: Types.IVolumeViewport = renderingEngine.getViewport(
        selectedViewportId
    ) as Types.IVolumeViewport;

    const viewportElement = document.getElementById(selectedViewportId);
    if (!viewportElement || !viewportElement.parentElement) return;

    const props = viewport.getProperties();

    viewport.render();

    const visible = !isColorBarVisible;

    if (!colorbarContainer) {
        colorbarContainer = document.createElement('div');
        Object.assign(colorbarContainer.style, {
            position: 'absolute',
            right: '30px',
            top: '50%',
            transform: 'translateY(-50%)',
            width: '15px',
            height: '60%',
            pointerEvents: 'auto',
            zIndex: '5',
            cursor: 'ns-resize'
        });
        viewportElement.appendChild(colorbarContainer);

        const colormaps = vtkColormaps.rgbPresetNames.map((name) => vtkColormaps.getPresetByName(name));

        viewportColorbar = new ViewportColorbar({
            id: 'myColorbar',
            element: viewportElement,
            container: colorbarContainer,
            colormaps,
            activeColormapName: 'Grayscale',
            ticks: {
                position: ColorBarEnums.ColorbarRangeTextPosition.Left,
                style: { font: '12px Arial', color: '#fff' }
            }
        });
    }

    colorbarContainer.style.display = visible ? 'block' : 'none';

    store.dispatch({
        type: 'viewer/setColorBarVisible',
        payload: visible
    });
};

const WindowPresetMenu = () => {
    return (
        <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
                <div className="flex items-center space-x-2 hover:bg-AAPrimary cursor-pointer p-1 px-3">
                    <SettingsIcon />
                    <span>Window Presets</span>
                </div>
            </DropdownMenu.Trigger>

            <DropdownMenu.Portal>
                <DropdownMenu.Content
                    sideOffset={4}
                    className="z-50 ml-2 mt-2 min-w-[19px] bg-[#060C18] rounded-md p-1 shadow-lg"
                >
                    {CT_PRESETS.map((preset) => (
                        <DropdownMenu.Item
                            key={preset.label}
                            className="text-white px-2 py-1.5 text-sm rounded-sm cursor-pointer hover:bg-AAPrimary focus:bg-AAPrimary outline-none"
                            onSelect={() => {
                                const { selectedViewportId, renderingEngineId } = store.getState().viewer;
                                const viewport = getRenderingEngine(renderingEngineId)?.getViewport(
                                    selectedViewportId
                                ) as Types.IVolumeViewport;
                                if (!viewport) return;

                                try {
                                    const { ww, wc } = preset;
                                    console.log(`Setting VOI range to: WW=${ww}, WC=${wc}`);
                                    // Convert window/level to range using the utility function
                                    const range = utilities.windowLevel.toLowHighRange(ww, wc);
                                    console.log(`Calculated VOI range:`, range);
                                    viewport.setProperties({
                                        voiRange: { lower: range.lower, upper: range.upper }
                                    });
                                    viewport.render();
                                } catch (error) {
                                    console.error('Error setting VOI range:', error);
                                }
                            }}
                        >
                            <div className="flex justify-between items-center">
                                <span>{preset.label}</span>
                                <span className="ml-4 text-gray-200">{`${preset.ww}/${preset.wc}`}</span>
                            </div>
                        </DropdownMenu.Item>
                    ))}
                </DropdownMenu.Content>
            </DropdownMenu.Portal>
        </DropdownMenu.Root>
    );
};

const ColormapPresetMenu = () => {
    return (
        <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
                <div className="flex items-center space-x-2 hover:bg-AAPrimary cursor-pointer p-1 px-3">
                    <ColorLUTIcon />
                    <span>Color LUT</span>
                </div>
            </DropdownMenu.Trigger>

            <DropdownMenu.Portal>
                <DropdownMenu.Content
                    sideOffset={4}
                    className="z-50 ml-2 mt-9 min-w-[19px] bg-[#060C18] rounded-md p-1 shadow-lg"
                >
                    <ColormapSelectorMenu
                        applyColormap={(vtkPresetName) => {
                            const state = store.getState();
                            const {
                                selectedViewportId,
                                renderingEngineId,
                                currentStudyInstanceUid,
                                selectedSeriesInstanceUid
                            } = state.viewer;

                            // Get the correct volumeId based on file type
                            let volumeId = '';
                            if (
                                currentStudyInstanceUid?.endsWith('.nii') ||
                                currentStudyInstanceUid?.endsWith('.gz')
                            ) {
                                const niftiURL = `${import.meta.env.VITE_NIFTI_DOMAIN}/${currentStudyInstanceUid}`;
                                volumeId = 'nifti:' + niftiURL;
                            } else {
                                volumeId = `cornerstoneStreamingImageVolume:${selectedSeriesInstanceUid}`;
                            }

                            // Use the existing applyColormapToViewport function
                            applyColormapToViewport(
                                vtkPresetName,
                                renderingEngineId,
                                selectedViewportId,
                                volumeId
                            );

                            // Update colorbar if it exists
                            if (viewportColorbar) {
                                viewportColorbar.setColormap(vtkPresetName);
                            }
                        }}
                    />
                </DropdownMenu.Content>
            </DropdownMenu.Portal>
        </DropdownMenu.Root>
    );
};

const WindowItems: TViewerButtonItems[] = [
    {
        label: 'Invert',
        icon: <InvertTool />,
        onClick: invert
    },
    {
        label: 'Display Color Bar',
        icon: <ColorBarIcon />,
        onClick: toggleColorBar
    },
    {
        label: 'Color LUT',
        icon: <ColorLUTIcon />,
        component: <ColormapPresetMenu />
    },
    {
        label: 'Window Presets',
        icon: <SettingsIcon />,
        component: <WindowPresetMenu />
    }
    // {
    //     label: 'Sync Window across viewports',
    //     icon: <SyncTool />
    // }
];

export default WindowItems;
