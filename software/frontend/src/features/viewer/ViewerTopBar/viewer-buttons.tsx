import {
    Contrast as ContrastIcon,
    ZoomIn as ZoomToolIcon,
    Straighten as MeasurementToolIcon,
    RotateLeft as RotationToolIcon,
    GridView as LayoutIcon,
    ZoomOutMap as FullScreenIcon,
    ImageSearch as MagnifyIcon,
    VerticalAlignBottom as ExportIcon,
    Info as InfoIcon,
    ViewInAr as ThreeDIcon,
    History as ResetIcon,
    NotificationImportantOutlined as NotificationOutlinedIcon,
    DarkModeOutlined as DarkModeOutlinedIcon,
    LightModeOutlined as LightModeOutlinedIcon,
    Translate as LanguagesIcon,
    SettingsOutlined as SettingsOutlinedIcon,
    Upload as UploadIcon,
    ThreeDRotationSharp as ThreeDRotationIcon
} from '@mui/icons-material';
import * as cornerstoneTools from '@cornerstonejs/tools';
import * as cornerstone from '@cornerstonejs/core';
import { LuAxis3D } from 'react-icons/lu';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faLayerGroup, faUpDownLeftRight, faCirclePlay } from '@fortawesome/free-solid-svg-icons';
import { GiCrosshair } from 'react-icons/gi'; // Import crosshair icon

import store from '@/redux/store.ts';
import {
    handleColorModeChange,
    handleLanguageChange,
    handleSettingsItemClick
} from '@features/top-bars/topbars-actions.ts';
import { ANNOTATION_TOOLS, SEGMENTATION_TOOLS } from '@/features/viewer/CornerstoneToolManager/';

import LayoutSelector from '@features/viewer/components/LayoutSelector.tsx';
import {
    handleToolClick,
    toggleFullScreen,
    toggleViewportOverlayShown,
    toggleMPRMode,
    toggleVolumeRendering
} from '@features/viewer/ViewerTopBar/viewer-top-bar-actions.ts';
import NotificationsMenu from '@features/notifications/NotificationsMenu.tsx';
import ViewerButtonMenu from '@features/viewer/components/ViewerButtonMenu.tsx';
import { LANGUAGE_MENU_ITEMS } from '@features/top-bars/HomeTopBar/home-buttons.tsx';
import {
    MeasurementsButtonItems,
    PanButtonItems,
    RotateButtonItems,
    ZoomButtonItems,
    WindowButtonItems
} from '@features/viewer/ViewerTopBar/options-menu-items';
import CornerstoneToolManager from '@/features/viewer/CornerstoneToolManager/CornerstoneToolManager';
import { viewerSliceActions } from '@features/viewer/viewer-slice.ts';

export const VIEWER_SETTINGS_MENU_ITEMS = ['About', 'License Agreement', 'Help', 'Shortcuts'];
import ColormapSelectorMenu from '@features/viewer/components/ColormapSelectorMenu';
import { applyColormapToViewport } from './viewer-top-bar-actions';
import { Palette as PaletteIcon } from '@mui/icons-material';
import { icon } from '@fortawesome/fontawesome-svg-core';
import { setActiveSegmentIndex } from '@cornerstonejs/tools/dist/types/stateManagement/segmentation/segmentIndex';
// import ColormapSelectorMenu from '@/components/ColormapSelectorMenu';

const is3DActive = store.getState().viewer.is3DActive;
console.log('is3DActive', is3DActive);
const VIEWER_TOOLS_BUTTONS = (is3DActive) => [
    {
        title: ANNOTATION_TOOLS['Window'].toolName,
        onClick: handleToolClick,
        icon: <ContrastIcon />,
        menuComponent: <ViewerButtonMenu items={WindowButtonItems} />,
        disabled: false
    },
    {
        title: ANNOTATION_TOOLS['Pan'].toolName,
        onClick: handleToolClick,
        icon: <FontAwesomeIcon icon={faUpDownLeftRight} />,
        menuComponent: <ViewerButtonMenu items={PanButtonItems} />,
        disabled: false
    },
    {
        title: ANNOTATION_TOOLS['Zoom'].toolName,
        onClick: handleToolClick,
        icon: <ZoomToolIcon />,
        menuComponent: <ViewerButtonMenu items={ZoomButtonItems} />,
        disabled: false
    },
    {
        title: 'Measurements',
        onClick: handleToolClick,
        icon: <MeasurementToolIcon />,
        menuComponent: <ViewerButtonMenu items={MeasurementsButtonItems} />,
        disabled: is3DActive
    },
    {
        title: 'Rotate',
        onClick: handleToolClick,
        icon: <RotationToolIcon />,
        menuComponent: <ViewerButtonMenu items={RotateButtonItems} />,
        disabled: false
      },
      {
          title: 'Layout',
          icon: <LayoutIcon />,
          menuComponent: <LayoutSelector rows={4} columns={4} />,
          disabled: false
      },
      {
        title: 'MPR',
        onClick: async () => {
          const state = store.getState();
          const renderingEngineId = state.viewer.renderingEngineId;
          const currentStudyInstanceUid = state.viewer.currentStudyInstanceUid;
          const selectedSeriesInstanceUid = state.viewer.selectedSeriesInstanceUid;
          const isActive = state.viewer.isMPRActive;
        //   const isMPRActive = state.viewer.isMPRActive;
      
          if (!isActive) {
            store.dispatch(viewerSliceActions.setMPRActive(true));
            // store.dispatch(setActiveSegmentIndex)
            store.dispatch(viewerSliceActions.setClickedSeries(selectedSeriesInstanceUid));
            await toggleMPRMode(renderingEngineId, selectedSeriesInstanceUid, currentStudyInstanceUid);
          } else {
            CornerstoneToolManager.disableAllTools();
            store.dispatch(viewerSliceActions.resetViewerLayout());
            store.dispatch(viewerSliceActions.setMPRActive(false));
          }
        },
        icon: <LuAxis3D />,
        disabled: false
      },
       {
        icon: <ThreeDIcon />,
        title: '3D',
        onClick: async () => {
            await toggleVolumeRendering();
        },
        disabled: false // Disable the 3D button itself when already in 3D mode
    },
        {
        icon: <ResetIcon />,
        title: 'Reset',
        disabled: false
    },
     {
        title: 'Crosshair',
        onClick: () => {
          const state = store.getState();
          const renderingEngineId = state.viewer.renderingEngineId;
          const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
          const isCurrentlyActive = state.viewer.isCrosshairActive;
      
          const viewports = renderingEngine?.getViewports();
          if (!viewports || viewports.length < 2) {
            console.warn('❌ Crosshairs require at least two viewports.');
            return;
          }
      
          store.dispatch(viewerSliceActions.setCrosshairActive(!isCurrentlyActive));
      
          if (!isCurrentlyActive) {
            CornerstoneToolManager.disableAllTools();
            CornerstoneToolManager.setToolActive(
              cornerstoneTools.CrosshairsTool.toolName,
              cornerstoneTools.Enums.MouseBindings.Primary
            );
          } else {
            CornerstoneToolManager.disableAllTools();
          }
        },
        icon: <GiCrosshair />,
        disabled: false
      },
   
    {
        icon: <ThreeDRotationIcon />,
        title: 'Render',
        onClick: handleToolClick,
        disabled: !is3DActive
    },
     {
        title: 'Colormap',
        icon: <PaletteIcon />,
        menuComponent: (
          <ColormapSelectorMenu
            applyColormap={(vtkPresetName) => {
              // This is your existing code that applies the selected preset
              const state = store.getState();
              const renderingEngineId = state.viewer.renderingEngineId;
              const viewportId = state.viewer.selectedViewportId;
      
              let volumeId = '';
              if (
                state.viewer.currentStudyInstanceUid.endsWith('.nii') ||
                state.viewer.currentStudyInstanceUid.endsWith('.gz')
              ) {
                const niftiURL = `${import.meta.env.VITE_NIFTI_DOMAIN}/${state.viewer.currentStudyInstanceUid}`;
                volumeId = 'nifti:' + niftiURL;
              } else {
                volumeId = `cornerstoneStreamingImageVolume:${state.viewer.selectedSeriesInstanceUid}`;
              }
      
              applyColormapToViewport(
                vtkPresetName,
                renderingEngineId,
                viewportId,
                volumeId
              );
            }}
          />
        ),
        disabled: false
      },
    {
        title: 'Magnify',
        onClick: handleToolClick,
        icon: <MagnifyIcon />,
        disabled: is3DActive
    },
    {
        title: 'Scroll',
        icon: <FontAwesomeIcon icon={faLayerGroup} />,
        onClick: handleToolClick,
        disabled: is3DActive
    },
    {
        title: 'Cine',
        icon: <FontAwesomeIcon icon={faCirclePlay} />,
        onClick: handleToolClick,
        disabled: is3DActive
    },
    {
        title: 'Full Screen',
        icon: <FullScreenIcon />,
        onClick: toggleFullScreen,
        disabled: is3DActive
    },
    {
        title: 'SEG',
        onClick: () => {
            CornerstoneToolManager.uploadSegmentation();
        },
        icon: <UploadIcon />,
        disabled: is3DActive
    },
    {
        icon: <InfoIcon />,
        title: 'Info',
        onClick: toggleViewportOverlayShown,
        disabled: is3DActive
    },
    

];
const VIEWER_OPTION_BUTTONS = [
    {
        onClick: () => {},
        icon: <NotificationOutlinedIcon />,
        menuComponent: <NotificationsMenu />
    },
    {
        onClick: handleColorModeChange,
        icon: store.getState().ui.themeMode === 'light' ? <DarkModeOutlinedIcon /> : <LightModeOutlinedIcon />
    },
    {
        onClick: handleLanguageChange,
        icon: <LanguagesIcon />,
        menuItems: LANGUAGE_MENU_ITEMS
    },
    {
        onClick: handleSettingsItemClick,
        icon: <SettingsOutlinedIcon />,
        menuItems: VIEWER_SETTINGS_MENU_ITEMS
    }
];
export { VIEWER_TOOLS_BUTTONS, VIEWER_OPTION_BUTTONS };
