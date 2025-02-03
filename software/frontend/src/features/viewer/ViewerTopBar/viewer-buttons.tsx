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
    ThreeDRotationSharp  as ThreeDRotationIcon 
} from '@mui/icons-material';
import { LuAxis3D } from 'react-icons/lu';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faLayerGroup, faUpDownLeftRight, faCirclePlay } from '@fortawesome/free-solid-svg-icons';
import { FaPaintBrush } from 'react-icons/fa';

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

const is3DActive = store.getState().viewer.is3DActive;
console.log('is3DActive',is3DActive);
const VIEWER_TOOLS_BUTTONS = [
    {
        title: ANNOTATION_TOOLS['Window'].toolName,
        onClick: handleToolClick,
        icon: <ContrastIcon />,
        menuComponent: <ViewerButtonMenu items={WindowButtonItems} />,
        disabled: is3DActive // ✅ Disable if 3D is active
    },
    {
        title: ANNOTATION_TOOLS['Pan'].toolName,
        onClick: handleToolClick,
        icon: <FontAwesomeIcon icon={faUpDownLeftRight} />,
        menuComponent: <ViewerButtonMenu items={PanButtonItems} />,
        disabled: is3DActive // ✅ Disable if 3D is active
    },
    {
        title: ANNOTATION_TOOLS['Zoom'].toolName,
        onClick: handleToolClick,
        icon: <ZoomToolIcon />,
        menuComponent: <ViewerButtonMenu items={ZoomButtonItems} />,
        disabled: is3DActive
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
        disabled: is3DActive
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
        title: 'Layout',
        icon: <LayoutIcon />,
        menuComponent: <LayoutSelector rows={4} columns={4} />,
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
    {
        title: 'MPR',
        onClick: () => {
            const state = store.getState();
            const renderingEngineId = state.viewer.renderingEngineId;
            const currentStudyInstanceUid = state.viewer.currentStudyInstanceUid;
            const selectedSeriesInstanceUid = state.viewer.selectedSeriesInstanceUid;
            store.dispatch(viewerSliceActions.setMPRActive(true));
            toggleMPRMode(renderingEngineId, selectedSeriesInstanceUid, currentStudyInstanceUid);
        },
        icon: <LuAxis3D />,
        disabled: is3DActive
    },
    {
        icon: <ThreeDIcon />,
        title: '3D',
        onClick: async () => {
            await toggleVolumeRendering();
        },
        
        disabled: is3DActive // Disable the 3D button itself when already in 3D mode
    },
    {
        icon: <ThreeDRotationIcon />,
        title: 'Render',
        onClick: handleToolClick,
        disabled: false // ✅ Render tool is always active
    },
    {
        icon: <ResetIcon />,
        title: 'Reset',
        disabled: is3DActive
    }
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
