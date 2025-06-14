import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { Button } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { getRenderingEngine } from '@cornerstonejs/core';
import { Types } from '@cornerstonejs/core';
import { IStore } from '@/models';
import { SvgIcon } from '@mui/material';

const OrientationIcon = (props: any) => (
    <SvgIcon {...props} viewBox="0 0 32 32">
        <text x="13" y="14" fontSize="12" fill="cyan" fontWeight="bold">
            A
        </text>
        <path
            d="M10 20c3 4 9 4 12 0"
            stroke="cyan"
            strokeWidth="4"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
        />
        <path d="M20 20l-2 -2" stroke="cyan" strokeWidth="2" strokeLinecap="round" />
    </SvgIcon>
);

const orientations: ('axial' | 'coronal' | 'sagittal')[] = ['axial', 'coronal', 'sagittal',];

export const OrientationMenu = ({ viewportId }: { viewportId: string }) => {
    const { renderingEngineId } = useSelector((store: IStore) => store.viewer);

    const handleSetOrientation = (orientation: 'axial' | 'coronal' | 'sagittal') => {
        const renderingEngine = getRenderingEngine(renderingEngineId);
        const viewport = renderingEngine?.getViewport(viewportId) as Types.IVolumeViewport;

        if (!viewport) return;

        viewport.setOrientation(orientation);
        viewport.render();
    };

    return (
        <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild className="flex items-center justify-center ">
                <Button variant="outline" size="sm">
                    <OrientationIcon style={{ fontSize: '2.1rem' }} />
                </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content className="z-50 bg-[#060C18] text-white p-1 rounded shadow-md ml-10 w-[100px]">
                <div className="font-semibold ">Orientation</div>
                {orientations.map((ori) => (
                    <DropdownMenu.Item
                        key={ori}
                        className="px-2 py-1 hover:bg-AAPrimary cursor-pointer rounded"
                        onSelect={() => handleSetOrientation(ori)}
                    >
                        {ori.charAt(0).toUpperCase() + ori.slice(1)}
                    </DropdownMenu.Item>
                ))}
            </DropdownMenu.Content>
        </DropdownMenu.Root>
    );
};
