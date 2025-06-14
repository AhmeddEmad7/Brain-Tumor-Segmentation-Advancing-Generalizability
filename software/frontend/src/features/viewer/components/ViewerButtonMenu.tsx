import { ReactNode, useState } from 'react';

type TMenuButtonProps = {
    icon: ReactNode;
    label: string;
    onClick?: (e: React.MouseEvent<HTMLDivElement>) => void;
};

export type TViewerButtonItems = {
    icon: ReactNode;
    label: string;
    onClick?: (e: React.MouseEvent) => void;
    divider?: boolean;
    component?: ReactNode;
};

type TViewerButtonMenu = {
    items: TViewerButtonItems[];
};

const buttonStyle =
    'flex space-x-2 items-center ' +
    'hover:bg-AAPrimary cursor-pointer p-1 px-3 ' +
    'transition-colors duration-200 ease-in-out';

const MenuButton = ({ icon, label, onClick }: TMenuButtonProps) => {
    return (
        <div className={buttonStyle} onClick={(e) => onClick?.(e)}>
            <div className={'text-lg'}>{icon}</div>
            <div className={'truncate text-xm'}> {label}</div>
        </div>
    );
};

const ViewerButtonMenu = ({ items }: TViewerButtonMenu) => {
    return (
        <div className={'min-w-36'}>
            {items.map((item, index) => {
                return (
                    <div key={index}>
                        {item.component ? (
                            item.component
                        ) : (
                            <MenuButton icon={item.icon} label={item.label} onClick={item.onClick} />
                        )}
                        {item.divider && <hr className={'m-2 border-gray-600'} />}
                    </div>
                );
            })}
        </div>
    );
};

export default ViewerButtonMenu;
export { MenuButton };
