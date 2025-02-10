import classnames from 'classnames';
import { useState, useEffect, ReactNode } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { PanelSection, Tooltip } from '@ui/library';
import { IconProp } from '@fortawesome/fontawesome-svg-core';
import ToolSettings from './ToolSettings.tsx';

export type TOption = {
    id?: string;
    name: string;
    type: string;
    min?: number;
    max?: number;
    value?: number | string;
    step?: number;
    values?: { label: string; value: string }[];
    onChange?: (value: number | string) => void;
    defaultActiveIndex?: number;
    children?: ReactNode | (() => ReactNode);
};

export type TItem = {
    name: string;
    icon: IconProp;
    active?: boolean;
    options?: TOption[];
    onClick?: (name: string) => void;
    disabled?: boolean;
};

export type TAdvancedToolboxProps = {
    title: string;
    items: TItem[];
};

const AdvancedToolbox = ({ title, items }: TAdvancedToolboxProps) => {
    const [activeItemName, setActiveItemName] = useState<string | null>(null);

    useEffect(() => {
        // see if any of the items are active from the outside
        const activeItem = items?.find((item) => item.active);
        setActiveItemName(activeItem ? activeItem.name : null);
    }, [items]);

    const activeItemOptions = items?.find((item) => item.name === activeItemName)?.options;

    return (
        <PanelSection title={title}>
            <div className="flex flex-col rounded-lg p-3 transition-all duration-300 shadow-md" >
                <div className="flex flex-wrap py-2">
                    {items?.map((item) => {
                        return (
                            <Tooltip
                                position="bottom"
                                content={<span className="text-white">{item.name}</span>}
                                key={item.name}
                            >
                                <div
                                    className="ml-2"
                                    onClick={() => {
                                        if (item.disabled) {
                                            return;
                                        }
                                        setActiveItemName(item.name);
                                        item.onClick && item.onClick(item.name);
                                    }}
                                >
                                    <div
                                        className={classnames(
                                            'text-white mt-2 grid h-10 w-10 place-items-center rounded-md',
                                            activeItemName === item.name ? 'bg-[#1E88E5] text-white' : 'bg-[#1B4965]', // Lighter dark blue for active
                                            item.disabled && 'opacity-50',
                                            !item.disabled && 'hover:bg-[#2086C0] cursor-pointer hover:text-white' // Even lighter on hover
                                        )}
                                    >
                                        <FontAwesomeIcon icon={item.icon} />
                                    </div>
                                </div>
                            </Tooltip>
                        );
                    })}
                </div>
                <div className="h-auto px-2">
                    {activeItemOptions && <ToolSettings options={activeItemOptions} />}
                </div>
            </div>
        </PanelSection>
    );
};

export default AdvancedToolbox;