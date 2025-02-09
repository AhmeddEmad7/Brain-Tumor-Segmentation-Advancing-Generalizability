import { ReactElement, ReactNode, useState } from 'react';
import { faArrowRightFromBracket } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

interface TSidePanelProps {
    children: ReactNode;
    title: string;
    side?: 'left' | 'right';
    tabs?: any[];
    headerComponent?: ReactElement;
}

const SidePanel = ({ children, title, headerComponent }: TSidePanelProps) => {
    const [isOpen, setIsOpen] = useState(false);

    const togglePanel = () => {
        setIsOpen(!isOpen);
    };

    return (
        <div
            className={`h-full transition-all duration-300 shadow-lg ${isOpen ? 'w-80' : 'w-10'} rounded-lg bg-gradient-to-r from-gray-900 to-gray-700 text-white`}
        >
            <div className="bg-gray-800 flex items-center justify-between p-3 rounded-t-lg">
                <button className="p-2 text-white hover:bg-gray-700 rounded-md" onClick={togglePanel}>
                    <FontAwesomeIcon
                        icon={faArrowRightFromBracket}
                        className={`transition-transform duration-300 ${isOpen ? '' : 'rotate-180'}`}
                    />
                </button>
                {headerComponent ? (
                    <div className="text-lg font-semibold">{headerComponent}</div>
                ) : (
                    <div className="text-lg font-semibold">{title}</div>
                )}
                <div className="opacity-0">hidd</div>
            </div>
            <div className="p-4">{isOpen && children}</div>
        </div>
    );
};

export default SidePanel;
