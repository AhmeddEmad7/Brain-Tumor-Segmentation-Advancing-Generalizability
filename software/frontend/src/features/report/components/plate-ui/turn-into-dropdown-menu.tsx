import { DropdownMenuProps } from '@radix-ui/react-dropdown-menu';
import { ELEMENT_BLOCKQUOTE } from '@udecode/plate-block-quote';
import {
    collapseSelection,
    focusEditor,
    getNodeEntries,
    isBlock,
    toggleNodeType,
    useEditorRef,
    useEditorSelector
} from '@udecode/plate-common';
import {
    ELEMENT_H1,
    ELEMENT_H2,
    ELEMENT_H3,
    ELEMENT_H4,
    ELEMENT_H5,
    ELEMENT_H6
} from '@udecode/plate-heading';
import { ELEMENT_PARAGRAPH } from '@udecode/plate-paragraph';

import { Icons } from '../icons';

import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuLabel,
    DropdownMenuRadioGroup,
    DropdownMenuRadioItem,
    DropdownMenuTrigger,
    useOpenState
} from './dropdown-menu';
import { ToolbarButton } from './toolbar';

const items = [
    {
        value: ELEMENT_PARAGRAPH,
        label: 'Paragraph',
        description: 'Paragraph',
        icon: Icons.paragraph
    },
    {
        value: ELEMENT_H1,
        label: 'Heading 1',
        description: 'Heading 1',
        icon: Icons.h1
    },
    {
        value: ELEMENT_H2,
        label: 'Heading 2',
        description: 'Heading 2',
        icon: Icons.h2
    },
    {
        value: ELEMENT_H3,
        label: 'Heading 3',
        description: 'Heading 3',
        icon: Icons.h3
    },
    {
        value: ELEMENT_H4,
        label: 'Heading 4',
        description: 'Heading 4',
        icon: Icons.h4
    },

    {
        value: ELEMENT_H5,
        label: 'Heading 5',
        description: 'Heading 5',
        icon: Icons.h5
    },

    {
        value: ELEMENT_H6,
        label: 'Heading 6',
        description: 'Heading 6',
        icon: Icons.h6
    },
    {
        value: ELEMENT_BLOCKQUOTE,
        label: 'Quote',
        description: 'Quote (⌘+⇧+.)',
        icon: Icons.blockquote
    },
    {
        value: 'ul',
        label: 'Bulleted list',
        description: 'Bulleted list',
        icon: Icons.ul
    },
    {
        value: 'ol',
        label: 'Numbered list',
        description: 'Numbered list',
        icon: Icons.ol
    }
];

const defaultItem = items.find((item) => item.value === ELEMENT_PARAGRAPH)!;

export function TurnIntoDropdownMenu(props: DropdownMenuProps) {
    const value: string = useEditorSelector((editor) => {
        let initialNodeType: string = ELEMENT_PARAGRAPH;
        let allNodesMatchInitialNodeType = false;
        const codeBlockEntries = getNodeEntries(editor, {
            match: (n) => isBlock(editor, n),
            mode: 'highest'
        });
        const nodes = Array.from(codeBlockEntries);

        if (nodes.length > 0) {
            initialNodeType = nodes[0][0].type as string;
            allNodesMatchInitialNodeType = nodes.every(([node]) => {
                const type: string = (node?.type as string) || ELEMENT_PARAGRAPH;

                return type === initialNodeType;
            });
        }

        return allNodesMatchInitialNodeType ? initialNodeType : ELEMENT_PARAGRAPH;
    }, []);

    const editor = useEditorRef();
    const openState = useOpenState();

    const selectedItem = items.find((item) => item.value === value) ?? defaultItem;
    const { icon: SelectedItemIcon, label: selectedItemLabel } = selectedItem;

    return (
        <DropdownMenu modal={false} {...openState} {...props}>
            <DropdownMenuTrigger asChild>
                <ToolbarButton
                    pressed={openState.open}
                    tooltip="Turn into"
                    isDropdown
                    className="lg:min-w-[130px] text-black"
                >
                    <SelectedItemIcon className="size-5 lg:hidden" />
                    <span className="max-lg:hidden">{selectedItemLabel}</span>
                </ToolbarButton>
            </DropdownMenuTrigger>

            <DropdownMenuContent align="start" className="min-w-0">
                <DropdownMenuLabel className='text-white bg-gray-950' >Turn into</DropdownMenuLabel>

                <DropdownMenuRadioGroup
                    className="flex flex-col gap-0.5 text-white text-lg bg-gray-950"
                    value={value}
                    onValueChange={(type) => {
                        // if (type === 'ul' || type === 'ol') {
                        //   if (settingsStore.get.checkedId(KEY_LIST_STYLE_TYPE)) {
                        //     toggleIndentList(editor, {
                        //       listStyleType: type === 'ul' ? 'disc' : 'decimal',
                        //     });
                        //   } else if (settingsStore.get.checkedId('list')) {
                        //     toggleList(editor, { type });
                        //   }
                        // } else {
                        //   unwrapList(editor);
                        toggleNodeType(editor, { activeType: type });
                        // }

                        collapseSelection(editor);
                        focusEditor(editor);
                    }}
                >
                    {items.map(({ value: itemValue, label, icon: Icon }) => (
                        <DropdownMenuRadioItem key={itemValue} value={itemValue} className="min-w-[180px]">
                            <Icon className="mr-2 size-5" />
                            {label}
                        </DropdownMenuRadioItem>
                    ))}
                </DropdownMenuRadioGroup>
            </DropdownMenuContent>
        </DropdownMenu>
    );
}
