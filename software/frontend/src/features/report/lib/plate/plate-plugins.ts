import { withProps } from '@udecode/cn';
import { createAlignPlugin } from '@udecode/plate-alignment';
import { createAutoformatPlugin } from '@udecode/plate-autoformat';
import {
  createBoldPlugin,
  createCodePlugin,
  createItalicPlugin,
  createStrikethroughPlugin,
  createSubscriptPlugin,
  createSuperscriptPlugin,
  createUnderlinePlugin,
  MARK_BOLD,
  MARK_ITALIC,
  MARK_STRIKETHROUGH,
  MARK_SUBSCRIPT,
  MARK_SUPERSCRIPT,
  MARK_UNDERLINE
} from '@udecode/plate-basic-marks';
import { createBlockquotePlugin, ELEMENT_BLOCKQUOTE } from '@udecode/plate-block-quote';
import { createExitBreakPlugin, createSoftBreakPlugin } from '@udecode/plate-break';
import { createCaptionPlugin } from '@udecode/plate-caption';
import { createComboboxPlugin } from '@udecode/plate-combobox';
import {
  createPlugins,
  isBlockAboveEmpty,
  isSelectionAtBlockStart,
  PlateElement,
  PlateLeaf,
  RenderAfterEditable,
  someNode
} from '@udecode/plate-common';
import { createDndPlugin } from '@udecode/plate-dnd';
import {
  createFontBackgroundColorPlugin,
  createFontColorPlugin,
  createFontSizePlugin
} from '@udecode/plate-font';
import {
  createHeadingPlugin,
  ELEMENT_H1,
  ELEMENT_H2,
  ELEMENT_H3,
  ELEMENT_H4,
  ELEMENT_H5,
  ELEMENT_H6,
  KEYS_HEADING
} from '@udecode/plate-heading';
import { createHighlightPlugin, MARK_HIGHLIGHT } from '@udecode/plate-highlight';
import { createHorizontalRulePlugin, ELEMENT_HR } from '@udecode/plate-horizontal-rule';
import { createIndentPlugin } from '@udecode/plate-indent';
import { createIndentListPlugin, KEY_LIST_STYLE_TYPE } from '@udecode/plate-indent-list';
import { createJuicePlugin } from '@udecode/plate-juice';
import { createKbdPlugin, MARK_KBD } from '@udecode/plate-kbd';
import { createLineHeightPlugin } from '@udecode/plate-line-height';
import { createLinkPlugin, ELEMENT_LINK } from '@udecode/plate-link';
import { ELEMENT_LI, ELEMENT_OL, ELEMENT_UL } from '@udecode/plate-list';
import {
  createImagePlugin,
  createMediaEmbedPlugin,
  ELEMENT_IMAGE,
  ELEMENT_MEDIA_EMBED
} from '@udecode/plate-media';
import { createNodeIdPlugin } from '@udecode/plate-node-id';
import { createParagraphPlugin, ELEMENT_PARAGRAPH } from '@udecode/plate-paragraph';
import { createResetNodePlugin } from '@udecode/plate-reset-node';
import { createSelectOnBackspacePlugin } from '@udecode/plate-select';
import { createBlockSelectionPlugin } from '@udecode/plate-selection';
import { createDeserializeDocxPlugin } from '@udecode/plate-serializer-docx';
import { createDeserializeMdPlugin } from '@udecode/plate-serializer-md';
import { createTabbablePlugin } from '@udecode/plate-tabbable';
import { createTablePlugin, ELEMENT_TABLE, ELEMENT_TD, ELEMENT_TH, ELEMENT_TR } from '@udecode/plate-table';
import { createTrailingBlockPlugin } from '@udecode/plate-trailing-block';

import { autoformatPlugin } from './autoformatPlugin';
import { dragOverCursorPlugin } from './dragOverCursorPlugin';
import { BlockquoteElement } from '../../components/plate-ui/blockquote-element';
import { HeadingElement } from '../../components/plate-ui/heading-element';
import { HighlightLeaf } from '../../components/plate-ui/highlight-leaf';
import { HrElement } from '../../components/plate-ui/hr-element';
import { ImageElement } from '../../components/plate-ui/image-element';
import { KbdLeaf } from '../../components/plate-ui/kbd-leaf';
import { LinkElement } from '../../components/plate-ui/link-element';
import { LinkFloatingToolbar } from '../../components/plate-ui/link-floating-toolbar';
import { ListElement } from '../../components/plate-ui/list-element';
import { ParagraphElement } from '../../components/plate-ui/paragraph-element';
import { withPlaceholders } from '../../components/plate-ui/placeholder';
import { TableCellElement, TableCellHeaderElement } from '../../components/plate-ui/table-cell-element';
import { TableElement } from '../../components/plate-ui/table-element';
import { TableRowElement } from '../../components/plate-ui/table-row-element';
import { withDraggables } from '../../components/plate-ui/with-draggables';
import { TabbableElement } from '../../components/tabbable-element';

const resetBlockTypesCommonRule = {
  types: [ELEMENT_BLOCKQUOTE],
  defaultType: ELEMENT_PARAGRAPH
};

export const plugins = createPlugins(
  [
    // Nodes
    createParagraphPlugin(),
    createHeadingPlugin(),
    createBlockquotePlugin(),
    createHorizontalRulePlugin(),
    createLinkPlugin({
      renderAfterEditable: LinkFloatingToolbar as RenderAfterEditable
    }),
    createImagePlugin(),
    createMediaEmbedPlugin(),
    createCaptionPlugin({
      options: { pluginKeys: [ELEMENT_IMAGE, ELEMENT_MEDIA_EMBED] }
    }),
    createTablePlugin(),

    // Marks
    createBoldPlugin(),
    createItalicPlugin(),
    createUnderlinePlugin(),
    createStrikethroughPlugin(),
    createCodePlugin(),
    createSubscriptPlugin(),
    createSuperscriptPlugin(),
    createFontColorPlugin(),
    createFontBackgroundColorPlugin(),
    createFontSizePlugin(),
    createHighlightPlugin(),
    createKbdPlugin(),

    // Block Style
    createAlignPlugin({
      inject: {
        props: {
          validTypes: [ELEMENT_PARAGRAPH, ELEMENT_H1, ELEMENT_H2, ELEMENT_H3]
        }
      }
    }),
    createIndentPlugin({
      inject: {
        props: {
          validTypes: [ELEMENT_PARAGRAPH, ELEMENT_H1, ELEMENT_H2, ELEMENT_H3, ELEMENT_BLOCKQUOTE]
        }
      }
    }),
    createIndentListPlugin({
      inject: {
        props: {
          validTypes: [ELEMENT_PARAGRAPH, ELEMENT_H1, ELEMENT_H2, ELEMENT_H3, ELEMENT_BLOCKQUOTE]
        }
      }
    }),
    createLineHeightPlugin({
      inject: {
        props: {
          defaultNodeValue: 1.5,
          validNodeValues: [1, 1.2, 1.5, 2, 3],
          validTypes: [ELEMENT_PARAGRAPH, ELEMENT_H1, ELEMENT_H2, ELEMENT_H3]
        }
      }
    }),

    // Functionality
    createAutoformatPlugin(autoformatPlugin),
    createBlockSelectionPlugin({
      options: {
        sizes: {
          top: 0,
          bottom: 0
        }
      }
    }),
    createComboboxPlugin(),
    createDndPlugin({
      options: { enableScroller: true }
    }),
    createExitBreakPlugin({
      options: {
        rules: [
          {
            hotkey: 'mod+enter'
          },
          {
            hotkey: 'mod+shift+enter',
            before: true
          },
          {
            hotkey: 'enter',
            query: {
              start: true,
              end: true,
              allow: KEYS_HEADING
            },
            relative: true,
            level: 1
          }
        ]
      }
    }),
    createNodeIdPlugin(),
    createResetNodePlugin({
      options: {
        rules: [
          {
            ...resetBlockTypesCommonRule,
            hotkey: 'Enter',
            predicate: isBlockAboveEmpty
          },
          {
            ...resetBlockTypesCommonRule,
            hotkey: 'Backspace',
            predicate: isSelectionAtBlockStart
          }
        ]
      }
    }),
    createSelectOnBackspacePlugin({
      options: {
        query: {
          allow: [ELEMENT_IMAGE, ELEMENT_HR]
        }
      }
    }),
    createSoftBreakPlugin({
      options: {
        rules: [
          { hotkey: 'shift+enter' },
          {
            hotkey: 'enter',
            query: {
              allow: [ELEMENT_BLOCKQUOTE, ELEMENT_TD]
            }
          }
        ]
      }
    }),
    createTabbablePlugin({
      options: {
        query: (editor: any) => {
          if (isSelectionAtBlockStart(editor)) return false;
          return !someNode(editor, {
            match: (n) => {
              return !!(
                n.type &&
                ([ELEMENT_TABLE, ELEMENT_LI].includes(n.type as string) ||
                  n[KEY_LIST_STYLE_TYPE])
              );
            }
          });
        }
      },
      plugins: [
        {
          key: 'tabbable_element',
          isElement: true,
          isVoid: true,
          component: TabbableElement
        }
      ]
    }),
    createTrailingBlockPlugin({
      options: { type: ELEMENT_PARAGRAPH }
    }),
    dragOverCursorPlugin,
    createDeserializeDocxPlugin(),
    createDeserializeMdPlugin(),
    createJuicePlugin()
  ],
  {
    components: withDraggables(
      withPlaceholders({
        [ELEMENT_BLOCKQUOTE]: BlockquoteElement,
        [ELEMENT_HR]: HrElement,
        [ELEMENT_H1]: withProps(HeadingElement, { variant: 'h1', className:  'text-slate-400  font-bold mt-4 mb-2'  }),
        [ELEMENT_H2]: withProps(HeadingElement, { variant: 'h2', className: 'text-slate-400  text-3xl font-bold mt-4 mb-2' }),
        [ELEMENT_H3]: withProps(HeadingElement, { variant: 'h3', className:  'text-slate-400  font-bold mt-4 mb-2'  }),
        [ELEMENT_H4]: withProps(HeadingElement, { variant: 'h4', className: 'text-slate-400   font-bold mt-4 mb-2'  }),
        [ELEMENT_H5]: withProps(HeadingElement, { variant: 'h5', className:  'text-slate-400   font-bold mt-4 mb-2' }),
        [ELEMENT_H6]: withProps(HeadingElement, { variant: 'h6', className:  'text-slate-400  font-bold mt-4 mb-2'  }),
        [ELEMENT_IMAGE]: ImageElement,
        [ELEMENT_LI]: withProps(PlateElement, { as: 'li' }),
        [ELEMENT_LINK]: LinkElement,
        [ELEMENT_UL]: withProps(ListElement, { variant: 'ul' }),
        [ELEMENT_OL]: withProps(ListElement, { variant: 'ol' }),
        [ELEMENT_PARAGRAPH]: ParagraphElement,
        [ELEMENT_TABLE]: TableElement,
        [ELEMENT_TD]: TableCellElement,
        [ELEMENT_TH]: TableCellHeaderElement,
        [ELEMENT_TR]: TableRowElement,
        [MARK_BOLD]: withProps(PlateLeaf, { as: 'strong' }),
        [MARK_HIGHLIGHT]: HighlightLeaf,
        [MARK_ITALIC]: withProps(PlateLeaf, { as: 'em' }),
        [MARK_KBD]: KbdLeaf,
        [MARK_STRIKETHROUGH]: withProps(PlateLeaf, { as: 's' }),
        [MARK_SUBSCRIPT]: withProps(PlateLeaf, { as: 'sub' }),
        [MARK_SUPERSCRIPT]: withProps(PlateLeaf, { as: 'sup' }),
        [MARK_UNDERLINE]: withProps(PlateLeaf, { as: 'u' }),
        ['note']: withProps(PlateElement, {
          as: 'p',
          className: 'text-slate-400 font-bold'
        })
      })
    )
  }
);
