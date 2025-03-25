import { cn, withRef } from '@udecode/cn';
import { PlateElement, withHOC } from '@udecode/plate-common';
import { ELEMENT_IMAGE, Image, useMediaState } from '@udecode/plate-media';
import { ResizableProvider } from '@udecode/plate-resizable';

import { Caption, CaptionTextarea } from './caption';
import { MediaPopover } from './media-popover';
import { mediaResizeHandleVariants, Resizable, ResizeHandle } from './resizable';

// Force image width to 300px
const fixedWidth = 400;

export const ImageElement = withHOC(
  ResizableProvider,
  withRef<typeof PlateElement>(({ className, children, nodeProps, ...props }, ref) => {
    const { readOnly, focused, selected, align = 'right' } = useMediaState();
    
    return (
      <MediaPopover pluginKey={ELEMENT_IMAGE}>
        <PlateElement
          ref={ref}
          // Add the custom class below:
          className={cn('py-2.5 custom-image-node', className)}
          {...props}
        >
          <figure
            className={cn("group relative m-0 inline-block")}
            contentEditable={false}
            style={{ verticalAlign: 'top' }}
          >
            <Resizable
              align={align}
              options={{
                align,
                readOnly,
                defaultWidth: fixedWidth
              }}
            >
              {/* Left Resize Handle */}
              <ResizeHandle
                options={{ direction: 'left' }}
                className={mediaResizeHandleVariants({ direction: 'left' })}
              />
              <Image
                className={cn(
                  'block cursor-pointer object-cover px-0 rounded-sm',
                  'w-[300px] max-w-[300px]',
                  focused && selected && 'ring-2 ring-ring ring-offset-2'
                )}
                alt=""
                {...nodeProps}
              />
              {/* Right Resize Handle */}
              <ResizeHandle
                options={{ direction: 'right' }}
                className={mediaResizeHandleVariants({ direction: 'right' })}
              />
            </Resizable>
            <Caption align={align} style={{ width: fixedWidth }}>
              <CaptionTextarea placeholder="Write a caption..." readOnly={readOnly} />
            </Caption>
          </figure>
          {children}
        </PlateElement>
      </MediaPopover>
    );
  })
);
