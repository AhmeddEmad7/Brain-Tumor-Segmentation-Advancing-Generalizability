import React from 'react';
import { PlateElementProps } from '@udecode/plate-common';

/**
 * A custom paragraph element that checks if its children are “empty” text
 * (i.e. the paragraph only contains image nodes) and if so, renders its children in a flex row.
 */
export const ParagraphElement = (props: PlateElementProps) => {
  const { attributes, children, element } = props;

  // Check if the paragraph contains any non-empty text.
  // If it doesn't, we'll assume it’s a gallery.
  const hasText = React.Children.toArray(children).some(child => {
    if (typeof child === 'string') {
      return child.trim().length > 0;
    }
    // Optionally, if child is a React element, you can check for specific props or types.
    return false;
  });

  const style = !hasText
    ? { display: 'flex', flexWrap: 'wrap', gap: '16px' }
    : {};

  return (
    <p {...attributes} style={style}>
      {children}
    </p>
  );
};
