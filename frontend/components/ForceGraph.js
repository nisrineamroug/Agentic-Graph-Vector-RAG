import React, { forwardRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const ForceGraphWrapper = forwardRef((props, ref) => {
  return <ForceGraph2D {...props} ref={ref} />;
});

ForceGraphWrapper.displayName = 'ForceGraphWrapper';

export default ForceGraphWrapper;
