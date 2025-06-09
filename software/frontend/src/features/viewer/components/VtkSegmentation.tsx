// VtkSegmentation.tsx
import React, { useEffect, useRef } from 'react';
import vtkImageData from 'vtk.js/Sources/Common/DataModel/ImageData';
import vtkDataArray from 'vtk.js/Sources/Common/Core/DataArray';
import vtkImageMarchingCubes from 'vtk.js/Sources/Filters/General/ImageMarchingCubes';
import vtkActor from 'vtk.js/Sources/Rendering/Core/Actor';
import vtkMapper from 'vtk.js/Sources/Rendering/Core/Mapper';
import vtkFullScreenRenderWindow from 'vtk.js/Sources/Rendering/Misc/FullScreenRenderWindow';
export type SegmentationVolume = {
  scalarData:number[] ;
  dimensions: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
};

type Props = {
  segmentationVolume: SegmentationVolume;
  cameraParams?: { position: number[]; focalPoint: number[] };
};

const VtkSegmentation: React.FC<Props> = ({ segmentationVolume, cameraParams }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    // 1. Wrap data in vtkImageData
    const imageData = vtkImageData.newInstance();
    imageData.setDimensions(segmentationVolume.dimensions);
    imageData.setSpacing(segmentationVolume.spacing);
    imageData.setOrigin(segmentationVolume.origin);
    imageData.getPointData().setScalars(
      vtkDataArray.newInstance({
        name: 'Segmentation',
        numberOfComponents: 1,
        values: segmentationVolume.scalarData,
      })
    );

    // 2. Run Marching Cubes
    const marchingCubes = vtkImageMarchingCubes.newInstance({
      contourValue: 1,
      computeNormals: true,
      mergePoints: true,
    });
    marchingCubes.setInputData(imageData);
    marchingCubes.update();
    const polyData = marchingCubes.getOutputData();

    // 3. Setup VTK renderer embedded in parent div
    const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
      rootContainer: containerRef.current,
      background: [0, 0, 0],
      containerStyle: {
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none', // let Cornerstone handle mouse
      },
    });
    const renderer = fullScreenRenderer.getRenderer();
    const renderWindow = fullScreenRenderer.getRenderWindow();

    // 4. Create mapper & actor
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(polyData);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    renderer.addActor(actor);

    // 5. Sync camera if provided
    if (cameraParams) {
      const vtkCamera = renderer.getActiveCamera();
      vtkCamera.setPosition(...cameraParams.position);
      vtkCamera.setFocalPoint(...cameraParams.focalPoint);
      renderer.resetCameraClippingRange();
    } else {
      renderer.resetCamera();
    }

    // 6. Render the scene
    renderWindow.render();

    // 7. Cleanup on unmount
    return () => {
      fullScreenRenderer.delete();
    };
  }, [segmentationVolume, cameraParams]);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative', // parent must be relative so overlay can fill it
        width: '100%',
        height: '100%',
      }}
    />
  );
};

export default VtkSegmentation;
