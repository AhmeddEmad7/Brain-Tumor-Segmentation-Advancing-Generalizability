import { useSelector, useDispatch } from 'react-redux';
import React, { useState } from 'react';
import { IStore } from '@/models';
import ReportingTable from '@/ui/library/ReportTable/ReportingTable';
import { createReportThunk, deleteReportbyIdThunk } from '@/features/report/report-actions';
import cropBlack from '@/features/report/components/cropBlack';

import store from '@/redux/store.ts';
import { Types } from '@cornerstonejs/core';
import * as cornerstone from '@cornerstonejs/core';
import DeleteIcon from '@mui/icons-material/Delete';
import classnames from 'classnames';

interface snapshotItem {
    index: number;
    image: string; // base64 data URL
}
const ReportingTab = () => {
    const { selectedStudyReports, currentStudyInstanceUid, renderingEngineId, selectedViewportId } =
        useSelector((store: IStore) => store.viewer);

    const [snapshots, setSnapshots] = useState<snapshotItem[]>([]);
    const dispatch = useDispatch();

    const captureSnapshotFromElement = (viewport: Types.IViewport) => {
        return new Promise<string>((resolve, reject) => {
            // 1. Get the main canvas element (DICOM image)
            const mainCanvas = viewport.element.querySelector('canvas');
            if (!mainCanvas) {
                return reject('No main canvas found in the viewport.');
            }

            // 2. Use the viewport's container as the common reference.
            const container = viewport.element;
            if (!container) {
                return reject('Viewport container not found.');
            }

            // 3. Get the container’s display (CSS) dimensions.
            const { width: displayWidth, height: displayHeight } = container.getBoundingClientRect();

            // 4. Create an offscreen canvas sized to the container’s display dimensions.
            const offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width = displayWidth;
            offscreenCanvas.height = displayHeight;
            const ctx = offscreenCanvas.getContext('2d');
            if (!ctx) {
                return reject('Failed to get 2D context from offscreen canvas.');
            }

            // 5. Draw the main canvas onto the offscreen canvas.
            // Scale from the main canvas’s intrinsic size to the container’s display size.
            ctx.drawImage(
                mainCanvas,
                0,
                0,
                mainCanvas.width,
                mainCanvas.height, // source: intrinsic dimensions
                0,
                0,
                displayWidth,
                displayHeight // destination: display dimensions
            );

            // 6. Get the SVG overlay from the container.
            const svgElement = container.querySelector('svg');
            if (!svgElement) {
                // If there's no SVG, simply output the main canvas image.
                offscreenCanvas.toBlob((blob) => {
                    if (!blob) return reject('Failed to capture canvas.');
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result as string);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                }, 'image/png');
                return;
            }

            // 7. Serialize the SVG overlay.
            const svgData = new XMLSerializer().serializeToString(svgElement);
            const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
            const svgUrl = URL.createObjectURL(svgBlob);

            // 8. Create an image from the SVG blob.
            const svgImg = new Image();
            svgImg.onload = () => {
                // Draw the SVG overlay at (0,0) with the container’s display size.
                ctx.drawImage(svgImg, 0, 0, displayWidth, displayHeight);

                // Free the temporary object URL.
                URL.revokeObjectURL(svgUrl);

                // (Optional) Crop the result if needed.
                const croppedCanvas = cropBlack(offscreenCanvas, 9); // adjust threshold if desired

                // 9. Convert the final offscreen canvas to a data URL.
                croppedCanvas.toBlob((blob) => {
                    if (!blob) {
                        return reject('Failed to capture snapshot.');
                    }
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result as string);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                }, 'image/png');
            };

            svgImg.onerror = () => {
                reject('Failed to load SVG image.');
            };

            svgImg.src = svgUrl;
        });
    };

    // Handler to create a new snapshot and add it to the array
    const handleCreateSnapshot = () => {
        // For this example, assume your viewport container element has the id equal to selectedViewportId.
        // Adjust this logic if your DOM structure differs.
        // const viewportElement = document.getElementById(selectedViewportId);
        const renderingEngine = cornerstone.getRenderingEngine(renderingEngineId);
        if (!renderingEngine) {
            console.error('Rendering engine not found!');
            return;
        }
        const viewport = renderingEngine.getViewport(selectedViewportId);
        viewport.render();
        if (!viewport) {
            console.error('Viewport element not found!');
            return;
        }
        captureSnapshotFromElement(viewport)
            .then((dataUrl: string) => {
                // Add the new snapshot to the snapshots array
                setSnapshots((prevSnapshots) => [
                    {
                        index: prevSnapshots.length, // or use a unique ID if you prefer
                        image: dataUrl
                    },
                    ...prevSnapshots
                ]);
            })
            .catch((error) => console.error('Error capturing snapshot:', error));
    };
    const handleDeleteSnapshot = (index: number) => {
        setSnapshots((prevSnapshots) => prevSnapshots.filter((snapshot) => snapshot.index !== index));
    };
    const renderSnapshotContent = () => {
        if (!snapshots.length) {
            return <p>No snapshots found</p>;
        }
        return (
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'left' }}>
                {snapshots.map((snapshot) => (
                    <div key={snapshot.index} style={{ margin: '5px' }}>
                        <img
                            src={snapshot.image}
                            alt={`Snapshot ${snapshot.index}`}
                            style={{ maxWidth: '100px', display: 'block' }}
                        />
                        <DeleteIcon
                            className={classnames(
                                'w-4 cursor-pointer text-white transition duration-300 hover:text-gray-400'
                            )}
                            onClick={() => handleDeleteSnapshot(snapshot.index)}
                        >
                            Delete
                        </DeleteIcon>
                    </div>
                ))}
            </div>
        );
    };
    const renderContent = (content: string, wordLimit: number) => {
        try {
            const parsedContent = JSON.parse(content);
            let textContent = parsedContent
                .map((item: any) => {
                    if (item.type === 'p') {
                        return item.children.map((child: any) => child.text).join(' ');
                    }
                    return '';
                })
                .join(' ');

            // Limit the number of words displayed
            const words = textContent.split(' ');
            if (words.length > wordLimit) {
                textContent = words.slice(0, wordLimit).join(' ') + '...';
            }

            return textContent;
        } catch (error) {
            return 'No content found';
        }
    };

    const deleteReport = (reportId: number, studyId: string) => {
        store.dispatch(deleteReportbyIdThunk(reportId, studyId));
    };

    const createReport = (studyId: string, navigate: any) => {
        store.dispatch(createReportThunk(studyId, navigate, snapshots));
    };

    return (
        <div className="h-full overflow-y-auto">
            <div className="h-full overflow-y-auto">
                <ReportingTable
                    data={selectedStudyReports}
                    renderContent={renderContent}
                    currentStudyInstanceUid={currentStudyInstanceUid}
                    onDelete={deleteReport}
                    onCreate={createReport}
                />
            </div>
            {/* Button to capture a new snapshot */}
            <button onClick={handleCreateSnapshot}>Create Snapshot</button>

            {/* Show captured snapshots */}
            <div>{renderSnapshotContent()}</div>
        </div>
    );
};

export default ReportingTab;
