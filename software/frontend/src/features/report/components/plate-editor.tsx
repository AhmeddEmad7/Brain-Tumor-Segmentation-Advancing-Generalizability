'use client';

import React, { useMemo, useRef } from 'react';
import { cn } from '@udecode/cn';
import { Plate } from '@udecode/plate-common';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { Box, Button } from '@mui/material';
import { CursorOverlay } from './plate-ui/cursor-overlay';
import { Editor } from './plate-ui/editor';
import { FixedToolbar } from './plate-ui/fixed-toolbar';
import { FixedToolbarButtons } from './plate-ui/fixed-toolbar-buttons';
import { FloatingToolbar } from './plate-ui/floating-toolbar';
import { FloatingToolbarButtons } from './plate-ui/floating-toolbar-buttons';
import { plugins } from '../lib/plate/plate-plugins';
import SaveButton from './SaveButton';
import { useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { IStore } from '@/models';
import store from '@/redux/store.ts';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import { generatePdfReportThunk } from '@/features/report/report-actions';
import { DicomUtil } from '@/utilities';
interface PlateEditorProps {
    initialReport: any[];
    initialReadOnly: boolean;   
}

export default function PlateEditor({ initialReport, initialReadOnly }: PlateEditorProps) {
    const selectedStudy = useSelector((store: IStore) => store.studies.selectedDicomStudy);
    const containerRef = useRef(null);
    const navigate = useNavigate();

    const key = useMemo(() => {
        if (initialReport) {
            return window.crypto.randomUUID();
        }
    }, [initialReport]);
    const handleGeneratePdf = () => {
        // console.log("selectedStudy.patientBirthDate", selectedStudy.patientBirthDate);
        // console.log("selectedStudy.studyDate", selectedStudy.studyDate);
    const headers =  {
            "patientId": selectedStudy.patientId,
            "patientName": selectedStudy.patientName,
            "studyDate":  DicomUtil.formatDate(selectedStudy.studyDate),
            "modality": "Brain MR",
        }
        store.dispatch(generatePdfReportThunk(selectedStudy.studyInstanceUid,headers ,initialReport));
    }
    return (
        <DndProvider backend={HTML5Backend}>
            <Box
                ref={containerRef}
                sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    p: 2,
                    gap: 2
                }}
            >
                    <Button
                        variant="contained"
                        color="primary"
                        sx={{
                            width: '200px',
                        }}
                        // className="relative top-[10px] left-[960px] z-10 w-[100px] h-[40px] bg-[#152564] hover:opacity-90 hover:bg-[#152564] gap-1 text-white"
                        onClick={handleGeneratePdf}
                        >
                        <span className="text-base font-bold text-white">Export </span>
                        <PictureAsPdfIcon />
                    </Button>

                <Plate id="report" key={key} plugins={plugins} initialValue={initialReport}>
                    
                    <div
                        ref={containerRef}
                        className={cn(
                            'relative',
                            '[&_.slate-start-area-left]:!w-[64px] [&_.slate-start-area-right]:!w-[64px] [&_.slate-start-area-top]:!h-4'
                        )}
                    >
                        
                        <FixedToolbar>
                            <FixedToolbarButtons />
                        </FixedToolbar>

                        <div className=" text-white font-bold rounded-lg shadow-md px-[48px] py-8 overflow-y-auto">
                            <Editor
                                autoFocus
                                focusRing={false}
                                variant="ghost"
                                size="md"
                                initialReadOnly={initialReadOnly}
                            />
                            
                        </div>

                        <CursorOverlay containerRef={containerRef} />
                        
                        <div className="flex justify-end gap-4 mt-2">
                            <Button
                                variant="outlined"
                                color="secondary"
                                onClick={() => {
                                    navigate(-1);
                                }}
                            >
                                Cancel
                            </Button>
                            
                            <SaveButton key={key} />
                        </div>
                    </div>
                </Plate>
            </Box>
        </DndProvider>
    );
}
