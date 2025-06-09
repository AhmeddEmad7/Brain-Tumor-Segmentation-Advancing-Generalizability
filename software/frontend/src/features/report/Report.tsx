import { ThemeProvider as NextThemesProvider } from 'next-themes';
import PlateEditor from './components/plate-editor';
import { TooltipProvider } from './components/plate-ui/tooltip';
import { Box } from '@mui/material';
import { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { TAppDispatch } from '@/redux/store';
import { useParams } from 'react-router-dom';
import { fetchDicomStudyByIdThunk } from '@features/studies-table/dicom-studies-table/dicom-studies-actions';
import { fetchStudyReportByIdThunk } from '@features/report/report-actions';
import { IStore } from '@/models';
import { ELEMENT_IMAGE } from '@udecode/plate-media';
import { ELEMENT_PARAGRAPH } from '@udecode/plate-paragraph';


export default function Report() {
  const dispatch = useDispatch<TAppDispatch>();
  const { reportId, studyId } = useParams();
  const [initialReport, setInitialReport] = useState<any[]>([]);

  useEffect(() => {
    if (studyId) {
      dispatch(fetchDicomStudyByIdThunk(studyId));
      dispatch(fetchStudyReportByIdThunk(studyId));
    }
  }, [studyId]);

  const selectedStudyReports = useSelector(
    (store: IStore) => store.viewer.selectedStudyReports || []
  );

  useEffect(() => {
    if (selectedStudyReports && selectedStudyReports.length > 0) {
      const selectedStudyReport = selectedStudyReports.find(
        (report) => String(report.id) === reportId
      );
      if (selectedStudyReport) {
        setInitialReport(JSON.parse(selectedStudyReport.content));
      }
    } else {
      setInitialReport([]);
    }
  }, [selectedStudyReports, reportId]);

  let snapshotsElement = null;
  let reportContent = initialReport;
  if (initialReport.length > 0 && initialReport[0].type === 'images') {
    snapshotsElement = initialReport[0];
    // Remove the snapshots element so PlateEditor only gets the rest of the content
    reportContent = initialReport.slice(1);
  }
  
  // Only map if snapshotsElement exists; otherwise, use an empty array.
  const imageNodes = snapshotsElement ? snapshotsElement.images.map((img: string) => ({
    type: ELEMENT_IMAGE,
    url: img, // the image URL or Base64 string
    children: [{ text: '' }] // Slate requires a children array
  })) : [];
  
  const galleryParagraph = {
    type: ELEMENT_PARAGRAPH,
    children: [...imageNodes, { text: '' }]
};
  
  const newReportContent = [galleryParagraph, ...reportContent];
  return (
    <div>
      <NextThemesProvider attribute="class" defaultTheme="dark" enableSystem={false}>
        <TooltipProvider>
          <Box className="flex-col mt-4 space-y-5">
            <Box className="h-3/4">
              {/* Pass the rest of the report content to the PlateEditor */}
              <PlateEditor initialReadOnly={true} initialReport={newReportContent} />
            </Box>
          </Box>
        </TooltipProvider>
      </NextThemesProvider>
    </div>
  );
}
