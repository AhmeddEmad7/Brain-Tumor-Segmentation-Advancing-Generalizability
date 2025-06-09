import { Dispatch } from '@reduxjs/toolkit';
import { AxiosUtil } from '@/utilities';
import { uiSliceActions } from '@ui/ui-slice.ts';
import { viewerSliceActions } from '@features/viewer/viewer-slice';
import store from '@/redux/store';
import { useNavigate } from 'react-router-dom';
import { urlToDataURL } from '../report/components/toBase64';
/**
 * The URL for the reporting API.
 */
// const REPORTING_API_URL = import.meta.env.VITE_REPORTING_API_URL;
 const REPORTING_API_URL = "http://localhost:9000";

const templateReportContent = `[
    {"id":"1","children":[{"text":"{Doctor's info}\\n"}],"type":"p"},
    {"children":[{"children":[{"children":[{"children":[{"text":"Patient's name: {patientName}"}],"type":"p","id":"6tjqr"}],"type":"td","id":"4nkmv"},
    {"children":[{"children":[{"text":"Branch: {branchInstitution}"}],"type":"p","id":"e82lq"}],"type":"td","id":"e7917"}],"type":"tr","id":"72zj3"},
    {"children":[{"children":[{"children":[{"text":"Date of Examination: {examinationDate}"}],"type":"p","id":"hdvz3"}],"type":"td","id":"f6hox"},
    {"children":[{"children":[{"text":"{Body parts}"}],"type":"p","id":"e9cb4"}],"type":"td","id":"s4vob"}],"type":"tr","id":"mqt5s"}],"type":"table","id":"j2bl1"},
    {"children":[{"text":"CLINICAL HISTORY: {clinicalHistory} \\n\\nCOMPARISON: {comparison}\\n\\nTECHNIQUE: {technique}\\n\\n FINDINGS: {findings}\\n\\nIMPRESSION: {impression}\\n"}],"type":"p","id":"flr32"}
]`;

/**
 * Creates a new report for the specified study.
 *
 * @param {string} studyInstanceUid - The study instance UID.
 * @param {string} reportContent - The report content.
//  */
function formatTextToSlateBlocks(text: string) {
    const blocks: any[] = [];
    const lines = text.split('\n');

    lines.forEach((line) => {
        if (
            line.startsWith("Findings:") ||
            line.startsWith("Composition analysis:") ||
            line.startsWith("Quantitative analysis:") ||
            line.startsWith("Impression*:") ||
            line.startsWith("Likely Diagnosis*:") ||
            line.startsWith("Recommendations*:")||
            line.startsWith("* These sections are preliminar")
        ) {
            blocks.push({
                id: crypto.randomUUID(),
                type: "h2",
                children: [{ text: line }]
            });
        } 
        else {
            blocks.push({
                id: crypto.randomUUID(),
                type: "p",
                children: [{ text: "    " + line }]
            });
        }
    });

    return blocks;
}

export const generatePdfReportThunk = (
    studyInstanceUid: string,
    header: {
      patientName: string,
      patientId: string;
      modality: string;
    },
    blocks: any[]
  ) => {
    return async (dispatch: Dispatch) => {
      try {
        const blocksCopy = [...blocks];
        const imageBlockIndex = blocksCopy.findIndex((b) => b.type === 'images');
  
        if (imageBlockIndex !== -1) {
          const imageBlock = blocksCopy[imageBlockIndex];
  
          const dataURIs = await Promise.all(
            imageBlock.images.map((imgUrl: string) => urlToDataURL(imgUrl))
          );
  
          blocksCopy[imageBlockIndex] = {
            ...imageBlock,
            data: dataURIs,
          };
          delete blocksCopy[imageBlockIndex].images;
        }
        console.log('blocksCopy', blocksCopy);
        console.log('JSON blocksCopy', JSON.stringify(blocksCopy));
        const response = await AxiosUtil.sendRequest({
          method: 'POST',
          url: `${REPORTING_API_URL}/report/generate-pdf`,
          data: {
            studyId: studyInstanceUid,
            header: header,
            content: JSON.stringify(blocksCopy),
          },
        });
  
        if (response?.code === 200 ) {
          dispatch(
            uiSliceActions.setNotification({
              type: 'success',
              content: '📄 PDF has been generated and saved on the server!',
            })
          );
        } else {
          throw new Error('Backend response status not success');
        }
      } catch (err) {
        console.error('PDF generation error:', err);
        dispatch(
          uiSliceActions.setNotification({
            type: 'error',
            content: '❌ Failed to generate PDF!',
          })
        );
      }
    };
  };
  
export const createReportThunk = (studyInstanceUid: string, navigate: any, snapshots: any) => {
    return async (dispatch: Dispatch) => {
        // 1. Try to get the report content from Redis
        const redisRes = await AxiosUtil.sendRequest({
            method: 'GET',
            url: `${REPORTING_API_URL}/report/redis/${studyInstanceUid}`,
        });

        let reportContent: any;
        console.log('redisRes', redisRes);
        console.log('studyInstanceUid', studyInstanceUid);
 if (redisRes && typeof redisRes.content === 'string' && redisRes.content.trim().length > 0) {
    console.log('✅ Found Redis report. Formatting...');
    reportContent = formatTextToSlateBlocks(redisRes.content);
} else {
    console.warn('⚠️ Redis is empty or error occurred. Using template.');
    reportContent = JSON.parse(templateReportContent);
}

        // 2. Add snapshot element (images)
        const snapshotsElements = {
            id: "snapshots",
            type: "images",
            images: snapshots.map((s: any) => s.image)
        };

        const finalContent = [snapshotsElements, ...reportContent];
        const finalContentString = JSON.stringify(finalContent);

        // 3. Save to the database via the backend API
        const res = await AxiosUtil.sendRequest({
            method: 'POST',
            url: `${REPORTING_API_URL}/report/`,
            data: {
                studyId: studyInstanceUid,
                content: finalContentString
            }
        });

        if (!res) return;

        store.dispatch(fetchStudyReportByIdThunk(studyInstanceUid));

        dispatch(
            uiSliceActions.setNotification({
                type: 'success',
                content: 'Report has been created successfully!'
            })
        );

        navigate(`/report/${res.data.result.id}/study/${studyInstanceUid}`);
    };
};


/**
 * Updates the report content for the specified study.
 *
 * @param {string} studyInstanceUid - The study instance UID.
 * @param {string} reportContent - The new report content.
 */
export const updateReport = (reportId: number, studyId: string, reportContent: string) => {
    return async (dispatch: Dispatch) => {
        const res = await AxiosUtil.sendRequest({
            method: 'PUT',
            url: `${REPORTING_API_URL}/report/${reportId}`,
            data: {
                studyId: studyId,
                content: reportContent
            }
        });

        if (!res) {
            return;
        }

        dispatch(
            uiSliceActions.setNotification({
                type: 'success',
                content: 'Report has been updated successfully!'
            })
        );
    };
};

/**
 * Fetches the report for the specified study.
 *
 * @param {string} studyInstanceUID - The study instance UID.
 */
export const fetchStudyReportByIdThunk = (studyInstanceUID: string) => {
    return async (dispatch: Dispatch) => {
        const report = await AxiosUtil.sendRequest({
            method: 'GET',
            url: `${REPORTING_API_URL}/report/${studyInstanceUID}`
        });

        if (!report) {
            return;
        }

        dispatch(viewerSliceActions.setSelectedStudyReports(report.result));
    };
};

/**
 * Deletes the report for the specified study.
 *
 * @param {string} reportId - The report ID.
 */
export const deleteReportbyIdThunk = (reportId: number, studyId: string) => {
    return async (dispatch: Dispatch) => {
        const result = await AxiosUtil.sendRequest({
            method: 'DELETE',
            url: `${REPORTING_API_URL}/report/${reportId}/study/${studyId}`
        });

        if (!result) {
            return;
        }

        store.dispatch(fetchStudyReportByIdThunk(studyId));
        dispatch(
            uiSliceActions.setNotification({
                type: 'success',
                content: 'Report has been deleted successfully!'
            })
        );
    };
};
