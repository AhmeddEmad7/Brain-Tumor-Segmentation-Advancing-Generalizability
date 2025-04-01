import { Dispatch } from '@reduxjs/toolkit';
import { AxiosUtil } from '@/utilities';
import { studiesSliceActions } from '@features/studies-table/studies-slice.ts';
import { IStoreStudiesSlice, INiftiTableStudy, INiftiStudyData } from '@/models';

const GATEWAY_URL = import.meta.env.VITE_SERVER_URL;

export const fetchNiftiStudiesThunk = () => {
    console.log(`fetching nifti studies from ${GATEWAY_URL}/nifti/query/db/files`);

    return async (dispatch: Dispatch) => {
        const response = await AxiosUtil.sendRequest({
            method: 'GET',
            url: `${GATEWAY_URL}/nifti/query/db/files`
        });
        console.log('response', response);
        if (!response) {
            return;
        }

        // Assuming the response has the shape { files: [ ... ] }
        const mappedStudies: INiftiTableStudy[] = response.files.map((file: any) => ({
            id: file.id,
            fileName: file.file_name,
            projectSub: file.subject,
            category: file.modality,
            session: file.session,
            filePath: file.file_path // save file path for later use (download/view)
        }));

        dispatch(studiesSliceActions.addNiftiStudies(mappedStudies));
    };
};

export const uploadNiftiFileThunk = (payload: {
    file: File;
    subject_num: number;
    session_num: number;
    modality: string;
    subject_age?: string;
    subject_sex?: string;
}) => {
    return async (dispatch: Dispatch) => {
        const formData = new FormData();
        formData.append('file', payload.file);
        formData.append('subject_num', payload.subject_num.toString());
        formData.append('session_num', payload.session_num.toString());
        formData.append('file_type', payload.modality);
        if (payload.subject_age) formData.append('subject_age', payload.subject_age);
        if (payload.subject_sex) formData.append('subject_sex', payload.subject_sex);

        try {
            const response = await AxiosUtil.sendRequest({
                method: 'POST',
                url: `${GATEWAY_URL}/nifti/store/`,
                data: formData,
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            // You can dispatch success actions here if needed
            return response;
        } catch (error: unknown) {
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : typeof error === 'object' &&
                        error !== null &&
                        'response' in error &&
                        error.response &&
                        typeof error.response === 'object'
                      ? (error.response as Record<string, unknown>).data || 'Unknown error'
                      : 'Unknown error';
            console.error('Upload NIfTI Error:', errorMessage);
            // Optionally dispatch error actions here
            throw error;
        }
    };
};

export const fetchNiftiStudyByIdThunk = (studyInstanceUID: string) => {
    return async (dispatch: Dispatch) => {
        const study = await AxiosUtil.sendRequest({
            method: 'GET',
            url: `${GATEWAY_URL}/nifti/studies/${studyInstanceUID}`
        });

        if (!study) {
            return;
        }

        dispatch(studiesSliceActions.setSelectedNiftiStudy(study));
    };
};

export const deleteNiftiThunk = (fileId: string) => {
    return async (dispatch: Dispatch) => {
        try {
            const response = await AxiosUtil.sendRequest({
                method: 'DELETE',
                url: `${GATEWAY_URL}/nifti/delete/db/files/${fileId}`
            });

            if (response) {
                // Refresh the studies list after successful deletion
                await fetchNiftiStudiesThunk()(dispatch);
            }

            return response;
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error('Delete NIfTI Error:', errorMessage);
            throw error;
        }
    };
};
