import { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Box, Button} from '@mui/material';
import { fetchNiftiStudiesThunk } from '@features/studies-table/nifti-studies-table/nifti-studies-actions.ts';
import { IStore } from '@models/store.ts';
import { TAppDispatch } from '@/redux/store.ts';
import NiftiStudiesTable from '@features/studies-table/nifti-studies-table/NiftiStudiesTable.tsx';
import UploadNiftiModal from '@features/studies-table/nifti-studies-table/UploadNiftiModal';

const NiftiStudies = () => {
    const dispatch = useDispatch<TAppDispatch>();
    const [isAddingNifti, setIsAddingNifti] = useState(false);

    useEffect(() => {
        dispatch(fetchNiftiStudiesThunk());
    }, []);

    const { niftiStudies } = useSelector((store: IStore) => store.studies);

    return (
        <Box className={'flex-col mt-4 space-y-5'}>
              <Button variant="contained" color="secondary" onClick={() => setIsAddingNifti(true)}>
                New NIfTI
            </Button>
            {/* Your studies table component goes here */}
            {isAddingNifti && (
                <UploadNiftiModal isOpen={isAddingNifti} onClose={() => setIsAddingNifti(false)} />
            )}
            <Box className={'h-3/4'}>
                <NiftiStudiesTable data={niftiStudies} />
            </Box>
        </Box>
    );
};

export default NiftiStudies;
