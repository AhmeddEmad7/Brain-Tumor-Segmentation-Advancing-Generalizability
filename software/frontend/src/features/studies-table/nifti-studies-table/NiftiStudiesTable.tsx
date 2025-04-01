import { useState } from 'react';
import { Link } from 'react-router-dom';
import { INiftiTableStudy } from '@/models';
import { MdDeleteForever } from 'react-icons/md';
import { deleteNiftiThunk } from '@features/studies-table/nifti-studies-table/nifti-studies-actions';
import {
    StyledTableCell,
    StyledTableRow
} from '@features/studies-table/components/StyledTableComponents.tsx';
import StudiesTableHeaderSearchInput from '@features/studies-table/components/StudiesTableHeaderSearchInput.tsx';
import tableColumnHeadings from '@features/studies-table/nifti-studies-table/nifti-table-head-row.ts';
import {
    Table,
    TableBody,
    TableContainer,
    TableHead,
    TableRow,
    Box,
    useTheme,
    Checkbox,
    IconButton
} from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import VisibilityIcon from '@mui/icons-material/Visibility';
import SearchIcon from '@mui/icons-material/Search';
import { groupDataBySubjectSessionCategory } from '@features/studies-table/nifti-studies-table/groupDataBysubject';
import CollapsibleTable from '@features/studies-table/nifti-studies-table/nifti-collapsibleTable.tsx';
function StudiesDataTable({ data }: { data: INiftiTableStudy[] }) {
    // 1) Group your flat data into hierarchical data
    const subjects = groupDataBySubjectSessionCategory(data);
  
    // 2) Pass that into the CollapsibleTable
    return <CollapsibleTable subjects={subjects} />;
  }
  
  export default StudiesDataTable;