import React, { useState } from 'react';
import {
  Table,
  TableHead,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  IconButton,
  Collapse,
  Box,
  Typography,
  Paper
} from '@mui/material';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { ISubject, ISession, ICategory, INiftiFile } from '@models/study.ts';
import { deleteNiftiThunk } from '@features/studies-table/nifti-studies-table/nifti-studies-actions';
import tableColumnHeadings from '@features/studies-table/nifti-studies-table/nifti-table-head-row.ts';
import { useDispatch } from 'react-redux';
import { Link } from 'react-router-dom';
import { MdDeleteForever } from 'react-icons/md';

//
// CATEGORY ROW
//
function CategoryRow({ category }: { category: ICategory }) {
  const [searchValues, setSearchValues] = useState(Array(tableColumnHeadings.length).fill(''));
  const [open, setOpen] = useState(false);
  const handleDelete = (fileID: string) => {
    dispatch(deleteNiftiThunk(fileID));
};
const handleSearchChange = (index: number, value: string) => {
    const newSearchValues = [...searchValues];
    newSearchValues[index] = value;
    setSearchValues(newSearchValues);
};
  const dispatch = useDispatch();



  return (
<>
  <TableRow
    sx={{
      backgroundColor: 'linear-gradient(0deg,rgb(12, 21, 41) 10%,rgb(45, 55, 80) 100%)',
      '&:hover': { backgroundColor: '#212c3d' }
    }}
  >
    <TableCell width="5%">
      <IconButton
        size="small"
        onClick={() => setOpen(!open)}
        sx={{ color: '#FFFFFF' }}
      >
        {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
      </IconButton>
    </TableCell>
    <TableCell sx={{ color: '#FFFFFF', fontWeight: 300, fontSize: '16px' }}>
      {category.categoryName}
    </TableCell>
  </TableRow>

  <TableRow>
    <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
      <Collapse in={open} timeout="auto" unmountOnExit>
        <Box
          margin={1}
          sx={{
            borderRadius: 2,
          }}
        >
          <Table size="small" aria-label="files">
            <TableBody>
              {category.files.map((file: INiftiFile) => (
                <TableRow
                  key={file.id}
                  sx={{ '&:hover': { backgroundColor: '#212c3d' } }}
                >
                  <TableCell sx={{ color: '#FFFFFF' }}>
                    {file.fileName}
                  </TableCell>
                  {/* New cell for the icons, aligned to the right */}
                  <TableCell align="right" sx={{ whiteSpace: 'nowrap' }}>
                    <Link
                      target={'_blank'}
                      to={`/viewer?StudyInstanceUID=${file.filePath}`}
                      style={{ marginRight: '7px' }} // spacing between icons
                    >
                      <VisibilityIcon
                        sx={{
                          '&:hover': {
                            cursor: 'pointer'
                          },
                          color: '#FFFFFF'
                        }}
                      />
                    </Link>
                    <IconButton
                      aria-label="delete"
                      onClick={() => handleDelete(file.id)}
                      sx={{
                        color: 'error.main',
                        '&:hover': { color: 'error.dark' }
                      }}
                    >
                      <MdDeleteForever />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      </Collapse>
    </TableCell>
  </TableRow>
</>

  );
}

//
// SESSION ROW
//
function SessionRow({ session }: { session: ISession }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <TableRow
        sx={{
          backgroundColor: 'linear-gradient(0deg,rgb(12, 21, 41) 10%,rgb(45, 55, 80) 100%)',
          '&:hover': { backgroundColor: '#212c3d' }
        }}
      >
        <TableCell width="5%">
          <IconButton
            size="small"
            onClick={() => setOpen(!open)}
            sx={{ color: '#FFFFFF' }}
          >
            {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
          </IconButton>
        </TableCell>
        <TableCell sx={{ color: '#FFFFFF', fontWeight: 300, fontSize: '15px' }}>
          {session.sessionName}
        </TableCell>
      </TableRow>

      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box
              margin={1}
            >
              <Table size="small" aria-label="categories">
                <TableBody>
                  {session.categories.map((cat) => (
                    <CategoryRow key={cat.categoryName} category={cat} />
                  ))}
                </TableBody>
              </Table>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

//
// SUBJECT ROW
//
function SubjectRow({ subject }: { subject: ISubject }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <TableRow
        sx={{
          backgroundColor: 'linear-gradient(10deg,rgb(12, 21, 41) 100%,rgb(45, 55, 80) 0%)',
          '&:hover': { backgroundColor: '#212c3d' }
        }}
      >
        <TableCell width="5%">
          <IconButton
            size="small"
            onClick={() => setOpen(!open)}
            sx={{ color: '#FFFFFF' }}
          >
            {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
          </IconButton>
        </TableCell>
        <TableCell sx={{ color: '#FFFFFF', fontWeight: 300, fontSize: '1rem' }}>
          {subject.subjectName}
        </TableCell>
      </TableRow>

      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
          <Collapse in={open} timeout="auto" unmountOnExit>
 
              <Table size="small" aria-label="sessions">

                <TableBody >
                  {subject.sessions.map((sess) => (
                    <SessionRow key={sess.sessionName} session={sess} />
                  ))}
                </TableBody>
              </Table>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

//
// MAIN COLLAPSIBLE TABLE COMPONENT
//
interface ICollapsibleTableProps {
  subjects: ISubject[];
}

export default function CollapsibleTable({ subjects }: ICollapsibleTableProps) {
  return (
    <TableContainer
      component={Paper}
      sx={{
        background: 'linear-gradient(20deg,rgb(13, 19, 34) 100%,rgb(45, 55, 80)0%)',
        color: '#FFFFFF',
        borderRadius: 2,
        boxShadow: 3,
        overflowX: 'auto'
      }}
    >
      <Table aria-label="collapsible table">
        <TableHead>
          <TableRow sx={{ backgroundColor: 'linear-gradient(90deg,rgb(19, 27, 46) 0%,rgb(39, 44, 59) 100%)' }}>
            <TableCell sx={{ color: '#FFFFFF' }} />
            <TableCell sx={{ color: '#FFFFFF', fontWeight: 500, fontSize: '1.1rem' }}>
              Subject
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {subjects.map((subject) => (
            <SubjectRow key={subject.subjectName} subject={subject} />
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
