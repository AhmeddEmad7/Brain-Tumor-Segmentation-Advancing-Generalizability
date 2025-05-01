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
import { flattenNiftiData } from './flattenNiftiData';
function StudiesDataTable({ data }: { data: INiftiTableStudy[] }) {
    const [searchValues, setSearchValues] = useState(Array(tableColumnHeadings.length).fill(''));
    const dispatch = useDispatch();
    const theme = useTheme();

    const handleSearchChange = (index: number, value: string) => {
      const newSearchValues = [...searchValues];
      newSearchValues[index] = value;
      setSearchValues(newSearchValues);
    };
  
    const filterRows = () => {
      return data.filter(row =>
        tableColumnHeadings.every((col, i) => {
          if (!col.searchable) return true;
          const searchVal = searchValues[i].toLowerCase();
          const rowVal = String((row as any)[col.key] || '').toLowerCase();
          return rowVal.includes(searchVal);
        })
      );
    };
  
    const filteredRows = filterRows();
  
    return (
      <Box>
        <TableContainer component={Box} className="overflow-auto">
          <Table size="small">
            <TableHead>
              <TableRow>
                {tableColumnHeadings.map((col, i) => (
                  <StyledTableCell key={i}>
                    <Box className="flex items-center">
                      {col.searchable ? (
                        <>
                          <StudiesTableHeaderSearchInput
                            displayName={col.displayName}
                            index={i}
                            onChange={handleSearchChange}
                            theme={theme}
                          />
                          <SearchIcon />
                        </>
                      ) : (
                        col.displayName
                      )}
                    </Box>
                  </StyledTableCell>
                ))}
              </TableRow>
            </TableHead>
  
            <TableBody>
              {filteredRows.map((row) => (
                <StyledTableRow key={row.id}>
                  <StyledTableCell>
                    <Checkbox />
                  </StyledTableCell>
                  <StyledTableCell>
                    <Link to={`/viewer?StudyInstanceUID=${row.filePath}`} target="_blank">
                      <VisibilityIcon sx={{ cursor: 'pointer' }} />
                    </Link>
                  </StyledTableCell>
                  <StyledTableCell>{row.fileName}</StyledTableCell>
                  <StyledTableCell>{row.projectSub}</StyledTableCell>
                  <StyledTableCell>{row.category}</StyledTableCell>
                  <StyledTableCell>{row.session}</StyledTableCell>
                  <StyledTableCell>
                    <IconButton onClick={() => dispatch(deleteNiftiThunk(row.id))}>
                      <MdDeleteForever />
                    </IconButton>
                  </StyledTableCell>
                </StyledTableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  }
  
  
  export default StudiesDataTable;