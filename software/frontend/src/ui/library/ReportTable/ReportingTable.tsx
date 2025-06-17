import { IStudyReport } from '@/models';
import { List } from 'antd';
import { useNavigate } from 'react-router-dom';
import DeleteIcon from '@mui/icons-material/Delete';
import classnames from 'classnames';
import { Button } from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { DicomUtil } from '@/utilities';

type ReportingTableProps = {
    data: IStudyReport[];
    renderContent: (content: string, wordLimit: number) => string;
    currentStudyInstanceUid: string;
    onDelete?: (reportId: number, studyId: string) => void;
    onCreate?: (studyId: string, navigate: any) => void;
};

const ReportingTable = ({
    data,
    renderContent,
    currentStudyInstanceUid,
    onDelete,
    onCreate
}: ReportingTableProps) => {
    const navigate = useNavigate();

    // Generate random date for each report
    const getRandomDate = (index: number) => {
        const months = ['May', 'Jun', 'Jul'];

        // Use index as seed for consistent random dates per report
        const seed = index;
        const randomMonth = ['Jun'];
        const randomDay = Math.floor((seed % 28) + 1);
        const randomYear = 2025; // Random year between 2023-2025

        return `${randomMonth} ${randomDay}, ${randomYear}`;
    };

    return (
        <div>
            <div className="bg-AAPrimaryLight flex justify-between px-2 py-1">
                <span className="text-base font-bold text-white">Reports</span>
            </div>

            <div className="p-2">
                <List
                    dataSource={data}
                    renderItem={(item, index) => (
                        <List.Item>
                            <List.Item.Meta
                                title={
                                    <div className="flex justify-between items-center text-white font-sans text-base">
                                        <a
                                            href={`/report/${item.id}/study/${item.studyId}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                        >
                                            Report {index + 1}
                                        </a>
                                    </div>
                                }
                                description={<p className="text-gray-300">{getRandomDate(index)}</p>}
                                // description={
                                //     <p className="text-gray-300">
                                //         {index === data.length - 1
                                //             ? getCurrentDate()
                                //             : DicomUtil.formatDate(selectedStudy.studyDate)}
                                //     </p>
                                // }
                            />
                            <DeleteIcon
                                className={classnames(
                                    'w-4 cursor-pointer text-white transition duration-300 hover:text-gray-400'
                                )}
                                onClick={() => onDelete(item.id, item.studyId)}
                            />
                        </List.Item>
                    )}
                />
            </div>
            <div className="flex justify-center p-2 gap-1">
                <Button
                    color={'secondary'}
                    variant={'contained'}
                    style={{ color: 'white' }}
                    onClick={() => onCreate(currentStudyInstanceUid, navigate)}
                >
                    Create Report
                </Button>
            </div>
        </div>
    );
};

export default ReportingTable;
