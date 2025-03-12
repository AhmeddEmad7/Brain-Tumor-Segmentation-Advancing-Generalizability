import { INiftiTableColumnHead } from '@models/studies-table.ts';

const tableColumnHeadings: INiftiTableColumnHead[] = [
    {
        displayName: '',
        key: 'checkbox',
        searchable: false
    },
    {
        displayName: 'View',
        key: 'viewer',
        searchable: false
    },
    {
        displayName: 'File Name',
        key: 'fileName',
        searchable: true
    },
    {
        displayName: 'Project Sub',
        key: 'projectSub',
        searchable: true
    },
    {
        displayName: 'Category',
        key: 'category',
        searchable: true
    },
    {
        displayName: 'Session',
        key: 'sequence',
        searchable: true
    }
];

export default tableColumnHeadings;
