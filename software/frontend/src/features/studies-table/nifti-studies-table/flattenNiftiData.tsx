
interface INiftiFlatRow {
    fileName: string;
    projectSub: string;
    session: string;
    category: string;
    filePath: string;
    id: string;
  }
  
  export function flattenNiftiData(subjects: ISubject[]): INiftiFlatRow[] {
    const flatRows: INiftiFlatRow[] = [];
  
    subjects.forEach(subject => {
      if (!subject.sessions) return;
  
      subject.sessions.forEach(session => {
        if (!session.categories) return;
  
        session.categories.forEach(category => {
          if (!category.files) return;
  
          category.files.forEach(file => {
            flatRows.push({
              fileName: file.fileName,
              projectSub: subject.subjectName,
              session: session.sessionName,
              category: category.categoryName,
              filePath: file.filePath,
              id: file.id
            });
          });
        });
      });
    });
  
    return flatRows;
  }
  