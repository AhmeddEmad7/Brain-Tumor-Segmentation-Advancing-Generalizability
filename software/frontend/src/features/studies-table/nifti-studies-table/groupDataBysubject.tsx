import { INiftiTableStudy,ISubject } from '@/models';
export function groupDataBySubjectSessionCategory(
    flatData :INiftiTableStudy[] 
):ISubject[]{
    const subjectsMap = new Map<string, ISubject>();
    flatData.forEach((item) => {
    const subjectName = item.projectSub;
    const sessionName = item.session;
    console.log('item', item.session);
    const categoryName = item.category;
    const fileName = item.fileName;
    const filePath = item.filePath;
    const fileId = item.id;

    if (!subjectsMap.has(subjectName)) {
        subjectsMap.set(subjectName, { subjectName, sessions: [] });
    }
    const subject = subjectsMap.get(subjectName)!;

    let session = subject.sessions.find((S)=> S.sessionName === sessionName);
    if (!session) {
        session = { sessionName, categories: [] };
        subject.sessions.push(session);
    }

    let category = session.categories.find((C)=> C.categoryName === categoryName);
    if (!category) {
        category = { categoryName, files: [] };
        session.categories.push(category);
    }
    category.files.push({ id: fileId, fileName, filePath });
    });
    // Convert the map values to an array
    console.log('subjectsMap', subjectsMap);
    console.log('subjectsMap values', Array.from(subjectsMap.values()));
    return Array.from(subjectsMap.values())
}
