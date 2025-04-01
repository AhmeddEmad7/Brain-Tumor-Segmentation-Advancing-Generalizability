export interface IDicomTableStudy {
    reportId: string | null;
    studyId: string;
    studyInstanceUid: string;
    studyDescription: string;
    studyOrthancId: string;
    studyDate: string;
    studyTime: string;
    patientId: string;
    patientName: string;
    accessionNumber: string;
    institutionName: string;
    modality: string;
}

export interface INiftiTableStudy {
    id: string; 
    fileName: string;
    category: string;
    projectSub: string;
    session: string;
    filePath?: string; // optional, for future use (like viewer/download)
}
export interface INiftiFile {
    id: string;
    fileName: string;
    filePath: string;
  }
  
export interface ICategory {
    categoryName : string;
    files : INiftiFile[];
}

export interface ISession {
    sessionName : string;
    categories : ICategory[];
}
export interface ISubject {
    subjectName: string; // e.g. "sub-01"
    sessions: ISession[];
  }
  
export interface IDicomSeriesData {
    studyInstanceUid: string;
    studyOrthancId: string;
    seriesModality: string;
    seriesDescription: string;
    seriesNumber: string;
    seriesId: string;
    seriesInstanceUid: string;
    numberOfInstances: number;
    retrieveUrl: string;
}

export interface IDicomStudyData {
    patientName: string;
    patientId: string;
    patientBirthDate: string;
    patientSex: string;
    studyDate: string;
    studyTime: string;
    studyInstanceUid: string;
    studyTotalInstances: number;
    modality: string;
    series: IDicomSeriesData[];
}

export interface INiftiStudyData {
    patientName: string;
    patientId: string;
    patientBirthDate: string;
    patientSex: string;
    studyDate: string;
    studyTime: string;
    studyInstanceUid: string;
    studyTotalInstances: number;
    modality: string;
    series: IDicomSeriesData[];
}

export interface IStudyReport {
    id: number;
    studyId: string;
    content: string;
}
