// src/components/FileUploader.tsx
// ... (Code from previous response)
// [Your FileUploader.tsx content goes here]
'use client'
import React, { useState } from 'react'
import axios from 'axios'
import { FileText, Upload } from 'lucide-react'

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL

interface FileUploaderProps {
  setJobId: (id: string | null) => void;
  setIsLoading: (loading: boolean) => void;
  setKpiData: (data: any) => void;
}

export default function FileUploader({ setJobId, setIsLoading, setKpiData }: FileUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleAnalysis = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setJobId(null);
    setKpiData(null); 

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Calls the new async endpoint
      const response = await axios.post(`${API_BASE_URL}/analyze/submit-job`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      const { job_id, status, data } = response.data;
      
      if (status === 'complete' && job_id === 'CACHED') {
          // Cached result returned immediately
          setKpiData(data);
          setIsLoading(false);
          setJobId(null);
      } else if (status === 'pending' && job_id) {
          // Trigger polling loop in page.tsx
          setJobId(job_id); 
      } else {
          throw new Error("Job submission failed to return a valid status or ID.");
      }
      
    } catch (error: any) {
      console.error("Submission Failed:", error);
      setIsLoading(false);
      setJobId(null);
      alert(`Upload Failed: Check your file format (CSV/XLSX) or contact support. Error: ${error.response?.data?.detail}`); 
    }
  };

  return (
    <div className="p-4 bg-white rounded-2xl shadow-sm space-y-3">
      <h3 className="font-semibold text-lg">Data Input</h3>
      <div className="flex items-center space-x-2">
        <FileText className="w-5 h-5 text-indigo-600" />
        <span className="text-sm font-medium">{selectedFile ? selectedFile.name : 'No file selected'}</span>
      </div>
      
      <input 
          type="file" 
          onChange={handleFileChange} 
          className="text-sm p-1 border rounded w-full file:mr-4 file:py-1 file:px-2 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100" 
      />
      
      <button 
        onClick={handleAnalysis} 
        disabled={!selectedFile}
        className="w-full flex items-center justify-center px-3 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 transition"
      >
        <Upload className="w-4 h-4 mr-2" />
        Run SmartBrain Analysis
      </button>
      
      <button 
        onClick={() => setJobId("MOCK_COMPLETE_123")} 
        className="w-full px-3 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition text-sm"
      >
        Use Example Dataset (MOCK)
      </button>
    </div>
  );
}