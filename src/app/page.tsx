// src/app/page.tsx (MODIFIED FOR ASYNC POLLING)
'use client'

import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import ChatConsole from '@/components/ChatConsole'
import KpiCard from '@/components/KpiCard'
import FileUploader from '@/components/FileUploader' 

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL
const POLLING_INTERVAL_MS = 3000 // 3 seconds

interface KpiData {
  kpis: {
    financials: {
      total_revenue: number;
      gross_margin_pct: number;
    }
  }
  [key: string]: any; 
}

export default function Page() {
  const [kpiData, setKpiData] = useState<KpiData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);

  // --- ASYNCHRONOUS POLLING LOGIC ---
  const pollJobStatus = useCallback(async (currentJobId: string) => {
    try {
      // 1. Call the status endpoint
      const response = await axios.get(`${API_BASE_URL}/analyze/status/${currentJobId}`);
      
      if (response.data.status === 'complete') {
        // Job finished: update state with final data and stop loading
        setKpiData(response.data.data as KpiData);
        setIsLoading(false);
        setJobId(null);
      } else {
        // Job still processing: poll again after interval
        setTimeout(() => pollJobStatus(currentJobId), POLLING_INTERVAL_MS);
      }
    } catch (e) {
      console.error("Polling Failed:", e);
      setIsLoading(false);
      setJobId(null);
      alert('Analysis failed during processing. Please check backend logs.');
    }
  }, []);

  // Effect hook to start polling when a job ID is set
  useEffect(() => {
    // Only start polling if we have a job ID and we are currently loading
    if (jobId && isLoading) {
      pollJobStatus(jobId);
    }
  }, [jobId, isLoading, pollJobStatus]);

  // Helper function to format currency (example)
  const formatCurrency = (value: number) => {
    if (typeof value !== 'number') return 'N/A';
    // Display in Millions
    return `₹${(value / 1000000).toFixed(1)}M`; 
  };
  
  // Use dynamic data if available
  const revenue = kpiData?.kpis?.financials?.total_revenue || 0;
  const grossMargin = kpiData?.kpis?.financials?.gross_margin_pct || 0;
  
  // Static placeholders for demonstration
  const activeUsers = 4320; 
  const churn = "2.1%"; 

  return (
    <main className="p-6 max-w-6xl mx-auto">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">SmartBrain</h1>
          <p className="text-sm text-gray-600">Business insights and KPI assistant.</p>
        </div>
        <div className="flex gap-3">
          <button className="px-3 py-2 bg-white border rounded-lg">Login</button>
          <button className="px-3 py-2 bg-indigo-600 text-white rounded-lg">Get Credits</button>
        </div>
      </header>

      {/* DYNAMIC KPI SECTION */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <KpiCard title="Total Revenue" value={kpiData ? formatCurrency(revenue) : 'N/A'} delta="+12%" />
        <KpiCard title="Gross Margin" value={kpiData ? `${grossMargin.toFixed(1)}%` : 'N/A'} delta="+3%" />
        <KpiCard title="Active Users" value={kpiData ? activeUsers : 'N/A'} delta="-0.4%" />
      </section>

      {/* LOADING INDICATOR (Timeout Prevention UX) */}
      {isLoading && jobId && (
          <div className="bg-yellow-50 border-l-4 border-yellow-500 text-yellow-800 p-4 mb-6 rounded-md">
            <p className="font-semibold">Processing Data...</p>
            <p className="text-sm">Job ID: **{jobId}**. This task is running on a dedicated worker and will not timeout.</p>
          </div>
      )}

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="p-4 bg-white rounded-2xl shadow-sm min-h-[400px]">
            <h2 className="text-lg font-semibold mb-4">Ask SmartBrain</h2>
            {/* Pass the data payload to the ChatConsole for context */}
            <ChatConsole kpiData={kpiData} /> 
          </div>
        </div>

        <aside className="space-y-4">
          <FileUploader 
            setJobId={setJobId} 
            setIsLoading={setIsLoading} 
            setKpiData={setKpiData}
          />
          
          <div className="p-4 bg-white rounded-2xl shadow-sm">
            <h3 className="font-semibold">Recent Reports</h3>
            <p className="mt-2 text-sm text-gray-700">No reports yet — run an analysis to create one.</p>
          </div>
        </aside>
      </section>

      <footer className="mt-8 text-center text-sm text-gray-500">
        © SmartBrain
      </footer>
    </main>
  )
}