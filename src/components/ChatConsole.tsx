// src/components/ChatConsole.tsx

'use client'
import React, { useState } from 'react'
import { runSmartBrain } from '@/lib/api' // Assuming this returns the full JSON payload
import { SmartBrainResponse } from '@/types/smartbrain' // <--- IMPORT NEW TYPE
// Import the new components we're about to create
import InsightsDashboard from './InsightsDashboard' 

interface ChatConsoleProps {
  // Use the new type for kpiData prop
  kpiData: any; 
}

export default function ChatConsole({ kpiData }: ChatConsoleProps) {
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<Array<{ q: string; a: string }>>([])
  // NEW STATE: To store the structured API response object
  const [analysisResult, setAnalysisResult] = useState<SmartBrainResponse | null>(null)


  async function handleRun() {
    if (!prompt.trim()) return
    setLoading(true)
    setAnalysisResult(null); // Clear previous result

    try {
      // runSmartBrain now returns the full structured JSON payload
      const res: SmartBrainResponse = await runSmartBrain(prompt, kpiData) 
      
      // Store the structured response object
      setAnalysisResult(res)
      
      // Update history with a success message, not raw data
      setHistory((h) => [{ q: prompt, a: 'Analysis Complete. Displaying Founder Insights...' }, ...h])
      setPrompt('')
    } catch (e: any) {
      setAnalysisResult(null);
      // Use the old output area for errors
      setHistory((h) => [{ q: prompt, a: `Error: ${e.message}` }, ...h])
    } finally {
      setLoading(false)
    }
  }
  // ... (rest of the ChatConsole logic)

  return (
    <div className="space-y-4">
      {/* ... Prompt and Run/Clear buttons here ... */}

      {analysisResult ? (
        // RENDER THE NEW DASHBOARD COMPONENT
        <InsightsDashboard data={analysisResult} />
      ) : (
        // Fallback or History Display
        <div className="...">
            {/* Display History or initial welcome message */}
        </div>
      )}
    </div>
  )
}
// FORCING UNCACHED REDEPLOY
// Final code change to resolve TypeScript build error.

