// src/components/InsightsDashboard.tsx (FINAL DEBUG CODE TO GET IT WORKING)

import React from 'react'
// 🛑 CRITICAL: This path must be correct. If your 'types' folder is not in 'src', 
// the path is wrong and will crash the component.
import { InsightsDashboardProps } from '@/types/smartbrain' 


export default function InsightsDashboard({ data }: InsightsDashboardProps) {
  if (!data || !data.kpis) {
    return <div className="p-4 bg-red-100">ERROR: Invalid data structure received.</div>;
  }
  
  // This is the only line of logic left.
  const kpis = data.kpis;
  
  // 💥 IF THIS RENDERS, THE COMPONENT IS WORKING 💥
  return (
    <div className="space-y-8 p-6 bg-green-100 rounded-xl border-4 border-green-500 shadow-2xl">
      <h1 className="text-3xl font-extrabold text-green-700">
        ✅ SUCCESS! Dashboard Component Mounted!
      </h1>
      <p className="text-lg text-gray-700">
        Your entire application stack is connected and ready.
      </p>
      <details>
        <summary className="font-semibold text-sm cursor-pointer">
          Click to view raw JSON data (Proof)
        </summary>
        <pre className="mt-2 p-2 bg-white overflow-auto text-xs border rounded">
          {JSON.stringify(kpis, null, 2)}
        </pre>
      </details>
    </div>
  )
}
