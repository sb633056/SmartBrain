// src/components/FounderSummary.tsx
import React from 'react'

export default function FounderSummary({ summary }: { summary: string }) {
  return (
    <div className="p-4 border-l-4 border-indigo-600 bg-indigo-50 text-indigo-800 shadow-md">
      <p className="font-semibold text-sm">CRITICAL ACTION:</p>
      <p className="mt-1 text-lg font-bold">{summary}</p>
    </div>
  )
}
