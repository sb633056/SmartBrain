// src/components/RecommendationsList.tsx
import React from 'react'
import { Recommendation } from '@/types/smartbrain'

const RecItem = ({ rec }: { rec: Recommendation }) => (
  <div className="p-3 border-b last:border-b-0">
    <h4 className="text-lg font-bold text-gray-800">{rec.Title}</h4>
    <p className="text-sm mt-1 text-gray-600">
      <span className="font-semibold">Action:</span> {rec.Action}
    </p>
    <ul className="list-disc list-inside text-xs text-gray-500 mt-2 space-y-1">
      <li><span className="font-medium">Evidence:</span> {rec['KPI Evidence']}</li>
      <li><span className="font-medium">Impact:</span> {rec['Expected business impact']}</li>
      <li><span className="font-medium">Confidence:</span> <span className={`font-bold ${rec['Analyst-confidence rating'] === 'High' ? 'text-green-500' : 'text-yellow-600'}`}>{rec['Analyst-confidence rating']}</span></li>
    </ul>
  </div>
);

export default function RecommendationsList({ recommendations }: { recommendations: any }) {
  const tiers = [
    { title: '🔥 Quick Wins (0–7 days)', key: '🔥 Quick Wins' },
    { title: '🚀 Mid-Term (2–8 weeks)', key: '🚀 Mid-Term' },
    { title: '🌱 Long-Term (3–12 months)', key: '🌱 Long-Term' },
  ];

  return (
    <div className="space-y-6">
      {tiers.map((tier) => (
        <div key={tier.key} className="p-4 bg-white border rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-indigo-700 mb-3">{tier.title}</h3>
          {recommendations[tier.key]?.map((rec: Recommendation, index: number) => (
            <RecItem key={index} rec={rec} />
          ))}
        </div>
      ))}
    </div>
  )
}
