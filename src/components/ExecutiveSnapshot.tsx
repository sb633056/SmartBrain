// src/components/ExecutiveSnapshot.tsx
import React from 'react'

const formatCurrency = (n: number | null) => (n !== null ? `₹${n.toLocaleString('en-IN')}` : 'N/A');
const formatPercent = (n: number | null) => (n !== null ? `${n.toFixed(1)}%` : 'N/A');

export default function ExecutiveSnapshot({ kpis }: { kpis: any }) {
  // Pull data from the structured JSON
  const gmPct = kpis.financials?.Gross_Margin_pct ?? null;
  const ltv_cac = kpis.valuation?.LTV_CAC ?? null;
  const aov = kpis.valuation?.AOV ?? null;
  const invRisk = kpis.inventory?.inventory_at_risk ?? null;

  const metrics = [
    { title: 'LTV:CAC Ratio', value: ltv_cac !== null ? `${ltv_cac.toFixed(1)}x` : 'N/A', trend: ltv_cac >= 3 ? 'text-green-600' : 'text-yellow-600' },
    { title: 'Gross Margin', value: formatPercent(gmPct), trend: gmPct >= 60 ? 'text-green-600' : 'text-red-600' },
    { title: 'Avg. Order Value (AOV)', value: formatCurrency(aov), trend: 'text-gray-600' },
    { title: 'Inventory At Risk', value: formatCurrency(invRisk), trend: invRisk > 0 ? 'text-red-600' : 'text-green-600' },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {metrics.map((m) => (
        <div key={m.title} className="p-4 bg-gray-50 border border-gray-200 rounded-lg shadow-sm">
          <p className="text-sm font-medium text-gray-500 truncate">{m.title}</p>
          <div className="mt-1 text-2xl font-bold">
            <span className={m.trend}>{m.value}</span>
          </div>
        </div>
      ))}
    </div>
  )
}
