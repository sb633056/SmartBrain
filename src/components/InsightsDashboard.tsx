// src/components/InsightsDashboard.tsx
import React from 'react'
import { InsightsDashboardProps } from '@/types/smartbrain'
import ExecutiveSnapshot from './ExecutiveSnapshot' // NEW
import FounderSummary from './FounderSummary'     // NEW
import RecommendationsList from './RecommendationsList' // NEW
import DataTable from './DataTable'                 // NEW (Generic)

export default function InsightsDashboard({ data }: InsightsDashboardProps) {
  if (!data || !data.kpis) return null;

  const kpis = data.kpis;
  const aiAdvisor = kpis.ai_advisor || {};
  const channelTable = kpis.marketing_attribution.channel_table;
  
  // NOTE: You must ensure your FastAPI endpoint returns the AI data nested under 'ai_advisor' as inferred.
  const summary = aiAdvisor['Founder Summary (One Liner)'] || "Analysis complete, but founder summary is unavailable.";
  const recommendations = aiAdvisor['3-Layer Recommendations Framework'];

  return (
    <div className="space-y-8 p-6 bg-white rounded-xl shadow-2xl">
      <h1 className="text-3xl font-extrabold text-indigo-700">SmartBrain Founder Insights</h1>
      <p className="text-gray-500">Board-ready intelligence derived from your dataset.</p>

      {/* 📌 EXECUTIVE SNAPSHOT (Matches PDF Grid) */}
      <section>
        <ExecutiveSnapshot kpis={kpis} />
      </section>

      {/* 1. FOUNDER SUMMARY (Matches PDF Block Quote) */}
      <section className="space-y-3">
        <h3 className="text-xl font-semibold border-b pb-2">1. Founder Summary (One Liner)</h3>
        <FounderSummary summary={summary} />
      </section>

      {/* 2. 3-LAYER RECOMMENDATIONS FRAMEWORK (Matches PDF List) */}
      <section className="space-y-3">
        <h3 className="text-xl font-semibold border-b pb-2">2. 3-Layer Recommendations Framework</h3>
        {recommendations ? (
          <RecommendationsList recommendations={recommendations} />
        ) : (
          <p className="text-red-500">Recommendations data structure is missing from API response.</p>
        )}
      </section>

      {/* DATA TABLES (INVENTORY HEALTH) */}
      <section className="space-y-3">
        <h3 className="text-xl font-semibold border-b pb-2">Inventory Health</h3>
        {/* Assuming 'expiry_warnings' is the canonical source for Inventory Health Table data (as per Python) */}
        {kpis.expiry_warnings ? (
          <DataTable 
            data={kpis.expiry_warnings} 
            title="Inventory Expiry Risk"
            // Define your column headers
            columns={['sku', 'on_hand', 'shelf_life_days', 'days_of_cover', 'flag', 'potential_writeoff']}
          />
        ) : (
          <p className="text-gray-500">Inventory Health data not available.</p>
        )}
      </section>

      {/* DATA TABLES (MARKETING ATTRIBUTION) */}
      <section className="space-y-3">
        <h3 className="text-xl font-semibold border-b pb-2">Marketing Attribution</h3>
        {channelTable && channelTable.length > 0 ? (
          <DataTable 
            data={marketingAttr.channel_table} 
            title="Channel ROAS Performance"
            // Define your column headers
            columns={['channel', 'spend', 'attributed_revenue', 'roas']}
          />
        ) : (
          <p className="text-gray-500">Marketing Attribution data not available.</p>
        )}
      </section>
      
    </div>
  )
}
