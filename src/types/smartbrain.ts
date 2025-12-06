// src/types/smartbrain.ts

export type Recommendation = {
  Title: string;
  'KPI Evidence': string;
  Action: string;
  'Expected business impact': string;
  'Analyst-confidence rating': 'High' | 'Medium' | 'Low';
};

export type SmartBrainResponse = {
  kpis: {
    financials: {
      Gross_Margin_pct: number | null;
    };
    valuation: {
      LTV_CAC: number | null;
      AOV: number | null;
      repurchase_rate: number | null;
    };
    inventory: {
      inventory_at_risk: number | null;
    };
    marketing_attribution: {
      channel_table: Array<{
        channel: string;
        spend: number;
        attributed_revenue: number;
        roas: number;
      }>;
    };
    // The AI's parsed JSON output (ai_advisor)
    ai_advisor?: {
      'Founder Summary (One Liner)': string;
      '3-Layer Recommendations Framework': {
        '🔥 Quick Wins': Recommendation[];
        '🚀 Mid-Term': Recommendation[];
        '🌱 Long-Term': Recommendation[];
      };
    };
  };
};

// Simplified type for the main data object
export interface InsightsDashboardProps {
  data: SmartBrainResponse;
}
