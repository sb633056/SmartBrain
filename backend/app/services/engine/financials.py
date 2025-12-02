import pandas as pd
from .helpers import to_num, safe_sum



def compute_pnl(df, kpis, mapping):
    pnl={}; total_revenue=kpis.get('sales',{}).get('total_revenue'); pnl['GMV']=total_revenue
    s_map = mapping.get('sales',{}); m_map = mapping.get('master',{})
    if s_map.get('sku') in df.columns and m_map.get('sku') in df.columns and m_map.get('cost_price') in df.columns:
        left = df[[s_map.get('sku'), s_map.get('quantity'), s_map.get('price')]].copy(); left.columns=['sku','quantity','price']
        master_table = df[[m_map.get('sku'), m_map.get('cost_price')]].dropna().drop_duplicates(subset=[m_map.get('sku')]); master_table.columns=['master_sku','master_cost']
        merged = left.merge(master_table, left_on='sku', right_on='master_sku', how='left'); merged['quantity']=to_num(merged['quantity']).fillna(0); merged['master_cost']=to_num(merged['master_cost']).fillna(0); merged['price']=to_num(merged['price']).fillna(0)
        merged['revenue']=merged['quantity']*merged['price']; merged['cost']=merged['quantity']*merged['master_cost']
        total_cost=float(merged['cost'].sum()); gross_profit=(total_revenue-total_cost) if total_revenue is not None else None; gross_margin_pct=round((gross_profit/total_revenue)*100,2) if total_revenue and total_revenue>0 else None
        pnl['COGS']=total_cost; pnl['Gross_Profit']=gross_profit; pnl['Gross_Margin_pct']=gross_margin_pct
    else: pnl['COGS']=None; pnl['Gross_Profit']=None; pnl['Gross_Margin_pct']=None
    marketing_attr=kpis.get('marketing_attribution',{}); marketing_spend = marketing_attr.get('cac_info',{}).get('total_marketing_spend') or kpis.get('marketing',{}).get('marketing_total_spend')
    pnl['Marketing_Spend']=marketing_spend
    if pnl.get('Gross_Profit') is not None and marketing_spend is not None: pnl['Contribution']=pnl['Gross_Profit']-marketing_spend
    else: pnl['Contribution']=None
    return pnl


def compute_cohort_monthly(df, mapping):
    s_map = mapping.get('sales',{}); date_col = s_map.get('date'); qty = s_map.get('quantity'); price = s_map.get('price')
    if date_col in df.columns and qty in df.columns and price in df.columns:
        d = df.copy(); d[date_col]=pd.to_datetime(d[date_col], errors='coerce'); d['month']=d[date_col].dt.to_period('M'); d['revenue']=to_num(d[qty])*to_num(d[price])
        monthly = d.groupby('month').agg({'revenue':'sum','order_id':'nunique'}).reset_index().sort_values('month'); monthly['month']=monthly['month'].astype(str)
        return monthly
    return None
