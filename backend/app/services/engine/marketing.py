import pandas as pd
from .helpers import safe_sum


def compute_marketing_attribution(df, mapping):
    mm = mapping.get('marketing',{})
    spend_col = mm.get('spend') if mm.get('spend') in df.columns else None
    rev_col = mm.get('attributed_revenue') if mm.get('attributed_revenue') in df.columns else None
    impr_col = mm.get('impressions') if mm.get('impressions') in df.columns else None
    clicks_col = mm.get('clicks') if mm.get('clicks') in df.columns else None
    orders_col = mm.get('attributed_orders') if mm.get('attributed_orders') in df.columns else None
    camp_col = mm.get('campaign') if mm.get('campaign') in df.columns else None
    channel_col = None
    for cand in ('channel','platform','sales_channel'): 
        if cand in df.columns: channel_col = cand; break
    warnings = []
    if not spend_col: warnings.append('Marketing spend column not found — attribution limited.')
    if not rev_col: warnings.append('Attributed revenue missing — ROAS limited.')
    if not (camp_col or channel_col): warnings.append('No campaign or channel column found — attribution unavailable.')
    def row_stats(grp, key):
        spend = safe_sum(grp, spend_col) if spend_col else 0.0
        rev = safe_sum(grp, rev_col) if rev_col else 0.0
        impr = safe_sum(grp, impr_col) if impr_col else 0.0
        clicks = safe_sum(grp, clicks_col) if clicks_col else 0.0
        orders = int(to_num(grp[orders_col]).sum()) if orders_col and orders_col in grp.columns else int(len(grp))
        roas = round(rev/spend,2) if spend>0 else None
        cpa = round(spend/orders,2) if orders>0 else None
        cpm = round(spend/(impr/1000.0),2) if impr>0 else None
        ctr = round(clicks/impr,4) if impr>0 else None
        cvr = round(orders/clicks,4) if clicks>0 else None
        return {**key, 'spend':round(spend,2), 'attributed_revenue':round(rev,2),'roas':roas,'orders':orders,'cpa':cpa,'impressions':int(impr) if impr else None,'clicks':int(clicks) if clicks else None,'ctr':ctr,'cpm':cpm,'cvr':cvr}
    channel_table=[]; campaign_table=[]
    if channel_col:
        for ch, grp in df.groupby(channel_col): channel_table.append(row_stats(grp, {'channel':ch}))
        channel_table = sorted(channel_table, key=lambda x: (x['roas'] is None, -x['roas'] if x['roas'] else 0))
    if camp_col:
        for c, grp in df.groupby(camp_col): campaign_table.append(row_stats(grp, {'campaign':c}))
        campaign_table = sorted(campaign_table, key=lambda x: (x['roas'] is None, -x['roas'] if x['roas'] else 0))
    
    # CAC info
    cac_info={}; cust_col=None
    for cand in ('customer_id','cust_id','customer','customerid'):
        if cand in df.columns: cust_col=cand; break
    if cust_col and spend_col:
        total_new_customers = int(df[cust_col].nunique()); total_spend = safe_sum(df, spend_col); cac = round(total_spend/total_new_customers,2) if total_new_customers>0 else None
        cac_info={'customer_id_column':cust_col,'total_new_customers':total_new_customers,'total_marketing_spend':round(total_spend,2),'cac':cac}
    else:
        if not cust_col:
            warnings.append('customer_id not found — CAC estimated as spend/orders (CPA).')
            if spend_col and orders_col and orders_col in df.columns:
                orders_tot=int(to_num(df[orders_col]).sum()); total_spend=safe_sum(df, spend_col); cac=round(total_spend/orders_tot,2) if orders_tot>0 else None
                cac_info={'approx_using_orders':True,'orders_total':orders_tot,'total_marketing_spend':round(total_spend,2),'cac_estimate':cac}
    
    # -------------------------------
    # Identify date + customer mapping
    # -------------------------------

    sales_map = mapping.get("sales", {})

    # Sales date
    date_col = None
    for cand in ["order_date", "date", "order_dt", "orderdate"]:
        if cand in sales_map and sales_map[cand] in df.columns:
            date_col = sales_map[cand]
            break

    # Customer ID
    cust_col = None
    for cand in ["customer_id", "cust_id", "cid", "customer"]:
        if cand in sales_map and sales_map[cand] in df.columns:
            cust_col = sales_map[cand]
            break

    # Prepare date field for segmentation
    if date_col and date_col in df.columns:
        df["__date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["__date"] = None
    
    

    # ---------- NEW vs RETARGETING ROAS ----------
    cust_col = None
    for cand in ["customer_id", "cust_id", "customer", "cid"]:
        if cand in df.columns:
            cust_col = cand
            break

    new_roas = None
    retargeting_roas = None

    if cust_col and rev_col and spend_col:
        # Identify first orders
        df['__date'] = pd.to_datetime(df[date_col], errors='coerce') if date_col in df.columns else None
        first_dates = df.groupby(cust_col)['__date'].min().to_dict()

        df['__is_new'] = df.apply(
            lambda row: row['__date'] == first_dates.get(row[cust_col]),
            axis=1
        )

        new_rev = safe_sum(df[df['__is_new']], rev_col)
        ret_rev = safe_sum(df[~df['__is_new']], rev_col)

        new_spend = safe_sum(df[df['__is_new']], spend_col)
        ret_spend = safe_sum(df[~df['__is_new']], spend_col)

        new_roas = round(new_rev / new_spend, 2) if new_spend > 0 else None
        retargeting_roas = round(ret_rev / ret_spend, 2) if ret_spend > 0 else None

    result = {
        "channel_table": channel_table,
        "campaign_table": campaign_table,
        "warnings": warnings,
        "cac_info": cac_info,
        "new_roas": new_roas,
        "retargeting_roas": retargeting_roas
    }

    return result


def compute_valuation_kpis(kpis, df):
    valuation={}; messages={}
    total_rev = kpis.get('sales',{}).get('total_revenue')
    total_orders = kpis.get('sales',{}).get('total_orders')
    marketing_attr = kpis.get('marketing_attribution',{}); cac_info = marketing_attr.get('cac_info',{})
    if total_rev and total_orders and total_orders>0: valuation['AOV']=round(total_rev/total_orders,2)
    else: valuation['AOV']=None; messages['AOV']='Not enough data - add order-level sales data to compute AOV.'
    cust_col=None
    for cand in ('customer_id','cust_id','customer'):
        if cand in df.columns: cust_col=cand; break
    if cust_col:
        total_spend = cac_info.get('total_marketing_spend'); cust_count=int(df[cust_col].nunique())
        if cust_count>0 and total_spend: valuation['CAC']=round(total_spend/cust_count,2)
        else: valuation['CAC']=None; messages['CAC']='Not enough data - customer_id or spend missing.'
    else:
        messages['CAC']='Not enough data - add customer_id to compute strict CAC. Will fallback to spend/orders (CPA) if orders present.'
        spend=cac_info.get('total_marketing_spend'); orders=cac_info.get('orders_total') or total_orders
        if spend and orders and orders>0: valuation['CAC']=round(spend/orders,2)
        else: valuation['CAC']=None; messages['CAC']='Not enough data - customer_id and orders missing.'
    # LTV fallback
    if cust_col:
        cust_order_counts = df.groupby(cust_col)['order_id'].nunique() if 'order_id' in df.columns else pd.Series([])
        if len(cust_order_counts)>0 and cust_order_counts.max()>1 and valuation.get('AOV'):
            gm=kpis.get('financials',{}).get('gross_margin_pct')
            if gm:
                avg_rev_per_customer = total_rev / cust_order_counts.sum()
                valuation['LTV']=round(avg_rev_per_customer*(gm/100),2)
            else: valuation['LTV']=None; messages['LTV']='Not enough data - gross margin missing.'
        else:
            messages['LTV']='customer repeat data insufficient - LTV estimated using AOV only.'; valuation['LTV']=valuation.get('AOV')
    else:
        if valuation.get('AOV'): messages['LTV']='customer_id not found - LTV estimated using AOV only.'; valuation['LTV']=valuation.get('AOV')
        else: valuation['LTV']=None; messages['LTV']='Not enough data - customer_id and AOV missing.'
    if isinstance(valuation.get('LTV'),(int,float)) and isinstance(valuation.get('CAC'),(int,float)) and valuation.get('CAC')>0:
        valuation['LTV_CAC']=round(valuation['LTV']/valuation['CAC'],2)
    else: valuation['LTV_CAC']=None; messages['LTV_CAC']='Not enough data - missing LTV or CAC.'
    valuation['messages']=messages
    # --- Repurchase Rate ---
    if cust_col and 'order_id' in df.columns:
        order_counts = df.groupby(cust_col)['order_id'].nunique()
        repeat_customers = (order_counts[order_counts >= 2].count())
        total_customers = order_counts.count()

        if total_customers > 0:
            valuation['repurchase_rate'] = repeat_customers / total_customers
        else:
            valuation['repurchase_rate'] = None
    else:
        valuation['repurchase_rate'] = None
        messages['repurchase_rate'] = 'Not enough customer/order data.'

    return valuation

def compute_spend_reallocation_adv(marketing_attr, pnl):
    # Ensure platform_fees resolved from marketing_attr if present
    platform_fees = marketing_attr.get('_platform_fees_by_channel') if isinstance(marketing_attr, dict) else None
    if not platform_fees:
        platform_fees = marketing_attr.get('_dataset_platform_fees') if isinstance(marketing_attr, dict) else {}
    if platform_fees is None:
        platform_fees = {}

    channels = marketing_attr.get("channel_table", [])
    if not channels or len(channels) < 2:
        return {"message": "Not enough channel data to recommend reallocation."}

    gm_pct = pnl.get("Gross_Margin_pct") or pnl.get("gross_margin_pct") or 0
    enriched = []

    # Build enriched channel performance list
    for c in channels:
        channel = c.get("channel")
        spend = c.get("spend") or 0
        roas = c.get("roas") or 0

        try:
            roas = float(roas)
        except:
            roas = 0.0
        channel_key = str(channel).lower().strip()
        fee = platform_fees.get(channel_key, 0.0)


        # Effective ROAS adjusted for platform fee and margin
        eff_roas = roas * (1 - fee) * (1 + (gm_pct / 100) * 0.1)

        enriched.append({
            **c,
            "platform_fee": fee,
            "effective_roas": round(eff_roas, 4),
            "spend": spend
        })

    # Sort by effective ROAS (lowest first)
    enriched_sorted = sorted(
        enriched,
        key=lambda x: (x["effective_roas"] is None, x["effective_roas"])
    )

    worst = enriched_sorted[0]
    best = enriched_sorted[-1]

    # Guard: check integrity
    if (
        best is None or worst is None
        or best.get("spend") is None
        or worst.get("spend") is None
        or best.get("effective_roas") is None
        or worst.get("effective_roas") is None
    ):
        return {"message": "Insufficient numeric spend/roas data to recommend reallocation."}

    spread = best["effective_roas"] - worst["effective_roas"]

    # If channels too similar → no shifting needed
    if spread <= 0.001:
        return {
            "message": (
                "Channels show similar performance after margin/fee adjustments — "
                "optimize creatives/targeting instead of shifting budget."
            )
        }

    # Recommend shifting 12% of the lowest-eff channel
    shift_pct = (marketing_attr.get("_ai_shift_override") or 0.12)
    shift_amount = round(worst["spend"] * shift_pct, 2)
    expected_revenue_delta = round(shift_amount * spread, 2)
    expected_profit_delta = (
        round(expected_revenue_delta * (gm_pct / 100), 2) if gm_pct else None
    )

    return {
        "from_channel": worst["channel"],
        "to_channel": best["channel"],
        "shift_pct": shift_pct,
        "shift_amount": shift_amount,
        "expected_revenue_delta": expected_revenue_delta,
        "expected_profit_delta": expected_profit_delta,
        "rationale": (
            f"Shift {int(shift_pct * 100)}% of {worst['channel']} spend → "
            f"{best['channel']} to improve effective ROAS after platform fees + margin."
        ),
    }