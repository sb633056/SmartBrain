import pandas as pd
from .helpers import to_num, safe_sum



def compute_inventory_health_strict(inv_kpis, mapping, df, ai_shelf_map = None):
    
    if ai_shelf_map is None or len(ai_shelf_map) == 0:
        return {
            "error": "Not enough data — provide shelf_life_days per SKU",
            "rows": [],
            "inventory_value": []
        }



    rows = []
    inv_per = inv_kpis.get('inventory_per_sku', {})

    s_map = mapping.get('sales', {})
    sku_col = s_map.get('sku')
    qty_col = s_map.get('quantity')
    date_col = s_map.get('date')
    on_hand_col = mapping.get('inventory', {}).get('on_hand')
    # Ensure AI shelf-life map exists
    try:
        ai_shelf_map = ai_shelf_map or {}
    except NameError:
        ai_shelf_map = {}


    # ✔ Basic required fields
    missing = []
    if not sku_col or sku_col not in df.columns: missing.append('sku')
    if not qty_col or qty_col not in df.columns: missing.append('quantity')
    if not date_col or date_col not in df.columns: missing.append('date')
    if not on_hand_col or on_hand_col not in df.columns: missing.append('on_hand')


    if missing:
        return {'error': f"Not enough data - add columns: {', '.join(missing)} to compute Days of Cover (DOC).", 'rows':[], 'inventory_value':[]}
            

    sub = df[[sku_col, qty_col, date_col]].rename(columns={sku_col:'sku', qty_col:'quantity', date_col:'date'}).copy()
    sub['quantity'] = to_num(sub['quantity']).fillna(0)
    grouped_qty = sub.groupby('sku')['quantity'].sum().to_dict()
    unique_days = sub.groupby('sku')['date'].nunique().to_dict()

    grouped_qty = sub.groupby('sku')['quantity'].sum().to_dict()

    m_map = mapping.get('master',{})
    sku_cost_map = {}
    if m_map.get('sku') in df.columns and m_map.get('cost_price') in df.columns:
        tmp = df[[m_map.get('sku'), m_map.get('cost_price')]].drop_duplicates(subset=[m_map.get('sku')])
        tmp.columns = ['sku','cost']
        sku_cost_map = {r['sku']: float(r['cost']) for _,r in tmp.iterrows()}
    
    for sku, on_hand in inv_per.items():
        total_sold = grouped_qty.get(sku, 0)
        days_with_sales = unique_days.get(sku, 0)

        # Determine shelf life for this SKU (priority: user-provided -> AI map -> deterministic)
        shelf_life_days = None
        if isinstance(ai_shelf_map, dict) and sku in ai_shelf_map and ai_shelf_map.get(sku):
            try:
                shelf_life_days = int(ai_shelf_map.get(sku))
            except:
                shelf_life_days = None
        # else deterministic fallback will be used later

        if days_with_sales <= 0:
            return_value = None
            note = 'Not enough sale-day granularity for this SKU'
        else:
            avg_daily_sales = total_sold / days_with_sales if days_with_sales>0 else None
            if avg_daily_sales and avg_daily_sales>0:
                doc = on_hand / avg_daily_sales if avg_daily_sales else None
                # no infinite values; cap for UI clarity but DO NOT alter flag logic
                if doc is None:
                    return_value = None
                else:
                    return_value = round(doc, 2)
                note = None
            else:
                return_value = None
                note = 'No sales for SKU in period'

        
        

        # Flagging logic using effective shelf life
        if return_value is None:
            flag = 'unknown'
        elif shelf_life_days is not None and return_value > shelf_life_days:
            flag = 'expiry_risk'
        elif return_value > 365:
            flag = 'overstock'
        elif return_value < 30:
            flag = 'risk_of_stockout'
        else:
            flag = 'ok'

        rows.append({
            'sku': sku,
            'on_hand': int(on_hand),
            'total_sold': int(total_sold),
            'days_with_sales': int(days_with_sales),
            'days_of_cover': return_value,
            'shelf_life_days': shelf_life_days,
            'flag': flag,
            'note': note
        })

    # inventory value & potential write-off (if cost exists)
    inv_value_rows = []
    for r in rows:
        cost = sku_cost_map.get(r['sku'])
        inv_value = None
        potential_writeoff = None
        if cost is not None:
            try:
                inv_value = r['on_hand'] * cost
            except:
                inv_value = None
        # write-off estimate if expiry risk
        if cost is not None and r.get('days_of_cover') is not None and r.get('shelf_life_days') is not None:
            if r['days_of_cover'] > r['shelf_life_days']:
                potential_writeoff = round(r['on_hand'] * cost, 2)
        inv_value_rows.append({
            'sku': r['sku'],
            'on_hand': r['on_hand'],
            'cost_per_unit': cost,
            'inventory_value': inv_value,
            'potential_writeoff': potential_writeoff
        })

    return {'rows': rows, 'inventory_value': inv_value_rows}
    # Inventory value:
    inv_value_rows = []
    m_map = mapping.get('master', {})
    if m_map.get('sku') in df.columns and m_map.get('cost_price') in df.columns:
        sku_cost = df[[m_map['sku'], m_map['cost_price']]].drop_duplicates()
        sku_cost.columns = ['sku', 'cost']
        for r in rows:
            match = sku_cost[sku_cost['sku'] == r['sku']]
            cost = float(match['cost'].values[0]) if not match.empty else None
            inv_value_rows.append({
                "sku": r["sku"],
                "on_hand": r["on_hand"],
                "cost_per_unit": cost,
                "inventory_value": r["on_hand"] * cost if cost else None
            })

    return {"rows": rows, "inventory_value": inv_value_rows}


def compute_expiry_warnings(df, mapping, inventory_health, use_ai_fallback=None):
    """
    Uses inventory_health + shelf_life_days (or inferred shelf life)
    to compute expiry risk, write-offs, and flags.
    """
    rows = []
    inv_rows = inventory_health.get("rows", [])
    if not inv_rows:
        return []

    sku_col = mapping["inventory"].get("sku")
    cost_col = mapping.get("master", {}).get("cost_price")

    has_shelf_life = "shelf_life_days" in df.columns

    for r in inv_rows:
        sku = r["sku"]
        on_hand = r["on_hand"]

        # --- Shelf life value ---
        # determine fallback preference: if function caller didn't specify, use global flag
        fallback = SMARTBRAIN_USE_AI_FALLBACK if use_ai_fallback is None else bool(use_ai_fallback)
        if has_shelf_life:
            shelf_days = df[df[sku_col] == sku]['shelf_life_days'].dropna()
            shelf_days = int(shelf_days.values[0]) if not shelf_days.empty else (infer_shelf_life_ai(sku) if fallback else None)
        else:
            if fallback:
                shelf_days = infer_shelf_life_ai(sku)
            else:
                shelf_days = None


        # --- Days of Cover we already computed ---
        doc = r.get("days_of_cover")

        # --- Days Left before expiry (approx) ---
        days_left = (shelf_days - doc) if (shelf_days is not None and doc is not None and isinstance(doc, (int, float)) and isinstance(shelf_days, (int, float))) else None

        # --- Flag logic ---
        if days_left is None:
            expiry_flag = "unknown"
        elif days_left < 0:
            expiry_flag = "expired_risk"
        elif days_left < 90:
            expiry_flag = "high_risk"
        elif days_left < 180:
            expiry_flag = "medium_risk"
        else:
            expiry_flag = "low_risk"

        # --- Potential write-off ---
        cost_val = None
        if cost_col in df.columns:
            cost_match = df[df[mapping["master"].get("sku")] == sku][cost_col]
            if not cost_match.empty:
                cost_val = float(cost_match.values[0])

        potential_writeoff = (on_hand * cost_val) if (cost_val and expiry_flag in ["expired_risk", "high_risk"]) else 0

        rows.append({
            "sku": sku,
            "on_hand": on_hand,
            "shelf_life_days": shelf_days,
            "days_of_cover": doc,
            "days_left": days_left,
            "expiry_flag": expiry_flag,
            "potential_writeoff": potential_writeoff
        })

    return rows