# smartbrain/backend/app/services/engine/kpi.py
import pandas as pd
from .helpers import detect_domains, normalize_column_mapping, to_num, safe_sum, make_json_safe
from .inventory import compute_inventory_health_strict, compute_expiry_warnings
from .marketing import compute_marketing_attribution, compute_spend_reallocation_adv, compute_valuation_kpis
from .financials import compute_pnl, compute_cohort_monthly


def build_kpi_payload(df):
    mapping, domains = detect_domains(df)
    mapping = normalize_column_mapping(mapping, df)
    # -------- FIX MISTAKEN INVENTORY MAPPING --------
    # Avoid mapping sales.sku to inventory.sku
    if mapping['inventory'].get('sku') == mapping['sales'].get('sku'):
        # Find a true inventory SKU column
        for col in df.columns:
            if 'on_hand' in col.lower() and 'on_hand' not in mapping['inventory']:
                mapping['inventory']['on_hand'] = col
            if 'sku' in col.lower() and col != mapping['sales'].get('sku') and 'sku' not in mapping['inventory']:
                mapping['inventory']['sku'] = col

    kpi = {'domains_detected':domains,'mappings':mapping,'kpis':{}}
    # sales
    if domains.get('sales'):
        s = mapping['sales']; qty = to_num(df[s.get('quantity')]) if s.get('quantity') in df.columns else pd.Series([0]*len(df))
        price = to_num(df[s.get('price')]) if s.get('price') in df.columns else pd.Series([0]*len(df))
        df['__rev'] = qty.fillna(0)*price.fillna(0)
        total_revenue = float(df['__rev'].sum()); total_orders = int(df[s.get('order_id')].nunique()) if s.get('order_id') in df.columns else int(len(df)); total_units = int(qty.fillna(0).sum())
        aov = round(total_revenue/total_orders,2) if total_orders>0 else None
        kpi['kpis']['sales']={'total_revenue':round(total_revenue,2),'total_orders':total_orders,'total_units':total_units,'aov':aov}
        # --- ADD THIS: units_per_sku ---
        if s.get('sku') in df.columns and s.get('quantity') in df.columns:
            units_per_sku = (
                df.groupby(s.get('sku'))[s.get('quantity')]
                .apply(lambda x: to_num(x).fillna(0).sum())
                .to_dict()
            )
        else:
            units_per_sku = {}

        kpi['kpis']['sales']['units_per_sku'] = units_per_sku

        # --- ADD THIS: revenue_per_sku ---
        if s.get('sku') in df.columns:
            revenue_per_sku = (
                df.groupby(s.get('sku'))['__rev']
                .sum()
                .to_dict()
            )
        else:
            revenue_per_sku = {}

        kpi['kpis']['sales']['revenue_per_sku'] = revenue_per_sku

    # marketing
    if domains.get('marketing'):
        mm=mapping['marketing']; mk={}
        if mm.get('spend') in df.columns: mk['marketing_total_spend']=float(to_num(df[mm.get('spend')]).fillna(0).sum())
        if mm.get('attributed_revenue') in df.columns: mk['marketing_attributed_revenue']=float(to_num(df[mm.get('attributed_revenue')]).fillna(0).sum())
        if mk.get('marketing_total_spend') and mk.get('marketing_attributed_revenue'): mk['marketing_roas']=round(mk['marketing_attributed_revenue']/mk['marketing_total_spend'],2)
        kpi['kpis']['marketing']=mk; kpi['kpis']['marketing_attribution']=compute_marketing_attribution(df,mapping)
        # ===========================
        # STRICT PLATFORM FEES (USER-PROVIDED ONLY)
        # ===========================

        def normalize_channel(ch):
            """
            Convert channel labels like:
            'Amazon', 'amazon ', 'Amazon Marketplace', 'Shopify India'
            into strict canonical forms:
            'amazon', 'shopify', 'nykaa'
            """
            if not isinstance(ch, str):
                return None
            return (
                ch.lower().strip()
                .replace("india", "")
                .replace("marketplace", "")
                .replace("channel", "")
                .replace("_", "")
                .replace("-", "")
                .strip()
            )

        platform_fees = {}

        pf_col = mapping["marketing"].get("platform_fee")      # where platform fee exists
        ch_col = mapping["sales"].get("channel") or mapping["marketing"].get("channel")  # SALES > MARKETING

        if pf_col and pf_col in df.columns and ch_col and ch_col in df.columns:
            # Pull one fee per channel
            raw = (
                df[[ch_col, pf_col]]
                .dropna(subset=[pf_col])
                .drop_duplicates(subset=[ch_col])
            )

            def normalize_fee(v):
                try:
                    s = str(v).strip()
                    if s.endswith("%"):
                        return float(s.replace("%", "")) / 100

                    v = float(s)

                    # Excel special case: 0.0022 meaning 22%
                    if v < 0.05: 
                        return round(v * 100, 4)

                    # user typed 22 instead of 0.22
                    if v > 1:
                        return round(v / 100, 4)

                    return round(v, 4)
                except:
                    return None

            for _, row in raw.iterrows():
                ch_name = normalize_channel(row[ch_col])
                fee_val = normalize_fee(row[pf_col])
                if ch_name and fee_val is not None:
                    platform_fees[ch_name] = fee_val

        kpi["kpis"]["platform_fees"] = platform_fees
        # --- Propagate platform_fees into the marketing_attribution payload so strict sims use them ---
        # ensure marketing_attribution exists
        ma = kpi['kpis'].setdefault('marketing_attribution', {})

        # normalize keys to lowercase and values to floats
        normalized_pf = {}
        for ch, v in platform_fees.items():
            try:
                normalized_pf[str(ch).strip().lower()] = float(v)
            except:
                # skip invalid entries
                continue

        # store under the keys used by compute_spend_reallocation_adv
        ma['_platform_fees_by_channel'] = normalized_pf
        ma['_dataset_platform_fees'] = normalized_pf
        # also keep a copy at root for UI (already present), but making sure both are identical
        kpi['kpis']['platform_fees'] = normalized_pf

    # ---------------- INVENTORY (STRICT DOC, USER-PROVIDED ONLY) ----------------
    if domains.get('inventory'):
        im = mapping['inventory']
        inv_k = {}

        # Build inventory_per_sku (take last known on_hand)
        if im.get('sku') in df.columns and im.get('on_hand') in df.columns:
            inv_per = (
                df[[im.get('sku'), im.get('on_hand')]]
                .drop_duplicates(subset=[im.get('sku')], keep='last')
                .set_index(im.get('sku'))[im.get('on_hand')]
                .to_dict()
            )
            inv_k['inventory_per_sku'] = {k: int(v) for k, v in inv_per.items()}

        kpi['kpis']['inventory'] = inv_k

        # ---------- Shelf-life (STRICT: dataset only, no AI fallback) ----------
        shelf_candidates = [
            c for c in df.columns if c.lower().strip() in 
            ("shelf_life_days", "shelf_life", "expiry_days", "expected_shelf_life")
        ]

        shelf_map = {}
        if shelf_candidates:
            shelf_col = shelf_candidates[0]
            sku_col = mapping['inventory'].get('sku') or mapping['sales'].get('sku')


            if sku_col and sku_col in df.columns:
                tmp = df[[sku_col, shelf_col]].dropna(subset=[shelf_col])
                tmp = tmp.drop_duplicates(subset=[sku_col], keep='last')
                shelf_map = {
                    str(r[sku_col]).strip(): int(r[shelf_col])
                    for _, r in tmp.iterrows()
                }

        kpi['kpis']['shelf_life_map'] = shelf_map  # empty = strict failure



        # ---------- STRICT DOC: require shelf-life to compute expiry ----------
        inventory_health = compute_inventory_health_strict(
            inv_kpis=inv_k,
            mapping=mapping,
            df=df,
            ai_shelf_map=shelf_map  # renamed argument
        )
        kpi['kpis']['inventory_health'] = inventory_health


    # --- v13.4 standardize DOC across pipeline (use active-sales-days retail standard) ---
    try:
        # Build canonical doc_by_sku using active sales days (days with sales for that SKU)
        s_map = mapping.get('sales', {})
        sku_col = s_map.get('sku')
        qty_col = s_map.get('quantity')
        date_col = s_map.get('date') or s_map.get('order_date')

        doc_by_sku = {}
        shelf_life_by_sku = kpi['kpis'].get('shelf_life_map', {}) or {}
        if sku_col in df.columns and qty_col in df.columns and date_col in df.columns:
            sub = df[[sku_col, qty_col, date_col]].rename(columns={sku_col:'sku', qty_col:'quantity', date_col:'date'}).copy()
            sub['quantity'] = to_num(sub['quantity']).fillna(0)
            total_sold = sub.groupby('sku')['quantity'].sum().to_dict()
            active_days = sub.groupby('sku')['date'].nunique().to_dict()

            inv_rows = kpi['kpis'].get('inventory_health', {}).get('rows', [])
            inv_rows_map = {r['sku']: r for r in inv_rows}
            # Compute DOC using active-sales-days and update inventory_health rows
            for sku, on_hand in kpi['kpis'].get('inventory', {}).get('inventory_per_sku', {}).items():
                sold = total_sold.get(sku, 0)
                days = active_days.get(sku, 0)
                if days > 0 and sold > 0:
                    daily = sold / days
                    doc = round(on_hand / daily, 2) if daily>0 else None
                elif sold == 0:
                    doc = float('inf')  # no sales, infinite cover
                else:
                    doc = None

                doc_by_sku[sku] = doc if (doc is None or doc==float('inf')) else float(round(doc,2))
                # update rows if present
                row = inv_rows_map.get(sku)
                if row is not None:
                    row['days_of_cover'] = doc if (doc is None or doc==float('inf')) else float(round(doc,2))
                    # ensure shelf life present where possible
                    if sku in shelf_life_by_sku:
                        row['shelf_life_days'] = shelf_life_by_sku.get(sku)
                    # recompute flag using same logic as inventory_health function
                    try:
                        sld = row.get('shelf_life_days')
                        rdoc = row.get('days_of_cover')
                        if rdoc is None:
                            row['flag'] = 'unknown'
                        elif sld is not None and rdoc > sld:
                            row['flag'] = 'expiry_risk'
                        elif rdoc > 365:
                            row['flag'] = 'overstock'
                        elif rdoc < 30:
                            row['flag'] = 'risk_of_stockout'
                        else:
                            row['flag'] = 'ok'
                    except Exception:
                        pass

            # write back standardized rows and doc map
            kpi['kpis']['inventory_health']['rows'] = list(inv_rows_map.values())
            kpi['kpis']['inventory']['doc_by_sku'] = doc_by_sku
            kpi['kpis']['inventory']['shelf_life_days_by_sku'] = shelf_life_by_sku
    except Exception as _e:
        # non-fatal; keep original inventory payload if anything goes wrong
        kpi['kpis']['_inventory_doc_standardize_error'] = str(_e)


    # --- v13.5: ensure expiry_warnings and inventory_at_risk reflect inventory_health potential writeoffs ---
    try:
        # inventory_value is the canonical place where potential_writeoff is stored
        inv_value_rows = inventory_health.get('inventory_value', []) if isinstance(inventory_health, dict) else []
        # Normalize potential_writeoff to numeric (inventory_value entries include potential_writeoff)
        for r in inv_value_rows:
            try:
                # some entries may have None; coerce to 0.0
                r['potential_writeoff'] = float(r.get('potential_writeoff') or 0.0)
            except:
                r['potential_writeoff'] = 0.0

        # Build a quick map sku -> potential_writeoff and on_hand (so UI can show unified expiry_warnings)
        expiry_rows = []
        sku_to_writeoff = {str(r.get('sku')): r.get('potential_writeoff', 0.0) for r in inv_value_rows}

        # Pull doc/flag rows (if present) and merge writeoff numbers into them for a single canonical row object
        doc_rows = inventory_health.get('rows', []) if isinstance(inventory_health, dict) else []
        for r in doc_rows:
            sku = str(r.get('sku'))
            expiry_rows.append({
                'sku': sku,
                'on_hand': r.get('on_hand'),
                'shelf_life_days': r.get('shelf_life_days'),
                'days_of_cover': r.get('days_of_cover'),
                'days_with_sales': r.get('days_with_sales'),
                'flag': r.get('flag'),
                'potential_writeoff': float(sku_to_writeoff.get(sku, 0.0))
            })

        # Fallback: if doc_rows empty but inventory_value exists, expose inventory_value directly
        if not expiry_rows and inv_value_rows:
            for r in inv_value_rows:
                expiry_rows.append({
                    'sku': str(r.get('sku')),
                    'on_hand': r.get('on_hand'),
                    'shelf_life_days': None,
                    'days_of_cover': None,
                    'days_with_sales': None,
                    'flag': 'unknown',
                    'potential_writeoff': float(r.get('potential_writeoff', 0.0))
                })

        # Put the canonical expiry rows into the payload so UI reads the same truth as inventory table
        kpi['kpis']['expiry_warnings'] = expiry_rows

        # Aggregate numeric writeoff so dashboard widgets can sum quickly (and backwards compatible keys)
        total_writeoff = float(round(sum(r.get('potential_writeoff', 0.0) for r in expiry_rows), 2))
        # Root-level field (handy)
        kpi['kpis']['inventory_at_risk'] = total_writeoff
        # Also put under inventory namespace (UI may read this path)
        if 'inventory' not in kpi['kpis']:
            kpi['kpis']['inventory'] = {}
        kpi['kpis']['inventory']['inventory_at_risk'] = total_writeoff
        # Extra helpful fields for UI and debugging
        kpi['kpis']['inventory']['inventory_at_risk_sku_count'] = int(sum(1 for r in expiry_rows if r.get('potential_writeoff',0) > 0))
        kpi['kpis']['_inventory_at_risk_computed_from'] = 'inventory_health.inventory_value + inventory_health.rows'
    except Exception as e:
        # Keep payload stable if anything goes wrong and surface error for debugging
        kpi['kpis']['_inventory_at_risk_error'] = str(e)


    
    # DO NOT add any AI overrides for fees or shelf life



    # financials
    if domains.get('master') and domains.get('sales') and mapping['master'].get('cost_price') in df.columns and mapping['sales'].get('sku') in df.columns:
        try:
            s=mapping['sales']; m=mapping['master']; left=df[[s.get('sku'), s.get('quantity'), s.get('price')]].copy(); left.columns=['sku','quantity','price']
            master_table = df[[m.get('sku'), m.get('cost_price')]].dropna().drop_duplicates(subset=[m.get('sku')]); master_table.columns=['master_sku','master_cost']
            merged = left.merge(master_table, left_on='sku', right_on='master_sku', how='left'); merged['quantity']=to_num(merged['quantity']).fillna(0); merged['price']=to_num(merged['price']).fillna(0); merged['master_cost']=to_num(merged['master_cost']).fillna(0)
            merged['revenue']=merged['quantity']*merged['price']; merged['cost']=merged['quantity']*merged['master_cost']
            total_revenue=float(merged['revenue'].sum()); total_cost=float(merged['cost'].sum()); gross_margin_pct=round((total_revenue-total_cost)/total_revenue*100,2) if total_revenue>0 else None
            kpi['kpis']['financials']={'total_revenue':round(total_revenue,2),'total_cost':round(total_cost,2),'gross_margin_pct':gross_margin_pct,'_evidence':{'rows_used':len(merged)}}
        except Exception as e:
            kpi['kpis']['financials']={'error':str(e)}
    kpi['kpis']['valuation']=compute_valuation_kpis(kpi['kpis'], df)
    kpi['kpis']['pnl']=compute_pnl(df, kpi['kpis'], mapping)
    # Use strict DOC calculation per user instruction; if missing cols return error message in payload
    kpi['kpis']['cohort_monthly']=compute_cohort_monthly(df, mapping)
    kpi['_meta']={'confidence':'High' if kpi.get('kpis',{}).get('sales',{}).get('total_orders',0)>10 else 'Low'}
    kpi['mappings']=mapping; kpi['domains']=domains
    
    from .helpers import make_json_safe
    return make_json_safe(kpi)
