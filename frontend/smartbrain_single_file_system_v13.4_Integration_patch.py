
# smartbrain_v8.py
"""
SmartBrain V8 - Founder-grade Streamlit app
- Uses strict/original DOC formula; if required columns missing, shows a clear fallback message.
- Clickable SKU cards: click a SKU to show detailed metrics in an expander.
- Compact charts by default; Compact toggle available.
- Platform analyst prompt placed near top for LLM calls.
- Example file path set to the latest uploaded dataset path (local file used as 'url' for testing).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, json, math

# Global flag: control whether AI shelf-life fallback is allowed (False = strict dataset-only)
SMARTBRAIN_USE_AI_FALLBACK = False
from datetime import datetime
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()



# Global chart sizing
plt.rcParams["figure.figsize"] = (5, 2.8)
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["figure.autolayout"] = True


# ---------------- CONFIG & PROMPT ----------------
EXAMPLE_LOCAL_PATH = "/mnt/data/f5c404ac-c7a4-417e-87d1-bf77157b57eb.xlsx"  # example dataset path (local)
EXAMPLE_FILE_URL = EXAMPLE_LOCAL_PATH  # treat the local path as a URL placeholder for your integrations
DEFAULT_DOC_DAYS_FALLBACK = 30
DOC_CAP_DAYS = 3650  # cap extreme values for UI clarity
PLATFORM_FEE_ESTIMATES = {"amazon":0.15, "shopify":0.05, "nykaa":0.12, "myntra":0.12, "flipkart":0.15, "default":0.10}

# Master system prompt (replace old/incomplete prompt with this)
PLATFORM_ANALYST_PROMPT = """You are a senior beauty & personal care industry analyst with 15+ years of experience 
working across Amazon Beauty, Nykaa, Myntra, Flipkart, and D2C Shopify brands.

You must behave like a seasoned operator who has:
• Seen brands scale from ₹50 lakhs → ₹100 crores  
• Managed CAC volatility, ROAS crashes, ad fatigue, and platform fees  
• Handled expiry-sensitive inventory (serums, actives, vitamin C, retinol)  
• Diagnosed cash-burn and retention failures for D2C founders  

You understand:
- marketplace fee structures, channel-level CAC/ROAS norms for beauty categories,
- hero SKU dynamics, cannibalization risk, inventory expiry/aging patterns,
- D2C vs marketplace LTV curves, seasonal product behavior (winter moisturizers, Q4 gifting etc.).

Task:
- Interpret the KPI JSON provided, compare to marketplace norms, and produce founder-grade recommendations:
  Quick Wins, Mid-Term Fixes, Long-Term Strategy, Platform-specific advice (Amazon/Nykaa/Myntra/Shopify),
  Spend reallocation guidance, SKU-level alerts (with margin context).
- If required columns are missing to compute a KPI, DO NOT attempt approximation; instead display a clear message:
  "Not enough data - add <column_name> column to process this".
- Do not hallucinate; if data missing, explicitly list the columns to upload next.
- Also extract platform fees / commission rates for each channel in the dataset
  (Amazon/Nykaa/Myntra/Flipkart/Shopify/etc.). 
  Return them in a JSON block called "platform_fees" with percentage values.
  If you are unsure, estimate based on the latest industry norms and clearly 
  state that these are estimates based on public category fee structures.
  Also respond with:
  {
  "platform_fees": { "amazon": 0.15, "nykaa": 0.12, ... },
  "recommended_shift_pct": 0.08,
  "justification": "Shift a smaller amount due to low spread and platform fee gap."
  }



Your output includes:
-------------------------------------------------------------
1. **Founder Summary (One Liner)**
   A sharp, investor-ready statement describing:
   • Top risks  
   • Biggest opportunities  
   • Immediate next steps  

2. **3-Layer Recommendations Framework**
   Produce a list of actions classified into:
   • 🔥 Quick Wins (0–7 days)  
   • 🚀 Mid-Term (2–8 weeks)  
   • 🌱 Long-Term (3–12 months)  

   Each recommendation MUST include:
   • Title  
   • KPI Evidence (use KPI JSON directly)  
   • Action  
   • Expected business impact  
   • Analyst-confidence rating (High/Medium/Low)  

3. **Advanced Beauty-Sector Insights**
   You must evaluate:
   • **ROAS < 1 → Cash burn risk**  
   • **Contribution Margin → Whether unit economics are viable**  
   • **Expiry Risk**  
        If “days_of_cover” >> shelf_life (if provided), flag a HIGH-RISK expiry issue  
   • **SKU concentration**  
        Overdependence on 1 hero SKU (>30–40% revenue)  
   • **Channel profitability**  
        Consider marketplace fees: 
        Amazon ~15%, Nykaa ~20–25%, Myntra ~20–22%, Shopify ~2%  
   • **Retention quality**  
        Use CAC/LTV/AOV exactly as provided.  
        If CAC/LTV missing → mention the retention insight gap.  

4. **Platform-Specific Commentary**
   Interpret marketing_attribution tables:
   • If Amazon ROAS < 1 → note margin compression due to referral fees  
   • If Nykaa ROAS < 1 → note expensive beauty CPCs  
   • Highlight if Shopify ROAS is most profitable due to lowest fees  

5. **Inventory Expiry Risk Layer**
   If any SKU has:
       days_of_cover >> expected_shelf_life_days  
   or  
       days_of_cover > 365  
   Flag:
       “At risk of expiry — urgent clearance required”

6. **Repurchase / Gateway SKU Commentary**
   Use SKU revenue data to:
   • Identify “hero product”  
   • Identify “gateway SKU” (typically low-ticket, high-repeat category)  
   • Suggest bundling, trial packs, or free-gifts for slow-moving SKUs  

7. **Spend Reallocation Commentary**
   Interpret the system-generated `spend_reallocation` output:
   • DO NOT recalculate profitability  
   • Only explain whether the suggestion makes operational and strategic sense  
   • Mention fee-adjusted ROAS dynamics  
   • Advise if creative fatigue may be the root issue

8. **Communication Style**
   • Insight-first, crisp, founder-grade intelligence  
   • No fluff  
   • Use emojis sparingly for readability  
   • Prioritize business impact (>5 crores+ thinking)  
   • Output must feel like a board-meeting memo  




"""

import re

def extract_json_from_ai(text):
    try:
        block = re.search(r"\{(.|\n)*\}", text)
        if block:
            return json.loads(block.group(0))
    except:
        return {}
    return {}






# ---------------- Utilities ----------------


def to_num(s): return pd.to_numeric(s, errors='coerce')
def safe_sum(df, col): return float(to_num(df[col]).fillna(0).sum()) if col and col in df.columns else 0.0
def platform_fee_for_channel(channel):
    if not isinstance(channel, str): return PLATFORM_FEE_ESTIMATES['default']
    return PLATFORM_FEE_ESTIMATES.get(channel.strip().lower(), PLATFORM_FEE_ESTIMATES['default'])

@st.cache_data(show_spinner=False)
def detect_domains(df):
    cols = [c.lower().strip() for c in df.columns]
    
    mapping = {
        "sales": {},
        "marketing": {},
        "inventory": {},
        "master": {}
    }
    domains = {"sales": False, "marketing": False, "inventory": False, "master": False}

    # ---------------- SALES DOMAIN ----------------
    if any(x in cols for x in ["order_id", "orderid"]) and any(x in cols for x in ["sku", "price", "quantity"]):
        domains["sales"] = True

        # Correct column names expected by SmartBrain
        sales_map = {
            "order_id": ["order_id", "orderid"],
            "order_date": ["order_date", "date", "order_dt"],
            "customer_id": ["customer_id", "cust_id", "cid", "customer"],
            "sku": ["sku", "product_sku", "item_sku"],
            "quantity": ["quantity", "qty", "units"],
            "price": ["price", "unit_price", "selling_price"]
        }

        for key, candidates in sales_map.items():
            for cand in candidates:
                if cand in cols:
                    mapping["sales"][key] = df.columns[cols.index(cand)]
                    break

    # ---------------- MARKETING DOMAIN ----------------
    if any(x in cols for x in ["spend", "attributed_revenue", "impressions", "clicks"]):
        domains["marketing"] = True
        marketing_map = {
            "spend": ["spend", "ad_spend"],
            "attributed_revenue": ["attributed_revenue", "revenue"],
            "impressions": ["impressions"],
            "clicks": ["clicks"],
            "channel": ["channel", "platform"],
            "platform_fee": ["platform_fee", "channel_fee", "fee", "commission_rate"]

        }
        for key, candidates in marketing_map.items():
            for cand in candidates:
                if cand in cols:
                    mapping["marketing"][key] = df.columns[cols.index(cand)]
                    break

    # ---------------- INVENTORY DOMAIN ----------------
    if any(x in cols for x in ["on_hand", "inventory", "stock", "onhand"]):
        domains["inventory"] = True
        inv_map = {
            "sku": ["sku"],
            "on_hand": ["on_hand", "inventory", "stock"],
            "allocated": ["allocated"]
        }
        for key, candidates in inv_map.items():
            for cand in candidates:
                if cand in cols:
                    mapping["inventory"][key] = df.columns[cols.index(cand)]
                    break

    # ---------------- MASTER DOMAIN ----------------
    if any(x in cols for x in ["cost_price", "mrp", "cost"]):
        domains["master"] = True
        master_map = {
            "sku": ["sku"],
            "cost_price": ["cost_price", "cost"],
            "mrp": ["mrp"]
        }
        for key, candidates in master_map.items():
            for cand in candidates:
                if cand in cols:
                    mapping["master"][key] = df.columns[cols.index(cand)]
                    break

    return mapping, domains



def normalize_column_mapping(mapping, df):
    """
    Ensure mapping has the canonical keys SmartBrain expects and that each
    mapped value is an actual column name from df (or None). This makes downstream
    KPI code simpler and prevents silent mapping failures.
    Returns the normalized mapping (mutates a copy).
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}  # map lower->actual column name

    # canonical keys we expect per domain
    expected = {
        "sales": ["order_id", "order_date", "customer_id", "sku", "quantity", "price", "channel", "order_value", "campaign"],
        "marketing": ["spend", "attributed_revenue", "impressions", "clicks", "campaign", "channel", "platform_fee"],
        "inventory": ["sku", "on_hand", "allocated", "batch_date", "shelf_life_days"],
        "master": ["sku", "cost_price", "mrp"]
    }

    # Make a new mapping copy to avoid accidental external mutation surprises
    nm = {"sales": {}, "marketing": {}, "inventory": {}, "master": {}}
    for domain in ("sales", "marketing", "inventory", "master"):
        domain_map = mapping.get(domain, {}) or {}
        for key in expected[domain]:
            # if user detect_domains already provided a mapping for this canonical key, use it
            if key in domain_map:
                val = domain_map[key]
                # If the detected value is already the exact column name, keep it
                if isinstance(val, str) and val in df.columns:
                    nm[domain][key] = val
                    continue
                # If detected value was a lowercase candidate name, map to actual df column
                if isinstance(val, str) and val.lower().strip() in cols_lower:
                    nm[domain][key] = cols_lower[val.lower().strip()]
                    continue	

            # otherwise try to find a matching column in df using synonyms / heuristics
            # common synonyms (extendable)
            synonyms = {
                "order_date": ["order_date", "date", "order_dt", "orderdate"],
                "order_id": ["order_id", "orderid", "id"],
                "customer_id": ["customer_id", "cust_id", "cid", "customer"],
                "sku": ["sku", "product_sku", "item_sku", "product"],
                "quantity": ["quantity", "qty", "units"],
                "price": ["price", "unit_price", "selling_price", "order_value", "order_value"],
                "spend": ["spend", "ad_spend", "marketing_spend"],
                "attributed_revenue": ["attributed_revenue", "attributed_rev", "attributed_revenue"],
                "on_hand": ["on_hand", "inventory", "stock", "onhand"],
                "cost_price": ["cost_price", "cost", "unit_cost"],
                "shelf_life_days": ["shelf_life_days", "shelf_life", "expiry_days"],
                "campaign": ["campaign", "campaign_name"],
                "channel": ["channel", "platform"],
                "platform_fee": ["platform_fee", "channel_fee", "fee", "commission_rate"]

            }
            assigned = None
            if key in synonyms:
                for cand in synonyms[key]:
                    if cand in cols_lower:
                        assigned = cols_lower[cand]
                        break
            # final fallback: check if key itself exists verbatim
            if not assigned and key in cols_lower:
                assigned = cols_lower[key]

            nm[domain][key] = assigned  # may be None if not found

    # Additional safety: if inventory.sku equals sales.sku but inventory.on_hand is missing,
    # try to find a distinct inventory sku column (some files use inventory_sku)
    if nm["inventory"].get("sku") == nm["sales"].get("sku") and nm["inventory"].get("on_hand") is None:
        for cand in ("inventory_sku","sku_inventory","inv_sku"):
            if cand in cols_lower:
                nm["inventory"]["sku"] = cols_lower[cand]
                break
    # Ensure retention logic sees a unified 'date' field
    if nm["sales"].get("order_date") and not nm["sales"].get("date"):
        nm["sales"]["date"] = nm["sales"]["order_date"]


    return nm

# ---------------- Shelf-life helpers (HYBRID: deterministic + AI batch) ----------------

def deterministic_shelf_life_from_sku(sku: str) -> int:
    """Fast deterministic fallback mapping based on SKU keywords."""
    sku_l = (sku or "").lower()
    # ordered by specificity
    if any(x in sku_l for x in ["vitamin c", "vit c", "vitc", "ascorbic"]):
        return 270
    if "retinol" in sku_l:
        return 300
    if "serum" in sku_l:
        return 365
    if any(x in sku_l for x in ["moist", "cream", "lotion"]):
        return 540
    if any(x in sku_l for x in ["sunscreen", "spf"]):
        return 540
    if any(x in sku_l for x in ["shampoo", "conditioner", "hair"]):
        return 720
    if any(x in sku_l for x in ["facewash", "cleanser", "wash"]):
        return 540
    if any(x in sku_l for x in ["mask", "sheet mask", "peel"]):
        return 540
    # default conservative value for beauty
    return 365


def ai_batch_estimate_shelf_life(skus: list[str], openai_client=None, prompt_extra=""):
    """
    Send ONE batch prompt to the LLM asking for shelf-life estimates for all SKUs.
    Returns: dict {sku: estimated_days} or {} on failure.
    - openai_client: optional OpenAI client (if None uses try/except or falls back).
    """
    if not skus:
        return {}

    # short-circuit if too many nearly-duplicates
    unique_skus = list(dict.fromkeys([s.strip() for s in skus if s and str(s).strip()]))
    if not unique_skus:
        return {}

    # Build the single prompt for all SKUs, ask for JSON mapping only
    sku_list_text = "\n".join([f"- {s}" for s in unique_skus])
    prompt = (
        "You are an industry-experienced beauty & personal care analyst. "
        "For each SKU name below, return a shelf life in days (integer). "
        "Return ONLY a JSON object mapping the SKU string to integer days. "
        "If you are uncertain, give a conservative estimate. Do NOT include commentary.\n\n"
        f"{prompt_extra}\n\nSKUs:\n{sku_list_text}\n\nRespond with JSON only like: {{\"SKU1\":365, \"SKU2\":540}}"
    )

    # Try to call OpenAI client if provided and available
    try:
        if openai_client is None:
            # Try to import and create a client lazily (if user has API key)
            try:
                from openai import OpenAI as _OpenAI
                import os as _os
                if _os.getenv("OPENAI_API_KEY"):
                    openai_client = _OpenAI(api_key=_os.getenv("OPENAI_API_KEY"))
            except Exception:
                openai_client = None

        if openai_client:
            # one-shot chat completion
            resp = openai_client.chat.create(model="gpt-4o-mini", messages=[
                {"role": "system", "content": "You are a concise industry analyst."},
                {"role": "user", "content": prompt}
            ], temperature=0.2)
            # Extract text
            content = ""
            # safe extraction depending on SDK
            if hasattr(resp, "choices") and len(resp.choices) > 0:
                # older or newer SDK shape
                try:
                    content = resp.choices[0].message["content"]
                except Exception:
                    try:
                        content = resp.choices[0].get("message", {}).get("content","")
                    except:
                        content = str(resp)
            else:
                content = str(resp)

            # parse JSON object out of content
            import json as _json, re as _re
            # extract first {...} block
            m = _re.search(r"\{.*\}", content, flags=_re.S|_re.M)
            if not m:
                return {}
            parsed = _json.loads(m.group(0))
            # convert values to int
            cleaned = {}
            for k, v in parsed.items():
                try:
                    cleaned[k.strip()] = int(v)
                except:
                    try:
                        cleaned[k.strip()] = int(float(v))
                    except:
                        pass
            return cleaned
    except Exception:
        # any AI failure falls back to deterministic mapping
        pass

    # fallback deterministic mapping
    fallback = {}
    for s in unique_skus:
        fallback[s] = deterministic_shelf_life_from_sku(s)
    return fallback

def normalize_ai_shelf_map(raw):
    """
    Converts ANY AI output into the strict:
        { "SKU": days }
    format.
    Guaranteed to never fail.
    """

    if not isinstance(raw, dict):
        return {}

    # CASE 1 — AI returns {"shelf_life_days": {...}}
    if "shelf_life_days" in raw and isinstance(raw["shelf_life_days"], dict):
        cleaned = {}
        for sku, days in raw["shelf_life_days"].items():
            try:
                cleaned[str(sku).strip()] = int(days)
            except:
                pass
        return cleaned

    # CASE 2 — AI returned nested dicts, find the one that looks like {sku: days}
    for key, val in raw.items():
        if isinstance(val, dict):
            if all(
                isinstance(k, str) and str(v).replace(".", "").isdigit()
                for k, v in val.items()
            ):
                cleaned = {}
                for sku2, days2 in val.items():
                    try:
                        cleaned[str(sku2).strip()] = int(float(days2))
                    except:
                        pass
                if cleaned:
                    return cleaned

    # CASE 3 — raw itself is {sku: days}
    if all(isinstance(k, str) and str(v).replace(".", "").isdigit() for k, v in raw.items()):
        cleaned = {}
        for sku, days in raw.items():
            try:
                cleaned[str(sku).strip()] = int(days)
            except:
                pass
        return cleaned

    return {}



# ---------------- KPI builders ----------------
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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




# Inventory health using original DOC formula with strict checks
@st.cache_data(show_spinner=False)
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

def infer_shelf_life_ai(sku_name):
    """
    Uses the beauty analyst LLM prompt to infer shelf life based on product name.
    Acts only if no shelf_life_days column exists.
    """
    try:
        prompt = f"""
        You are a senior beauty & personal care analyst with 15+ years of FMCG + D2C experience.
        Infer the typical shelf life (in days) for the SKU based on its name: "{sku_name}"

        Respond with ONLY a number. No text. No explanation.
        General guidelines:
        - Vitamin C serum: 365–540 days
        - Retinol: 365 days
        - Moisturizers: 540–720 days
        - Sunscreen: 365–540 days
        - Facewash / Cleanser: 540–900 days
        - Haircare (shampoo/conditioner): 720–1080 days
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.2
        )
        val = response.choices[0].message["content"].strip()
        return int(float(val))
    except:
        return 540  # safe fallback

@st.cache_data(show_spinner=False)
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



# cohort monthly stub
@st.cache_data(show_spinner=False)
def compute_cohort_monthly(df, mapping):
    s_map = mapping.get('sales',{}); date_col = s_map.get('date'); qty = s_map.get('quantity'); price = s_map.get('price')
    if date_col in df.columns and qty in df.columns and price in df.columns:
        d = df.copy(); d[date_col]=pd.to_datetime(d[date_col], errors='coerce'); d['month']=d[date_col].dt.to_period('M'); d['revenue']=to_num(d[qty])*to_num(d[price])
        monthly = d.groupby('month').agg({'revenue':'sum','order_id':'nunique'}).reset_index().sort_values('month'); monthly['month']=monthly['month'].astype(str)
        return monthly
    return None



def make_json_safe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    return obj


# build payload
@st.cache_data(show_spinner=False)
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
    
    return kpi

    # Always store the raw text so UI can display summary even if JSON not found
    st.session_state["ai_commentary"] = raw_text

    # Try extracting JSON (non-blocking)
    try:
        parsed = extract_json_from_ai(raw_text) or {}
        if parsed:
            st.session_state["ai_advisor"] = parsed
        else:
            # clear only the ai_advisor keys we populate (do not remove other keys)
            # leave previous advisor if you want persistence or set to empty to indicate parse fail:
            st.session_state["ai_advisor"] = {}
    except Exception as e:
        # do not block: keep raw_text display
        st.warning("AI JSON parsing failed: " + str(e))
        st.session_state["ai_advisor"] = {}

    return True




# ---------------- UI ----------------

def render_compact_chart(fig):
    """Ensures consistent, compact charts in Streamlit."""
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


st.set_page_config(page_title='SmartBrain Founder V8', layout='wide')
st.title('🚀 SmartBrain — Founder-Grade Insights (V8)')

uploaded = st.file_uploader('Upload combined CSV/Excel (sales+marketing+inventory+master)', type=['csv','xlsx'])
use_example = st.checkbox('Use example file (recommended)', value=True)
if uploaded is None and not use_example: st.info('Upload a file or use example to test.')



df=None
if uploaded is not None:
    fname=uploaded.name.lower()
    if fname.endswith('.csv'):
        try: df=pd.read_csv(uploaded)
        except: df=pd.read_csv(uploaded, sep=';', engine='python')
    else: df=pd.read_excel(uploaded)
elif use_example:
    if os.path.exists(EXAMPLE_LOCAL_PATH): df=pd.read_excel(EXAMPLE_LOCAL_PATH); st.success(f'Loaded example: {EXAMPLE_LOCAL_PATH} (url placeholder: {EXAMPLE_FILE_URL})')
    else: st.error(f'Example not found at {EXAMPLE_LOCAL_PATH}')

if df is not None:
   
    
  


  
    df.columns = df.columns.str.strip().str.lower()
   
    mapping, domains = detect_domains(df)
    if mapping.get('inventory', {}).get('sku') == mapping.get('sales', {}).get('sku'):
        for cand in ['inventory_sku', 'sku_inventory', 'item_sku', 'inv_sku']:
            if cand in df.columns:
                mapping['inventory']['sku'] = cand
                break
    debug = st.sidebar.checkbox('Developer mode (show mapping)', value=False)
    compact = st.sidebar.checkbox('Compact View (smaller charts)', value=True)
    if compact: CHART_SZ=(6.5,2.6); MED_SZ=(6.5,3.4)
    else: CHART_SZ=(10,4.5); MED_SZ=(10,5)
    if debug: st.write('Detected domains:', domains); st.json(mapping)

    # --- Monkey-patched backend-driven version of build_kpi_payload ---
    import requests
    import json
    import hashlib
    from io import BytesIO

    BACKEND_URL = "http://127.0.0.1:8000"   # change when deployed

    def _smartbrain_backend_call(df, uploaded_file=None):
        """Send uploaded file to backend and return JSON payload."""
        if uploaded_file is None:
            st.error("No uploaded file found.")
            return {"kpis": {}, "_meta": {"confidence": "Error"}}

        # Prepare file bytes
        if hasattr(uploaded_file, "getvalue"):
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
            file_like = BytesIO(file_bytes)
        else:
            # Example file fallback
            file_bytes = open(uploaded_file, "rb").read()
            filename = "example.xlsx"
            file_like = BytesIO(file_bytes)

        files = {"file": (filename, file_like, "application/octet-stream")}

        try:
            resp = requests.post(f"{BACKEND_URL}/kpi/build", files=files, timeout=120)
            if resp.ok:
                return resp.json()
            else:
                st.error(f"Backend error: {resp.status_code}")
                st.code(resp.text)
                return {"kpis": {}, "_meta": {"confidence": "Backend error"}}
        except Exception as e:
            st.error("Backend unreachable.")
            st.code(str(e))
            return {"kpis": {}, "_meta": {"confidence": "Backend unreachable"}}

    # OVERRIDE the existing build_kpi_payload with backend call
    _original_build_kpi_payload = build_kpi_payload

    def build_kpi_payload(df):
        """Override: always fetch processed KPIs from backend instead of local engine."""
        if uploaded is not None:
            return _smartbrain_backend_call(df, uploaded_file=uploaded)
        elif use_example:
            return _smartbrain_backend_call(df, uploaded_file=EXAMPLE_LOCAL_PATH)
        else:
            # Fallback to local logic only if backend is not available
            return _original_build_kpi_payload(df)

    # --- END of monkey patch ---

    
    kpi_payload = build_kpi_payload(df)
    tab1, tab2, tab3 = st.tabs(['📄 Founder Insights','📊 Visual Insights','🧪 Simulations — Spend Reallocation (Quick)'])
    with tab1:
        st.header('Founder Summary')
        if st.button("⚡ Generate AI Commentary"):
            with st.spinner("AI analyst reviewing your KPIs..."):
                try:
                    resp = requests.post(
                        "http://127.0.0.1:8000/ai/commentary",
                        json={"kpi_payload": kpi_payload},
                        timeout=120
                    )

                    # 🔥🔥 CRITICAL: SHOW EXACT BACKEND RESPONSE 🔥🔥
                    st.error("RAW BACKEND RESPONSE:")
                    st.json(resp.json())  # <--- This is what I need to see


                    if resp.ok:
                        commentary = resp.json().get("commentary", "")
                        st.session_state["ai_commentary"] = commentary
                        st.success("AI commentary updated.")
                    else:
                        st.error(f"Backend returned {resp.status_code}")
                        st.write(resp.text)
                except Exception as e:
                    st.error(f"Failed to get AI commentary: {e}")

        # Immediately render whatever is currently in session_state (if any)
        if st.session_state.get("ai_commentary"):
            # Use st.markdown for richer formatted content
            st.markdown(st.session_state["ai_commentary"])
        else:
            st.info("AI commentary not generated yet. Click ⚡ Generate AI Commentary.")


       
        # ---------- Executive Snapshot Bar (Founder view) ----------
        st.markdown("### 📌 Executive Snapshot — At a glance")

        kpis = kpi_payload.get('kpis', {})
        financials = kpis.get('financials', {})
        sales = kpis.get('sales', {})
        marketing = kpis.get('marketing', {})
        inventory_health = kpis.get('inventory_health', {})

        revenue = financials.get('total_revenue') or sales.get('total_revenue') or 0
        gross_margin_pct = financials.get('gross_margin_pct') if financials else None
        aov = sales.get('aov') or None
        roas = marketing.get('marketing_roas') if marketing else None
        cogs = financials.get('total_cost')
        ad_spend = marketing.get('marketing_total_spend')
        contribution = None
        if revenue is not None and cogs is not None:
            if ad_spend is not None:
                contribution = revenue - cogs - ad_spend
            else:
                contribution = revenue - cogs

        # Beauty KPIs
        sku_revenue = sales.get('revenue_per_sku') or {}
        top_sku_pct = None
        if sku_revenue:
            total_rev = sum(sku_revenue.values()) or 1
            top_sku = max(sku_revenue, key=sku_revenue.get)
            top_sku_pct = round((sku_revenue[top_sku] / total_rev) * 100,2)

        inv_expiry_rows = kpi_payload.get('kpis', {}).get('expiry_warnings', []) or []
        inventory_writeoff_risk = sum([r.get('potential_writeoff',0) or 0 for r in inv_expiry_rows]) if inv_expiry_rows else 0

        valuation = kpis.get('valuation', {})
        rpr = valuation.get('repurchase_rate') if valuation else None

        ma = kpis.get('marketing_attribution', {})
        new_roas = ma.get('new_roas') if ma else None
        ret_roas = ma.get('retargeting_roas') if ma else None

        doc_rows = inventory_health.get('rows') or []
        doc_sl_vals = []
        if doc_rows:
            for r in doc_rows:
                expiry_match = next((e for e in inv_expiry_rows if e.get('sku')==r.get('sku')), {})
                shelf = expiry_match.get('shelf_life_days')
                days_of_cover = r.get('days_of_cover')
                if shelf and days_of_cover is not None:
                    doc_sl_vals.append(min(days_of_cover, shelf))
                elif days_of_cover is not None:
                    doc_sl_vals.append(days_of_cover)
        doc_sl_median = int(sorted(doc_sl_vals)[len(doc_sl_vals)//2]) if doc_sl_vals else None

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Revenue (₹)", f"₹{int(revenue)}" if revenue else "N/A")
        col2.metric("Gross Margin %", f"{gross_margin_pct}%" if gross_margin_pct is not None else "N/A")
        col3.metric("AOV (₹)", f"₹{aov}" if aov else "N/A")
        col4.metric("Blended ROAS", f"{round(roas,2)}" if roas is not None else "N/A")
        col5.metric("Contribution (₹)", f"₹{int(contribution)}" if isinstance(contribution,(int,float)) else "N/A")

        col6, col7, col8, col9, col10 = st.columns(5)
        col6.metric("Hero SKU %", f"{top_sku_pct}%" if top_sku_pct is not None else "N/A")
        col7.metric("Inventory at Risk (₹)", f"₹{int(inventory_writeoff_risk)}" if inventory_writeoff_risk else "₹0")
        col8.metric("Repurchase Rate", f"{round(rpr*100,2)}%" if isinstance(rpr,(int,float)) else "N/A")
        col9.metric("New ROAS / Ret. ROAS", f"{round(new_roas,2) if new_roas else 'N/A'} / {round(ret_roas,2) if ret_roas else 'N/A'}")
        col10.metric("DOC-SL (median days)", f"{doc_sl_median} days" if doc_sl_median else "N/A")

        # ---- Platform fees & shelf life (user-sourced only) ----
        st.markdown("### 📦 Platform Fees (user-provided) — required for channel profitability")
        dataset_pf = kpi_payload['kpis'].get('platform_fees', {}) or {}
        if dataset_pf:
            # normalize display
            pf_rows = []
            for ch, fee in dataset_pf.items():
                pf_rows.append({"channel": ch, "platform_fee_pct": f"{round(float(fee)*100,2)}%"})
            st.table(pd.DataFrame(pf_rows))
        else:
            st.error("Not enough data — provide platform_fee (column) per channel in the dataset to compute channel profitability & reallocation.")

        st.markdown("### 🧪 Shelf-life (user-provided) — required for expiry risk")
        shelf_map = kpi_payload['kpis'].get('shelf_life_map', {}) or {}
        if shelf_map:
            sl_rows = [{"sku": sku, "shelf_life_days": days if days is not None else "N/A (user value missing)"} for sku, days in shelf_map.items()]
            st.table(pd.DataFrame(sl_rows))
        else:
            st.info("Shelf-life not provided. To compute expiry risk, upload a 'shelf_life_days' (or 'shelf_life' / 'expiry_days') column mapping SKUs to shelf life in days.")
        
        # Preview — first 10 rows (founder tab)
        st.subheader('Preview — first 10 rows')
        st.dataframe(df.head(10))

        st.markdown(f"**🧾 Overall Confidence:** {kpi_payload.get('_meta',{}).get('confidence')}")
        # Valuation KPIs
        st.markdown('## 💰 Valuation KPIs'); val=kpi_payload['kpis'].get('valuation',{})
        c1,c2,c3,c4 = st.columns(4); c1.metric('AOV', val.get('AOV') or 'N/A'); c2.metric('CAC', val.get('CAC') or 'N/A'); c3.metric('LTV', val.get('LTV') or 'N/A'); c4.metric('LTV:CAC', val.get('LTV_CAC') or 'N/A')
        for k,msg in val.get('messages',{}).items(): st.caption(f"{k}: {msg}")
        # P&L
        st.markdown('## 📊 Mini P&L Snapshot'); 
        # Mini P&L founder-friendly
        pnl = kpi_payload.get('kpis', {}).get('financials')
        marketing_k = kpi_payload.get('kpis', {}).get('marketing')
        if pnl:
            revenue = pnl.get('total_revenue')
            cogs = pnl.get('total_cost')
            ad_spend = marketing_k.get('marketing_total_spend') if marketing_k else None
            if revenue is not None and cogs is not None:
                net_after_cogs = revenue - cogs
                if ad_spend is not None:
                    net_after_ads = net_after_cogs - ad_spend
                    st.markdown(f"**P&L Snapshot:** Revenue ₹{revenue} → COGS ₹{cogs} → Gross ₹{int(net_after_cogs)} → Ad Spend ₹{int(ad_spend)} → Contribution ₹{int(net_after_ads)}")
                else:
                    st.markdown(f"**P&L Snapshot:** Revenue ₹{revenue} → COGS ₹{cogs} → Gross ₹{int(net_after_cogs)} (Ad spend missing)")

        

        # Spend Reallocation (AI-Driven Qualitative Guidance)
        st.markdown("### 🔁 Spend Reallocation – AI Analyst Recommendation")

        attr = kpi_payload['kpis'].get("marketing_attribution", {})
        platform_fees = kpi_payload['kpis'].get("platform_fees", {})


        channel_table = attr.get("channel_table", [])

        # Require platform fees to exist
        if not platform_fees or len(platform_fees) < 2:
            st.info("Not enough data — please upload platform fee per channel to enable AI spend guidance.")
        else:
            # Build a compact input summary for AI
            try:
                
                gm = kpi_payload['kpis'].get("financials", {}).get("gross_margin_pct")

                
                # Send request to backend AI engine
                spend_resp = requests.post(
                    "http://127.0.0.1:8000/ai/spend",
                    json={
                        "channel_table": channel_table,
                        "platform_fees": platform_fees,
                        "gross_margin": gm
                    },
                    timeout=60
                )
                if spend_resp.ok:
                    guidance = spend_resp.json().get("guidance", "")
                    st.markdown(
                        f"""
                        <div style="
                            padding:15px;
                            border:1px solid #e0e0e0;
                            border-radius:10px;
                            background:#f7f7f7;
                        ">
                            {guidance}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error(f"Backend returned {spend_resp.status_code}")
                    st.write(spend_resp.text)

            except Exception as e:
                st.error(f"AI guidance failed: {e}")
            
# Recommendations
        st.markdown('## 🔥 Recommendations (senior beauty analyst voice)'); recs=[]
        invh = kpi_payload['kpis'].get('inventory_health',{})
        # ----- EXPIRY RISK ENGINE -----
        try:
            expiry_warnings = compute_expiry_warnings(df, mapping, invh)
            kpi_payload['kpis']['expiry_warnings'] = expiry_warnings
        except Exception as e:
            kpi_payload['kpis']['expiry_warnings'] = {'error': str(e)}

        if invh.get('error'): st.error(invh.get('error'))
        else:
            for r in invh.get('rows',[]): 
                if r.get('flag')=='overstock': recs.append({'title':f"Clear overstock for {r['sku']}",'priority':'High','score':95,'details':f"{r['sku']} has {r['on_hand']} units ({r['days_of_cover']} days). Run targeted promos & pause purchases."})
        marketing=kpi_payload['kpis'].get('marketing',{})
        if marketing and marketing.get('marketing_roas') is not None and marketing['marketing_roas']<1.0: recs.append({'title':'Improve marketing efficiency','priority':'Medium','score':70,'details':'Review creatives & targeting. Reallocate conservatively to higher effective-ROAS channels.'})
        try:
            sku_col = mapping['sales'].get('sku'); qty_col = mapping['sales'].get('quantity'); price_col = mapping['sales'].get('price')
            if sku_col in df.columns and qty_col in df.columns and price_col in df.columns:
                sku_revs = df.groupby(sku_col).apply(lambda x: (to_num(x[qty_col]).fillna(0)*to_num(x[price_col]).fillna(0)).sum()).sort_values(ascending=False)
                if not sku_revs.empty and sku_revs.iloc[0]/sku_revs.sum() > 0.3: recs.append({'title':'Reduce SKU concentration risk','priority':'Medium','score':60,'details':'Create bundles around lower-performing SKUs and cross-sell with hero SKU.'})
        except: pass
        for r in sorted(recs, key=lambda x: -x['score']): st.markdown(f"**🔷 {r['title']} — {r['priority']}**  \\n- 🎯 Action: {r['details']}")
        # Marketing attribution
        st.subheader('📣 Marketing Attribution (Channel & Campaign)')
        ma = kpi_payload['kpis'].get('marketing_attribution',{})
        if ma.get('warnings'): 
            for w in ma.get('warnings'): st.warning(w)
        ch = ma.get('channel_table',[])
        if ch: st.dataframe(pd.DataFrame(ch))
        else: st.info('No channel-level marketing data found.')
        camp = ma.get('campaign_table',[])
        if camp: st.markdown('Campaign-level attribution'); st.dataframe(pd.DataFrame(camp))
        else: st.info('No campaign-level data. Upload campaign column to enable campaign ROAS.')
        # ---------- SKU clickable cards (Option 2: larger cards with icons) ----------
        # ---------- Safe SKU revenue computation ----------
        try:
            s_map = mapping.get('sales', {})
            sku_col = s_map.get('sku')
            qty_col = s_map.get('quantity')
            price_col = s_map.get('price')

            if sku_col in df.columns and qty_col in df.columns and price_col in df.columns:
                df['_qty_'] = to_num(df[qty_col]).fillna(0)
                df['_price_'] = to_num(df[price_col]).fillna(0)
                df['_rev_'] = df['_qty_'] * df['_price_']
        
                sku_revs = df.groupby(sku_col)['_rev_'].sum().sort_values(ascending=False)
            else:
                sku_revs = None

        except Exception as e:
            sku_revs = None
            st.warning(f"SKU grouping failed: {e}")

        st.subheader('🛒 SKU quick view (click a SKU card to open details)')

        try:
            sales_kpis = kpi_payload.get('kpis', {}).get('sales', {}) or {}
            units_per_sku = sales_kpis.get('units_per_sku', {}) or {}
            if units_per_sku:
                skus_sorted = sorted(units_per_sku.items(), key=lambda x: x[1], reverse=True)
                top_skus = skus_sorted[:12]  # top 12 by units
                cols = st.columns(4)
                for idx, (sku, qty) in enumerate(top_skus):
                    col = cols[idx % 4]
                    with col:
                        # Larger clickable card styling using a button with emoji. Text only (no revenue).
                        label = f"🧴  {sku}"
                        clicked = st.button(label, key=f"sku_card_{sku}", use_container_width=True)
                        if clicked:
                            st.session_state['selected_sku'] = sku
            else:
                st.info('SKU data not available (ensure sales quantity/sku columns exist).')
        except Exception as e:
            st.warning('SKU quick view unavailable: ' + str(e))

        # Show selected SKU details below the cards
        selected = st.session_state.get('selected_sku')
        if selected:
            st.markdown(f"### 🔍 Details — {selected}")
            # derive fields safely from mapping and kpi_payload; do not recompute core KPIs
            s_map = mapping.get('sales', {})
            qty_col = s_map.get('quantity'); price_col = s_map.get('price'); sku_col = s_map.get('sku')
            # sales metrics from kpi_payload if available
            sales_kpis = kpi_payload.get('kpis', {}).get('sales', {})
            units_map = sales_kpis.get('units_per_sku', {})
            units = int(units_map.get(selected)) if units_map and selected in units_map else 'N/A'
            # revenue: try read from precomputed sku revenue table if present in kpi_payload
            revenue = None
            if isinstance(sku_revs, pd.Series):
                revenue = sku_revs.get(selected)   
            # inventory expiry info (if exists)
            inv_rows = kpi_payload.get('kpis', {}).get('expiry_warnings', []) or []
            sku_expiry = next((r for r in inv_rows if r.get('sku') == selected), None)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric('Units sold', units)
            col2.metric('Revenue', f"₹{int(revenue)}" if isinstance(revenue, (int,float)) else revenue)
            if sku_expiry:
                col3.metric('On hand', sku_expiry.get('on_hand'))
                col4.metric('Expiry write-off (est)', f"₹{sku_expiry.get('potential_writeoff')}")
            else:
                col3.metric('On hand', 'N/A')
                col4.metric('Expiry Flag', 'N/A')
            # extra recommended action
            if sku_expiry and sku_expiry.get('potential_writeoff') and sku_expiry.get('potential_writeoff') > 0:
                st.warning(f"Expiry risk — {selected}: potential write-off ₹{sku_expiry.get('potential_writeoff')}. Suggest bundle or promo to clear.")



        # Inventory health table / value
        st.subheader('📦 Inventory Health')

        invh = kpi_payload['kpis'].get('inventory_health', {})

        if invh.get('error'):
            st.error(invh['error'])
        else:
            st.dataframe(pd.DataFrame(invh.get('rows', [])))

        if invh.get('inventory_value'):
            st.markdown("Inventory value (per SKU)")
            st.dataframe(pd.DataFrame(invh['inventory_value']))

    with tab2:
        st.header('Visual Insights — Executive Summary')
        # executive bullets
        exec_lines = []
        ma = kpi_payload['kpis'].get('marketing_attribution',{})
        ch = ma.get('channel_table',[])
        if ch:
            best = max([c for c in ch if c.get('roas') is not None], key=lambda x:x.get('roas',0)) if any(c.get('roas') is not None for c in ch) else None
            worst = min([c for c in ch if c.get('roas') is not None], key=lambda x:x.get('roas',0)) if any(c.get('roas') is not None for c in ch) else None
            if best: exec_lines.append(f"Best channel by ROAS: {best['channel']} (ROAS: {best['roas']})")
            if worst: exec_lines.append(f"Worst channel by ROAS: {worst['channel']} (ROAS: {worst['roas']})")
        cohort = kpi_payload['kpis'].get('cohort_monthly', [])

        # Convert to DataFrame if backend returned a list
        if isinstance(cohort, list):
            cohort = pd.DataFrame(cohort)

        if cohort is not None and not cohort.empty:
            last = cohort.tail(1)['revenue'].values[0]
            exec_lines.append(f"Last month revenue: {last}")
        for l in exec_lines: st.markdown(f"- {l}")
        st.markdown('---')
        # Channel ROAS chart (compact)
        st.subheader('Channel ROAS (compact)')
        if ch:
            ch_df = pd.DataFrame(ch)
            fig, ax = plt.subplots(figsize=CHART_SZ); ch_df.plot(kind='bar', x='channel', y='roas', ax=ax, legend=False)
            ax.set_ylabel('ROAS'); ax.set_xlabel('Channel'); ax.set_title('Channel ROAS'); ax.grid(axis='y', alpha=0.3)
            for p in ax.patches:
                h = p.get_height(); ax.annotate(str(round(h,2)) if h else 'N/A', (p.get_x()+p.get_width()/2., h), ha='center', va='bottom', fontsize=8)
            render_compact_chart(fig)
        else: st.info('No channel data to chart.')
        # CPA vs CTR bubble chart (compact)
        st.subheader('CPA vs CTR (bubble) — spend sized bubble')
        if ch:
            ch_df = pd.DataFrame(ch); ch_df['cpa_num'] = ch_df['cpa'].apply(lambda x: x if isinstance(x,(int,float)) else np.nan); ch_df['ctr_num'] = ch_df['ctr'].apply(lambda x: x if isinstance(x,(int,float)) else np.nan)
            sizes = (ch_df['spend'].fillna(0) / max(1, ch_df['spend'].max())) * 900
            fig2, ax2 = plt.subplots(figsize=CHART_SZ); ax2.scatter(ch_df['cpa_num'], ch_df['ctr_num'], s=sizes, alpha=0.6)
            for i,row in ch_df.iterrows(): ax2.annotate(row['channel'], (row['cpa_num'], row['ctr_num']), fontsize=8)
            ax2.set_xlabel('CPA'); ax2.set_ylabel('CTR'); ax2.grid(alpha=0.2); render_compact_chart(fig2)
        else: st.info('No channel-level metrics for bubble chart.')
        # SKU revenue distribution (medium)
        st.subheader('Top SKU revenue (top 20)')
        if sku_revs is not None and not sku_revs.empty:
            fig3, ax3 = plt.subplots(figsize=MED_SZ); sku_revs.head(20).plot(kind='bar', ax=ax3); ax3.set_ylabel('Revenue'); render_compact_chart(fig3)
        else: st.info('SKU revenue not available for chart.')
        # DOC chart (strict)
        st.subheader('Days of Cover per SKU (strict)')
        invh = kpi_payload['kpis'].get('inventory_health', {})
        doc_rows = invh.get('rows', [])

        if doc_rows:
            doc_df = pd.DataFrame(doc_rows)

            # SAFE DOC FOR CHART
            safe_doc = (
                doc_df['days_of_cover']
                .fillna(0)                       # None → 0
                .clip(upper=DOC_CAP_DAYS)        # cap to UI limit
            )

            fig4, ax4 = plt.subplots(figsize=CHART_SZ)
            ax4.bar(doc_df['sku'], safe_doc)
            ax4.set_ylabel("Days of Cover")
            render_compact_chart(fig4)

        else:
            if invh.get('error'):
                st.error(invh['error'])
            else:
                st.info("No inventory days-of-cover data.")

    with tab3:
        
        try:
            st.header("🧪 Simulations — Spend Reallocation")
            
            ma = kpi_payload.get('kpis', {}).get('marketing_attribution', {})
            channels = [c.get('channel') for c in (ma.get('channel_table') or [])]
            if not channels or len(channels) < 2:
                st.info('Not enough channel-level data to run simulations. Ensure marketing attribution channel_table is present.')
            else:
                col_from, col_to = st.columns(2)
                from_ch = col_from.selectbox('From channel', channels, index=0)
                to_ch = col_to.selectbox('To channel', [c for c in channels if c != from_ch], index=0)
                pct = st.slider('Percent of FROM-channel spend to shift (%)', 0, 100, 20)
                # compute estimate using provided KPIs only
                channel_table = ma.get('channel_table') or []
                from_row = next((r for r in channel_table if r.get('channel') == from_ch), None)
                to_row = next((r for r in channel_table if r.get('channel') == to_ch), None)
                if from_row and to_row:
                    from_spend = from_row.get('spend') or 0.0
                    shift_amount = from_spend * (pct / 100.0)
                    # use ROAS from payload (do not recompute)
                    from_roas = from_row.get('roas') or 0.0
                    to_roas = to_row.get('roas') or 0.0
                    rev_delta = shift_amount * (to_roas - from_roas)
                    gm_pct = None
                    fin = kpi_payload.get('kpis', {}).get('financials', {})
                    gm_pct = fin.get('gross_margin_pct') if fin else None
                    profit_delta = rev_delta * (gm_pct / 100.0) if gm_pct is not None else None
                    st.markdown(f"**Shift amount:** ₹{round(shift_amount,2)}")
                    st.markdown(f"**Revenue delta (estimate):** ₹{round(rev_delta,2)}")
                    st.markdown(f"**Profit delta (estimate):** {('₹'+str(round(profit_delta,2))) if profit_delta is not None else 'N/A — gross_margin_pct missing'}")
                    # quick human summary
                    summary = f"Shifting {pct}% of {from_ch} spend (₹{round(shift_amount,2)}) to {to_ch} changes revenue by ₹{round(rev_delta,2)}."
                    if profit_delta is not None:
                        if profit_delta < 0:
                            summary += " This is expected to reduce profit — not recommended unless strategic reasons exist."
                        else:
                            summary += " This is expected to increase profit."
                    st.info(summary)
                else:
                    st.info('Channel rows missing detailed spend/roas — cannot simulate.')
        except Exception as e:
            st.warning('Simulations UI failed: ' + str(e))


        # ---------- Creative fatigue signal ----------
        m = kpi_payload.get('kpis', {}).get('marketing', {})
        ma = kpi_payload.get('kpis', {}).get('marketing_attribution', {})
        # try to read CTR if available
        try:
            # some payloads may include global clicks/impressions or per-channel metrics
            global_clicks = m.get('clicks') if isinstance(m, dict) and m.get('clicks') is not None else None
            global_impressions = m.get('impressions') if isinstance(m, dict) and m.get('impressions') is not None else None
            if global_clicks is not None and global_impressions is not None and global_impressions > 0:
                ctr = (global_clicks / global_impressions) * 100.0
                if ctr < 1.0:
                    st.warning(f"Creative fatigue detected — CTR is low: {round(ctr,2)}%. Refresh creatives.")
                else:
                    st.info(f"CTR: {round(ctr,2)}%")
            else:
                # also check channel table for CTR per channel if present
                channel_rows = ma.get('channel_table', []) if isinstance(ma, dict) else []
                for ch in channel_rows:
                    if ch.get('clicks') is not None and ch.get('impressions') is not None and ch.get('impressions') > 0:
                        ch_ctr = (ch.get('clicks') / ch.get('impressions')) * 100.0
                        if ch_ctr < 1.0:
                            st.warning(f"Creative fatigue on {ch.get('channel')} — CTR {round(ch_ctr,2)}%")
        except Exception:
            pass
else:
    st.info('Upload a combined CSV/Excel or select example to begin.')