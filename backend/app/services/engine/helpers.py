import pandas as pd
import numpy as np



# ---------------- Utilities ----------------


def to_num(s): return pd.to_numeric(s, errors='coerce')
def safe_sum(df, col): return float(to_num(df[col]).fillna(0).sum()) if col and col in df.columns else 0.0
def platform_fee_for_channel(channel):
    if not isinstance(channel, str): return PLATFORM_FEE_ESTIMATES['default']
    return PLATFORM_FEE_ESTIMATES.get(channel.strip().lower(), PLATFORM_FEE_ESTIMATES['default'])


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
