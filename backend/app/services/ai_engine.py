import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_commentary(kpi_payload: dict):
    """
    Backend version of AI commentary:
    Takes the KPI payload and returns structured insights.
    """
    try:
        prompt = f"""
You are a seasoned senior D2C + Beauty & Personal Care founder-analyst 
with 15+ years of operational experience across Amazon Beauty, Nykaa, Myntra, 
Flipkart, and D2C Shopify brands.

Your job:
➡ Interpret the KPI JSON  
➡ Diagnose the real business situation  
➡ Produce commentary EXACTLY in the structured **PDF report format** below  
➡ Use founder-grade reasoning (no fluff, no hallucination)  
➡ Ensure every section is filled as per rules

===================================================
🎨 STRICT OUTPUT FORMAT (FOLLOW EXACTLY)
===================================================

## 🟦 FOUNDER SUMMARY (ONE LINER)
Write a sharp, investor-ready, one-sentence summary covering:
- Biggest risk  
- Biggest opportunity  
- Immediate founder priority  

---

## ⚡ QUICK WINS (0–7 DAYS)
For each item, use this template:

### 🔹 <Title>
- **KPI Evidence:** <Pull from kpi_payload>  
- **Action:** <What should the founder do this week?>  
- **Expected Impact:** <Revenue / ROAS / expiry / retention / contribution impact>  
- **Confidence:** High / Medium / Low  

Write **3–5 recommendations**.

---

## 🚀 MID-TERM OPPORTUNITIES (2–8 WEEKS)
Same bullet structure as above.  
Write **3–4 recommendations**.

---

## 🌱 LONG-TERM STRATEGY (3–12 MONTHS)
Same bullet structure as above.  
Write **2–4 recommendations**.

---

## 🟧 PLATFORM-SPECIFIC PERFORMANCE (Amazon, Nykaa, Shopify)
Interpret `marketing_attribution` using beauty-industry norms:
- Amazon typical 15% fees → ROAS < 1 means margin compression  
- Nykaa expensive beauty CPCs → ROAS < 1 means unprofitable scale  
- Shopify lowest fees → profitability driven by LTV + CAC  

For each channel present in data:
- What is happening?  
- Why?  
- Operational meaning  
- Founder implication  

---

## 🟨 INVENTORY EXPIRY RISK LAYER
For every SKU with risk:

### ⚠ SKU: <sku_name>
- **Evidence:** DOC, Shelf Life, On-Hand, Flag  
- **Risk Level:** High / Medium / Low  
- **Action:** What to do immediately  
- **Expected Write-Off Prevention:** if applicable  

If no risk:
Write: “No high-severity expiry risks detected.”

---

## 🟪 HERO SKU + REPEAT PURCHASE INSIGHTS
Using sales + valuation tables:
- Identify hero SKU (>30–40% revenue)  
- Identify gateway/retention SKU  
- Identify underperforming SKUs  
- Suggest bundles, trials, recurring pack strategy  
- Refer to CAC / LTV / Repurchase Rate if available  
If missing:
Note: “Retention insight limited — missing columns: <list>”

---

## 🟧 SPEND REALLOCATION (QUALITATIVE ONLY)
Use the platform fees + ROAS + CAC + contribution info.

DO NOT calculate numbers.  
Explain:
- Which channel deserves more spend  
- Which channel should be reduced  
- Why (fee-adjusted ROAS logic)  
- Any signs of creative fatigue  

If attribution missing:
Say: “Not enough data for spend reallocation — upload marketing columns.”

---

## 🔴 DATA GAPS & REQUIRED COLUMNS
Scan the JSON.  
List missing columns required for:
- LTV / CAC  
- Shelf life  
- Marketing attribution  
- SKU-level profitability  
- Inventory DOC  

If none missing:
Write: “No critical data gaps detected — dataset is healthy.”

===================================================
🧠 ANALYTICAL RULES YOU MUST FOLLOW
===================================================

You must apply the full intelligence of a marketplace + D2C operator:

1. ROAS < 1 → highlight cash burn risk  
2. Contribution margin must consider:
   revenue – COGS – platform fees – marketing spend  
3. Expiry Risk:
   days_of_cover >> shelf_life → HIGH RISK  
4. SKU Concentration:
   If hero SKU > 35% → flag as concentration risk  
5. Retention:
   Use CAC, AOV, LTV exactly as given  
6. Platform fees:
   Amazon ~15%, Nykaa ~20–25%, Myntra ~20–22%, Shopify ~2%  
7. DO NOT hallucinate:
   If data missing → explicitly call it out  
8. All recommendations must feel:
   ✔ Founder-grade  
   ✔ Board-meeting ready  
   ✔ Crisp, impact-first  

===================================================
INPUT DATA (DO NOT OMIT)
===================================================
{kpi_payload}
"""


        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return {"commentary": res.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}


async def generate_spend_guidance(channel_table, platform_fees, gross_margin):
    """
    Spend reallocation guidance using the ORIGINAL SmartBrain prompt.
    """
    try:
        ai_prompt = f"""
You are an expert performance marketer and a seasoned D2C strategist.

Below is the fee-adjusted performance summary:

Channels:
{channel_table}

Platform Fees:
{platform_fees}

Gross Margin: {gross_margin}

Based on this data:
- DO NOT output numbers (no shift %, no exact rupees)
- DO NOT calculate explicit reallocation moves
- INSTEAD give a qualitative recommendation:
    - Which channel is strongest & weakest?
    - Should founder increase/decrease spend on any channel?
    - What is the rationale?
    - Highlight creative + retention insights if needed.

Keep it crisp, 4–6 lines max.
"""

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a seasoned senior D2C analyst."},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.3,
        )

        return {"guidance": res.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}
