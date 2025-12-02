# ------------------- OpenAI Client Init -------------------
import os
from openai import OpenAI

client = None
try:
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_KEY:
        client = OpenAI(api_key=OPENAI_KEY)
except Exception as e:
    print("Failed to initialize OpenAI client:", e)


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






def run_ai_commentary(kpi_payload):
    """
    Robust AI caller:
    - Always stores the full raw text into st.session_state['ai_commentary'] (so UI can show it).
    - Attempts to extract JSON and stores into st.session_state['ai_advisor'] if available.
    - Writes helpful debug messages to UI if something fails.
    """
    import json, os, traceback

    # defensive: ensure session_state keys exist
    if "ai_commentary" not in st.session_state:
        st.session_state["ai_commentary"] = None
    if "ai_advisor" not in st.session_state:
        st.session_state["ai_advisor"] = {}

    # prepare safe payload (use your existing helper)
    try:
        safe_payload = make_json_safe(kpi_payload)
    except Exception as e:
        st.warning(f"Could not make payload safe for AI: {e}")
        safe_payload = kpi_payload  # still try

    # create client (fail gracefully and store message)
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        st.error("OpenAI client unavailable: " + str(e))
        st.session_state["ai_commentary"] = "AI unavailable: missing OpenAI client or API key."
        return False

    # build messages
    messages = [
        {"role": "system", "content": PLATFORM_ANALYST_PROMPT},
        {"role": "user", "content": json.dumps(safe_payload)}
    ]

    # call model
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.2
        )
    except Exception as e:
        st.error("AI call failed: " + str(e))
        st.session_state["ai_commentary"] = "AI call failed: " + str(e)
        return False

    # extract text robustly (many SDK shapes exist)
    raw_text = None
    try:
        # new shape (dict-like)
        raw_text = response.choices[0].message["content"]
    except Exception:
        try:
            # new attribute style
            raw_text = response.choices[0].message.content
        except Exception:
            try:
                # older style
                raw_text = response.choices[0].text
            except Exception:
                raw_text = str(response)

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