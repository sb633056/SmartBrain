// src/lib/api.ts (MODIFIED to pass KPI context)

interface AiRequestPayload {
  prompt: string;
  kpi_payload: any; // The full data payload from the backend
}

// Accepts the prompt and the KPI data for LLM context
export async function runSmartBrain(prompt: string, kpiData: any) {
  const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

  const payload: AiRequestPayload = {
    prompt: prompt,
    // Pass the analysis data to the LLM for deep context
    kpi_payload: kpiData || {}, 
  }

  // Assuming your FastAPI LLM endpoint is /ai/commentary
  const res = await fetch(`${API_URL}/ai/commentary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  let text = await res.text();
  console.log("RAW BACKEND RESPONSE:", text);

  if (!res.ok) {
    // Actionable error message
    throw new Error(`AI request failed: ${res.status} - ${text.substring(0, 100)}`);
  }

  try {
    const data = JSON.parse(text);
    return data.commentary || data.output || data.response || data.message || JSON.stringify(data, null, 2);
  } catch (e) {
    return text; // fallback if response is plain text
  }
}