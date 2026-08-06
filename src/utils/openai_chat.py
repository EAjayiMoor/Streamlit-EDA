import os
import requests


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


SYSTEM_PROMPT = """
You are an NHS elective analytics assistant supporting RTT performance analysis.

Your role:
- Answer questions using only the supplied data context.
- Explain insights clearly for a non-technical NHS / consulting audience.
- Use the actual numbers provided in the context.
- Be concise, structured, and careful.
- If the context does not contain enough information, say so.
- Do not invent causes, targets, benchmarks, or explanations not supported by the data.

Key definitions:
- Backlog means incomplete RTT pathways.
- Demand means New RTT Periods / additions entering the waiting list.
- Throughput means completed RTT pathways, admitted plus non-admitted.
- Net flow means demand minus throughput.
- 0–18 weeks reflects patients within the RTT standard window.
- 18–52 weeks reflects patients waiting beyond the target but under one year.
- 52+ weeks reflects long waits and backlog severity.
- Higher heatmap scores indicate higher relative pressure.
"""


def _extract_response_text(result: dict) -> str:
    output = result.get("output", [])
    if not output:
        return "No response returned."

    for item in output:
        content = item.get("content", [])
        for part in content:
            if part.get("type") == "output_text":
                return part.get("text", "No text returned.")

    return "No text content returned."


def ask_openai_about_data(user_question: str, data_context: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-5.4",
        "input": f"""{SYSTEM_PROMPT}

Data context:
{data_context}

User question:
{user_question}
""",
    }

    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    result = response.json()

    return _extract_response_text(result)
