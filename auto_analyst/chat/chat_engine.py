import os
from groq import Groq
from dotenv import load_dotenv
from chat.context_builder import build_system_prompt

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def get_groq_client():
    """Initialize Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")
    return Groq(api_key=api_key)


def chat_with_data(df, conversation_history, user_message, profile=None):
    """
    Send a user message to Groq LLM with full data context.

    Args:
        df: The uploaded DataFrame
        conversation_history: List of {"role": "user"/"assistant", "content": "..."} dicts
        user_message: The latest user question
        profile: Optional profile dict from profiler

    Returns:
        assistant_reply (str), updated_history (list)
    """
    client = get_groq_client()
    system_prompt = build_system_prompt(df, profile)

    # Build messages for API
    messages = []
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.3,
            max_tokens=1024,
        )
        assistant_reply = response.choices[0].message.content

    except Exception as e:
        assistant_reply = f"⚠️ Error communicating with AI: {str(e)}"

    # Update history
    updated_history = conversation_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_reply},
    ]

    return assistant_reply, updated_history


def generate_ai_narrative(df, profile, analysis_summary, profile_only=False):
    """
    Generate a human-readable narrative summary of the analysis
    using Groq LLM. Used for the report's executive summary.
    """
    client = get_groq_client()
    system_prompt = build_system_prompt(df, profile)

    if profile_only:
        user_prompt = f"""
Based on the dataset I've described, write a professional executive summary.
Include:
1. What kind of data this appears to be
2. Key highlights (row count, column count, quality score)
3. Most important columns and their distributions
4. Any data quality issues (missing values, duplicates)
5. 2-3 actionable recommendations

Keep it concise — max 300 words. Use plain business language.
"""
    else:
        user_prompt = f"""
Based on the dataset and analysis results below, write a professional executive summary report.

Analysis Summary:
{analysis_summary}

Include:
1. What kind of data this is
2. Key statistical highlights
3. Most interesting patterns or relationships found
4. Cluster/segment insights (if available)
5. Top 3-5 actionable business recommendations

Keep it under 400 words. Professional, clear, and insightful.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI narrative: {str(e)}"
