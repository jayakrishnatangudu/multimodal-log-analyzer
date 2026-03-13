"""
llm_interface.py
----------------
Uses the Groq Python SDK with model llama3-8b-8192 to explain a detected
anomaly given the log text and its cosine similarity score as context.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_MODEL = "llama-3.1-8b-instant"


def explain_anomaly(log_text: str, similarity_score: float) -> str:
    """
    Call Groq (llama3-8b-8192) to explain why a log entry was flagged as an
    anomaly and suggest possible remediations.

    Args:
        log_text:         The raw log line / text that was flagged.
        similarity_score: The cosine similarity between the image embedding
                          and the text embedding (float, typically in [-1, 1]).
                          A value below 0.25 triggered the anomaly flag.

    Returns:
        Explanation string (markdown formatted) from the LLM.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to your .env file or environment."
        )

    client = Groq(api_key=api_key)

    system_prompt = (
        "You are an expert Site Reliability Engineer (SRE) and security analyst. "
        "You analyze system logs and cosine similarity scores (threshold > 0.25 is NORMAL, < 0.25 is ANOMALY). "
        "Explain what the log indicates, identify the root cause, and suggest remediation steps. "
        "Be concise, technical, and actionable. Format in markdown.\n\n"
        "Here are three examples of how you should respond:\n\n"
        "--- EXAMPLE 1 (NORMAL) ---\n"
        "Log: INFO: All network connections are healthy and traffic is flowing normally.\n"
        "Similarity Score: 0.2800\n"
        "Assistant: **Status: Normal Operations**\n"
        "This log indicates standard healthy network traffic. The high similarity score confirms it matches baseline expectations. No remediation required.\n\n"
        "--- EXAMPLE 2 (ANOMALY) ---\n"
        "Log: CRITICAL: OOM Killer invoked. Process api-server (pid 4821) terminated. System unstable.\n"
        "Similarity Score: 0.2100\n"
        "Assistant: **Status: Critical Anomaly - Out of Memory**\n"
        "**Analysis:** The Linux kernel's Out-Of-Memory (OOM) killer has terminated the `api-server` process to free up memory.\n"
        "**Root Cause:** A memory leak in the API service or a sudden spike in traffic exceeding provisioned RAM.\n"
        "**Remediation:** 1) Restart the `api-server` systemd service immediately. 2) Check Grafana memory panels to isolate the leak. 3) Temporarily autoscale the node group or increase the Pod memory limit.\n\n"
        "--- EXAMPLE 3 (ANOMALY) ---\n"
        "Log: CRITICAL: Brute-force attack detected. 500 failed SSH logins from 185.220.101.42 in 60s.\n"
        "Similarity Score: 0.1850\n"
        "Assistant: **Status: Security Anomaly - Active SSH Brute-Force**\n"
        "**Analysis:** An attacker is actively trying to guess SSH credentials from IP 185.220.101.42.\n"
        "**Root Cause:** Exposed port 22 to the public internet without adequate rate-limiting or fail2ban.\n"
        "**Remediation:** 1) Immediately block IP 185.220.101.42 using `iptables` or cloud security groups. 2) Ensure password authentication is disabled; mandate SSH keys. 3) Install and configure `fail2ban`."
    )

    user_prompt = (
        f"**Log:** {log_text}\n"
        f"**Similarity Score:** {similarity_score:.4f}\n"
    )

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,  # lowered for more consistent outputs
        max_tokens=512,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Quick smoke-test (requires GROQ_API_KEY in .env)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_log = "CRITICAL: Disk I/O error on /dev/sda1 – bad sector detected at 0x1A3F"
    sample_score = 0.11  # well below the 0.25 threshold

    print(f"Log  : {sample_log}")
    print(f"Score: {sample_score}\n")
    explanation = explain_anomaly(sample_log, sample_score)
    print(explanation)
