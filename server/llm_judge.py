"""
server/llm_judge.py — LLM-based rubric judge for WhyDidItFail diagnoses.

The judge() function scores an agent's submit_diagnosis action using the
same LLM (via the already-created OpenAI client) as a secondary evaluator.
Score is in [0, 1]; the caller blends it with the keyword score.
"""

import json
import textwrap
from typing import Any, Dict, Optional

from openai import OpenAI


_JUDGE_SYSTEM = textwrap.dedent("""
    You are a strict ML-failure grader. You will be given:
    - The CORRECT answer (failure mode label, suggested fix, reasoning guidelines).
    - The AGENT's submitted diagnosis (label, suggested_fix, reasoning).

    Score the agent's submission on a 0.0–1.0 scale:
    1.0 — exact label match AND reasoning cites real numeric evidence AND fix is appropriate
    0.7 — exact label match AND reasoning is plausible but lacks specific numbers
    0.4 — close but label is slightly wrong (e.g. "overfitting" vs "missing regularization")
    0.1 — wrong label or no reasoning

    Reply with ONLY a JSON object: {"score": <float 0-1>, "rationale": "<one sentence>"}
    No markdown, no code fences.
""").strip()


def judge(
    client: OpenAI,
    model: str,
    diagnosis: str,
    reasoning: Optional[str],
    suggested_fix: Optional[str],
    scenario: Dict[str, Any],
    inspection_order: list,
) -> Optional[float]:
    """
    Call the LLM to evaluate the agent's diagnosis quality.

    Returns a float in [0, 1], or None if the call fails.
    """
    correct_label = scenario.get("correct_diagnosis", "")
    description   = scenario.get("description", "")

    agent_submission = json.dumps({
        "diagnosis":     diagnosis,
        "reasoning":     reasoning or "",
        "suggested_fix": suggested_fix or "",
        "inspection_order": inspection_order,
    }, indent=2)

    user_msg = textwrap.dedent(f"""
        Training failure description:
        {description}

        Correct failure mode: "{correct_label}"

        Agent submission:
        {agent_submission}

        Rate the agent's submission.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        raw  = float(data.get("score", 0.0))
        return round(max(0.0, min(1.0, raw)), 4)
    except Exception:
        return None
