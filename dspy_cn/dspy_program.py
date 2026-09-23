"""
Author: Preethi Gajawada
Institution: Northeastern University
Project: Multilingual Counter-Narrative Generation using DSPy
Description: Defines DSPy signatures and modules for generating counter-narratives.
Year: 2026
"""

# dspy_cn/dspy_program.py
import dspy


# ───────────────────────────────────────────────────────────────
# ENGLISH SIGNATURE
# ───────────────────────────────────────────────────────────────

class EnglishCNSignature(dspy.Signature):
    """
You are generating a COUNTER-NARRATIVE to harmful or hateful speech.

Rules (STRICT):
- Respond in exactly 2-3 sentences.
- Be polite, calm, and respectful.
- Directly address the claim in the hate speech.
- Do NOT lecture.
- Do NOT use long explanations.
- Do NOT include disclaimers.
- Do NOT repeat the hate speech.
- Avoid moralizing phrases like "it's important to remember" or "we should all".

Goal:
- Challenge the harmful idea constructively.
- Promote understanding and empathy.
- Encourage respectful reflection.
"""

    hate_speech = dspy.InputField(
        desc="a hateful or offensive comment"
    )
    counter_narrative = dspy.OutputField(
        desc="a natural English counter-narrative in exactly 2-3 sentences"
    )




# ───────────────────────────────────────────────────────────────
# TAMIL SIGNATURE (INSTRUCTIONS IN ENGLISH)
# ───────────────────────────────────────────────────────────────

class TamilCNSignature(dspy.Signature):
    """
You are required to generate a COUNTER-NARRATIVE responding to hateful or offensive content.

STRICT RULES:
- Write the response in TAMIL.
- Use EXACTLY 2 or 3 sentences only.
- Maintain a polite, calm, and respectful tone.
- Directly address and challenge the hateful statement.
- Do NOT lecture, preach, or give long explanations.
- Do NOT use aggressive, sarcastic, or toxic language.
- Do NOT include generic advice, disclaimers, or moral preaching.
- Stay focused on the specific content of the hate speech.

GOAL:
- Constructively challenge the harmful claim
- Promote empathy and understanding
- Encourage respectful dialogue
"""

    hate_speech = dspy.InputField(desc="a hateful or offensive comment")
    counter_narrative = dspy.OutputField(
        desc="a natural Tamil counter narrative (2-3 sentences)"
    )


# ───────────────────────────────────────────────────────────────
# PROGRAMS
# ───────────────────────────────────────────────────────────────

class EnglishCNProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(EnglishCNSignature)

    def forward(self, hate_speech: str):
        result = self.generate(hate_speech=hate_speech)

        # ✅ CRITICAL FIX: always return structured object
        if isinstance(result, str):
            return dspy.Prediction(counter_narrative=result)

        return result


class TamilCNProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(TamilCNSignature)

    def forward(self, hate_speech: str):
        result = self.generate(hate_speech=hate_speech)

        # ✅ CRITICAL FIX: avoids 'str has no attribute counter_narrative'
        if isinstance(result, str):
            return dspy.Prediction(counter_narrative=result)

        return result