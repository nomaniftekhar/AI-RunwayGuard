import os
import json
import base64
import re
from typing import Dict, Any

import gradio as gr
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "RunwayGuard AI"

# Groq multimodal vision model
VISION_MODEL = "qwen/qwen3.6-27b"


# ============================================================
# GROQ CLIENT
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. "
        "Please add your Groq API key as an environment variable."
    )

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# IMAGE ENCODING
# ============================================================

def encode_image(image_path: str) -> str:
    """
    Convert an uploaded image into a Base64 data URL.
    """

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from the model response.
    """

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    # Find JSON object
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        return {
            "fod_present": False,
            "object": "Unknown",
            "description": text,
            "risk": "UNKNOWN",
            "reason": "The AI response could not be parsed."
        }

    try:
        return json.loads(match.group(0))

    except json.JSONDecodeError:

        return {
            "fod_present": False,
            "object": "Unknown",
            "description": text,
            "risk": "UNKNOWN",
            "reason": "The AI response was not valid JSON."
        }


# ============================================================
# GROQ VISION ANALYSIS
# ============================================================

def analyze_image(image_path: str) -> Dict[str, Any]:

    image_data = encode_image(image_path)

    prompt = """
You are an AI aviation safety assistant analyzing an image of an airport runway.

Your task is to determine whether visible Foreign Object Debris (FOD)
may be present on the runway.

FOD can include objects such as:

- Metal pieces
- Tools
- Stones
- Plastic
- Tire fragments
- Cables
- Parts of equipment
- Luggage or other foreign objects
- Other debris that does not belong on the runway

Analyze ONLY what is visually supported by the image.

Return ONLY valid JSON using this structure:

{
    "fod_present": true,
    "object": "description of suspected FOD",
    "description": "short visual description",
    "risk": "LOW/MEDIUM/HIGH",
    "reason": "why this object may be a runway hazard"
}

If no obvious FOD is visible, return:

{
    "fod_present": false,
    "object": "None detected",
    "description": "No obvious foreign object debris is visible.",
    "risk": "LOW",
    "reason": "No visible FOD was identified in the uploaded image."
}

Do not invent objects that cannot be visually supported.
"""

    response = client.chat.completions.create(

        model=VISION_MODEL,

        messages=[
            {
                "role": "user",

                "content": [

                    {
                        "type": "text",
                        "text": prompt
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url": image_data
                        }
                    }
                ]
            }
        ],

        temperature=0.1,

        max_tokens=700
    )

    result_text = response.choices[0].message.content

    return extract_json(result_text)


# ============================================================
# RISK ASSESSMENT
# ============================================================

def calculate_risk(result: Dict[str, Any]) -> str:

    if not result.get("fod_present", False):
        return "LOW"

    ai_risk = str(
        result.get("risk", "MEDIUM")
    ).upper()

    if ai_risk in ["LOW", "MEDIUM", "HIGH"]:
        return ai_risk

    return "MEDIUM"


# ============================================================
# INCIDENT REPORT
# ============================================================

def generate_report(
    detection: Dict[str, Any],
    risk: str
) -> str:

    fod_present = detection.get(
        "fod_present",
        False
    )

    object_name = detection.get(
        "object",
        "Unknown"
    )

    description = detection.get(
        "description",
        "No description available."
    )

    reason = detection.get(
        "reason",
        "No reason provided."
    )

    if fod_present:

        prompt = f"""
Create a concise aviation-style FOD incident report.

Detection:
FOD Present: YES
Object: {object_name}
Description: {description}
Risk Level: {risk}
Reason: {reason}

Use exactly these sections:

INCIDENT STATUS
DETECTED OBJECT
VISUAL ASSESSMENT
RISK LEVEL
RECOMMENDED ACTION

Keep the report professional and concise.

Important:
This is an AI-assisted prototype.
Do not claim that the assessment is aviation-certified.
"""

    else:

        prompt = """
Create a concise runway inspection report.

The uploaded image does not show obvious Foreign Object Debris.

Use exactly these sections:

INCIDENT STATUS
DETECTED OBJECT
VISUAL ASSESSMENT
RISK LEVEL
RECOMMENDED ACTION

State that no obvious FOD was visually identified.
Do not claim that the runway is completely safe.
"""

    response = client.chat.completions.create(

        model=VISION_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=600
    )

    return response.choices[0].message.content


# ============================================================
# MAIN IMAGE FUNCTION
# ============================================================

def runwayguard_image(image_path):

    if image_path is None:

        return (
            "⚠️ Please upload a runway image.",
            "No analysis performed."
        )

    try:

        detection = analyze_image(
            image_path
        )

        risk = calculate_risk(
            detection
        )

        report = generate_report(
            detection,
            risk
        )

        # ----------------------------------------------------
        # Detection result
        # ----------------------------------------------------

        if detection.get(
            "fod_present",
            False
        ):

            status = "🔴 FOD DETECTED"

        else:

            status = "🟢 NO OBVIOUS FOD DETECTED"

        detection_text = f"""
{status}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Object:
{detection.get("object", "Unknown")}

Risk Level:
{risk}

Visual Assessment:
{detection.get("description", "Not available.")}

Reason:
{detection.get("reason", "Not available.")}
"""

        return detection_text, report

    except Exception as e:

        return (
            f"❌ SYSTEM ERROR\n\n{str(e)}",
            "Analysis could not be completed."
        )


# ============================================================
# PROFESSIONAL UI
# ============================================================

CSS = """

/* ==============================
   GLOBAL
   ============================== */

body {
    background: #06111f !important;
}

.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;

    background:
        radial-gradient(
            circle at top right,
            #102b45 0%,
            #06111f 45%
        ) !important;

    color: #e8f1fa !important;

    font-family:
        Arial,
        Helvetica,
        sans-serif !important;
}


/* ==============================
   HEADER
   ============================== */

.header {
    background:
        linear-gradient(
            135deg,
            #0d243b,
            #081522
        );

    border: 1px solid #1c3b56;

    border-radius: 18px;

    padding: 28px 30px;

    margin-bottom: 22px;

    box-shadow:
        0 10px 35px
        rgba(0,0,0,0.25);
}

.logo {
    font-size: 31px;

    font-weight: 800;

    letter-spacing: 2px;

    color: #ffffff;
}

.subtitle {
    color: #91a8bc;

    margin-top: 7px;

    font-size: 14px;
}

.status {
    background: #0a2a20;

    border: 1px solid #1e654a;

    border-radius: 30px;

    padding: 10px 16px;

    text-align: center;

    color: #5ee6a8;

    font-weight: 700;

    margin-top: 8px;
}


/* ==============================
   CARDS
   ============================== */

.card {
    background:
        rgba(10, 27, 43, 0.92);

    border:
        1px solid #1c3953;

    border-radius:
        16px;

    padding:
        20px;

    box-shadow:
        0 8px 25px
        rgba(0,0,0,0.20);
}

.section-title {
    font-size: 18px;

    font-weight: 700;

    color: #ffffff;

    margin-bottom: 12px;
}


/* ==============================
   BUTTON
   ============================== */

.analyze-btn {
    background:
        linear-gradient(
            135deg,
            #1683ff,
            #0864c5
        ) !important;

    color: white !important;

    border:
        none !important;

    border-radius:
        10px !important;

    font-size:
        16px !important;

    font-weight:
        700 !important;

    padding:
        14px !important;

    margin-top:
        12px !important;
}

.analyze-btn:hover {
    background:
        linear-gradient(
            135deg,
            #2691ff,
            #0871df
        ) !important;
}


/* ==============================
   INPUT / OUTPUT
   ============================== */

textarea,
input {
    background:
        #071522 !important;

    color:
        #e6eef7 !important;

    border-color:
        #21415d !important;
}

.output-box textarea {
    min-height:
        300px !important;
}


/* ==============================
   IMAGE
   ============================== */

.image-container {
    border-radius:
        12px !important;

    overflow:
        hidden !important;
}


/* ==============================
   INFO CARDS
   ============================== */

.info-card {
    background:
        #0a1a2a;

    border:
        1px solid #19364f;

    border-radius:
        14px;

    padding:
        18px;

    min-height:
        110px;
}


/* ==============================
   FOOTER
   ============================== */

.footer {
    text-align:
        center;

    color:
        #647d94;

    font-size:
        12px;

    padding:
        20px;

    margin-top:
        20px;
}


/* ==============================
   MOBILE
   ============================== */

@media (max-width: 768px) {

    .logo {
        font-size:
            24px;
    }

    .header {
        padding:
            20px;
    }

}

"""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    css=CSS,
    title=APP_TITLE,
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate"
    )
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    with gr.Row(
        elem_classes="header"
    ):

        with gr.Column(
            scale=5
        ):

            gr.Markdown(
                """
<div class="logo">
✈ RUNWAYGUARD AI
</div>

<div class="subtitle">
AI-Powered Foreign Object Debris Detection & Incident Analysis
</div>
"""
            )

        with gr.Column(
            scale=1
        ):

            gr.Markdown(
                """
<div class="status">
● SYSTEM ONLINE
</div>
"""
            )


    # --------------------------------------------------------
    # MAIN DASHBOARD
    # --------------------------------------------------------

    with gr.Row():

        # ====================================================
        # INPUT PANEL
        # ====================================================

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.Markdown(
                """
<div class="section-title">
📷 RUNWAY IMAGE
</div>
"""
            )

            image_input = gr.Image(
                type="filepath",
                label="Upload Runway Image",
                height=400,
                elem_classes="image-container"
            )

            analyze_button = gr.Button(
                "🔍 ANALYZE RUNWAY",
                elem_classes="analyze-btn"
            )

            gr.Markdown(
                """
**Supported formats:** JPG, JPEG, PNG

Upload an image containing a runway scene.
RunwayGuard AI will visually analyze the image
for possible Foreign Object Debris (FOD).
"""
            )


        # ====================================================
        # RESULTS PANEL
        # ====================================================

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.Markdown(
                """
<div class="section-title">
🛰 AI DETECTION RESULT
</div>
"""
            )

            detection_output = gr.Textbox(
                label="Detection Assessment",
                placeholder=(
                    "Upload an image and click "
                    "'ANALYZE RUNWAY'..."
                ),
                lines=10,
                elem_classes="output-box"
            )

            gr.Markdown(
                """
<div class="section-title">
⚠ AI INCIDENT REPORT
</div>
"""
            )

            report_output = gr.Textbox(
                label="Incident Report",
                placeholder=(
                    "The AI-generated incident "
                    "report will appear here..."
                ),
                lines=14,
                elem_classes="output-box"
            )


    # --------------------------------------------------------
    # SYSTEM CAPABILITIES
    # --------------------------------------------------------

    gr.Markdown(
        "## RunwayGuard AI Capabilities"
    )

    with gr.Row():

        with gr.Column(
            elem_classes="info-card"
        ):

            gr.Markdown(
                """
### 🔎 Detection

Analyzes uploaded runway imagery
for visible foreign objects.
"""
            )


        with gr.Column(
            elem_classes="info-card"
        ):

            gr.Markdown(
                """
### ⚠ Risk Assessment

Provides an AI-assisted
LOW / MEDIUM / HIGH risk assessment.
"""
            )


        with gr.Column(
            elem_classes="info-card"
        ):

            gr.Markdown(
                """
### 📋 Incident Reporting

Generates a structured
AI-assisted runway incident report.
"""
            )


    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    with gr.Accordion(
        "🔧 How RunwayGuard AI Works",
        open=False
    ):

        gr.Markdown(
            """
### AI Analysis Pipeline

**1. Upload**

The user uploads a runway image.

↓

**2. Vision AI**

Groq Vision analyzes the visual content.

↓

**3. FOD Assessment**

The system determines whether
obvious FOD may be present.

↓

**4. Risk Assessment**

The suspected hazard is assigned
an AI-assisted risk level.

↓

**5. Incident Report**

A structured report is generated
for review.

---

> **Note:** RunwayGuard AI is an educational
> and hackathon prototype. It does not replace
> certified airport inspection systems,
> human verification, or official aviation
> safety procedures.
"""
        )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.Markdown(
        """
<div class="footer">

RUNWAYGUARD AI • AI-ASSISTED AVIATION SAFETY PROTOTYPE

<br><br>

Computer Vision + Generative AI + Agentic AI

</div>
"""
    )


    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    analyze_button.click(
        fn=runwayguard_image,

        inputs=image_input,

        outputs=[
            detection_output,
            report_output
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )
    )
