"""Streamlit UI: upload, diagnosis card, 7-day path, student gap profile, class heatmap."""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from diagnose import Pipeline  # noqa: E402
from llm import Config  # noqa: E402
from store import class_heatmap, connect, gap_profile, init_db, repeat_error_rate  # noqa: E402
from vision import ParseResult, parse_photo  # noqa: E402

st.set_page_config(page_title="Pathfinder AI", page_icon="🌱", layout="wide")


@st.cache_resource
def get_pipeline():
    cfg = Config.from_env()
    pipe = Pipeline(cfg)
    return cfg, pipe


def samples_dir() -> str:
    return os.path.join(ROOT, "samples")


def list_samples():
    out = {}
    for name in sorted(os.listdir(samples_dir())):
        if name.endswith(".json"):
            with open(os.path.join(samples_dir(), name), "r", encoding="utf-8") as fh:
                out[name] = json.load(fh)
    return out


cfg, pipe = get_pipeline()
if pipe.provider.name == "offline-rule-baseline":
    st.warning(
        "Offline rule baseline active — no API key configured. Results are deterministic rule matches, "
        "NOT model diagnoses. Set LLM_API_KEY / LLM_BASE_URL in .env for the real pipeline.",
        icon="⚠️",
    )

st.title("Pathfinder AI 🌱")
st.caption("Every wrong answer is a map. Upload one and get a personalised 7-day review path.")

left, right = st.columns([1, 1])

with left:
    st.subheader("1 · The mistake")
    mode = st.radio("Input", ["Upload a photo", "Use a sample"], horizontal=True)
    image_path = None
    parsed = None

    if mode == "Upload a photo":
        up = st.file_uploader("Photo of the wrong answer", type=["jpg", "jpeg", "png"])
        if up is not None:
            tmp = os.path.join(ROOT, "data", "uploads")
            os.makedirs(tmp, exist_ok=True)
            image_path = os.path.join(tmp, up.name)
            with open(image_path, "wb") as fh:
                fh.write(up.getbuffer())
            st.image(image_path, use_column_width=True)
            parsed = parse_photo(pipe.provider, image_path)
            st.write(f"Parse status: **{parsed.status}**")
            if parsed.notes:
                st.info(parsed.notes)
    else:
        sample_map = list_samples()
        choice = st.selectbox("Sample", list(sample_map.keys()))
        sample = sample_map[choice]
        st.code(sample.get("question", "(unreadable)"), language="text")
        for i, s in enumerate(sample.get("steps", []), 1):
            st.text(f"step {i}: {s}")
        parsed = ParseResult(
            status=sample.get("parse_status", "ok" if sample.get("question") else "unreadable"),
            question=sample.get("question", ""),
            steps=sample.get("steps", []),
        )

    student_id = st.text_input("Student ID", value="student-01")

with right:
    st.subheader("2 · Diagnosis")
    if st.button("Run diagnosis", type="primary") or "result" in st.session_state:
        if parsed is None:
            st.info("Provide a photo or pick a sample first.")
        else:
            result = pipe.run(parsed, student_id=student_id, image_path=image_path, save=True)
            st.session_state["result"] = result

            if result["status"] != "ok":
                st.error(result.get("message") or "The work could not be read — please retake the photo.")
            else:
                st.metric("Error type", result["error_type"])
                st.metric("Knowledge point", result["knowledge_point"] or "unmapped")
                st.write(f"**Broken step:** `{result['broken_step']}`")
                st.write(f"**Confidence:** {result['confidence']}")
                if result["flags"]["needs_teacher_review"]:
                    st.warning("Low confidence — flagged for a teacher's check.", icon="👀")
                if result["flags"]["unmapped"]:
                    st.error(result.get("message") or "Knowledge point unmapped — not in the syllabus yet.")
                if result.get("explanation"):
                    st.write(result["explanation"])
                if result.get("remediation"):
                    st.info(result["remediation"])
                st.json(result)

st.divider()
st.subheader("3 · The 7-day path")
if "result" in st.session_state and st.session_state["result"].get("schedule"):
    result = st.session_state["result"]
    for entry in result["schedule"]:
        st.write(f"**Day {entry['day']}** — {entry['text']}  `{entry['difficulty']}`")
    if result.get("plan"):
        with st.expander("Full plan across open gaps (ranked by risk)"):
            for day_block in result["plan"]:
                st.write(f"**Day {day_block['day']} · {day_block['date']}**")
                for item in day_block["items"]:
                    st.write(f"- {item['knowledge_point']} · {item['text']} (risk {item['risk']})")
else:
    st.caption("Run a diagnosis to generate the path.")

st.divider()
st.subheader("4 · Gap profile & class heatmap")
conn = connect(os.path.join(ROOT, cfg.db_path))
init_db(conn)
col_a, col_b = st.columns([1, 1])
with col_a:
    profile = gap_profile(conn, student_id)
    if profile:
        st.dataframe(profile, use_container_width=True)
        st.write(repeat_error_rate(conn, student_id))
    else:
        st.caption("No mistakes recorded for this student yet.")
with col_b:
    heat = class_heatmap(conn)
    if heat:
        st.bar_chart({row["knowledge_point"]: row["mistakes"] for row in heat})
    else:
        st.caption("No class data yet.")
conn.close()
