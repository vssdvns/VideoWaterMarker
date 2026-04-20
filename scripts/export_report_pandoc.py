from __future__ import annotations

from pathlib import Path

from export_report_docx import (
    COMMITTEE_CHAIR,
    DEPARTMENT_NAME,
    DEGREE_NAME,
    GRAD_COORDINATOR,
    MAJOR_NAME,
    OUT_PATH,
    REPORT_TITLE,
    ROOT,
    SECOND_READER,
    STUDENT_NAME,
    TEX_PATH,
    TERM_NAME,
    extract_body,
    extract_references,
)


MD_PATH = ROOT / "MASTER_FINAL_REPORT_pandoc.md"
DOCX_PATH = ROOT / "MASTER_FINAL_REPORT_pandoc.docx"
PANDOC_EXE = Path(r"C:\Users\vssdv\AppData\Local\Pandoc\pandoc.exe")


def add(lines: list[str], text: str = "") -> None:
    lines.append(text)


def build_markdown() -> str:
    tex = TEX_PATH.read_text(encoding="utf-8")
    lines: list[str] = []

    add(lines, REPORT_TITLE)
    add(lines, "=" * len(REPORT_TITLE))
    add(lines)
    add(lines, "A Project")
    add(lines)
    add(lines, f"Presented to the faculty of the Department of {DEPARTMENT_NAME}")
    add(lines, "California State University, Sacramento")
    add(lines)
    add(lines, "Submitted in partial satisfaction of")
    add(lines, "the requirements for the degree of")
    add(lines)
    add(lines, DEGREE_NAME)
    add(lines)
    add(lines, "in")
    add(lines)
    add(lines, MAJOR_NAME)
    add(lines)
    add(lines, "by")
    add(lines)
    add(lines, STUDENT_NAME)
    add(lines)
    add(lines, TERM_NAME)
    add(lines)
    add(lines, r"\newpage")
    add(lines)

    add(lines, "Project Approval Page")
    add(lines, "---------------------")
    add(lines)
    add(lines, REPORT_TITLE)
    add(lines)
    add(lines, "A Project")
    add(lines)
    add(lines, f"by {STUDENT_NAME}")
    add(lines)
    add(lines, "Approved by:")
    add(lines)
    add(lines, f"- {COMMITTEE_CHAIR}, Committee Chair")
    add(lines, "- Date")
    add(lines, f"- {SECOND_READER}, Second Reader")
    add(lines, "- Date")
    add(lines)
    add(lines, r"\newpage")
    add(lines)

    add(lines, "Project Format Approval Page")
    add(lines, "----------------------------")
    add(lines)
    add(lines, f"Student: {STUDENT_NAME}")
    add(lines)
    add(
        lines,
        "I certify that this student has met the requirements for format contained in the University format manual, "
        "and this project is suitable for electronic submission to the library and credit is to be awarded for the project.",
    )
    add(lines)
    add(lines, f"{GRAD_COORDINATOR}, Graduate Coordinator")
    add(lines, "Date")
    add(lines, f"Department of {DEPARTMENT_NAME}")
    add(lines)
    add(lines, r"\newpage")
    add(lines)

    add(lines, "Abstract")
    add(lines, "--------")
    add(lines)
    add(lines, f"of {REPORT_TITLE}")
    add(lines)
    add(lines, f"by {STUDENT_NAME}")
    add(lines)
    add(lines, "**Statement of Problem**")
    add(lines)
    add(
        lines,
        "This project studies how to watermark video in a way that is practical for over-the-top streaming platforms. "
        "The main goal is to place a visible watermark where it is less distracting while still keeping it recoverable "
        "after common distortions such as re-encoding, blur, crop, grayscale conversion, frame-rate reduction, and resize. "
        "A second goal is traceability, so that a leaked copy can still be tied back to a user or session. To address both "
        "needs, the project combines visible watermarking, invisible payload embedding, detection, attack testing, and a user-facing application in one workflow.",
    )
    add(lines)
    add(lines, "**Sources of Data**")
    add(lines)
    add(
        lines,
        "The implemented system uses Laplacian-based complexity, semantic saliency from pretrained models, temporal smoothing, "
        "optional optical flow, visible text rendering, discrete cosine transform payload embedding, and a neural watermarking branch. "
        "It also stores frame-level placement metadata in a positions.json file so that detection can use both local search and a global fallback "
        "when the watermark shifts after attack. The broader project workflow draws on a standardized 150-clip training corpus for model development "
        "and a separate 26-clip HD benchmark for visible robustness evaluation.",
    )
    add(lines)
    add(lines, "**Conclusions Reached**")
    add(lines)
    add(
        lines,
        "The main conclusion is that content-aware visible watermarking is the strongest and most mature part of the system. "
        "The DCT and neural branches make the design more complete and more useful for forensic tracing, but they still involve stronger "
        "quality-versus-robustness tradeoffs.",
    )
    add(lines)
    add(lines, f"{COMMITTEE_CHAIR}, Committee Chair")
    add(lines, "Date")
    add(lines)
    add(lines, r"\newpage")
    add(lines)

    add(lines, "ACKNOWLEDGEMENTS")
    add(lines, "----------------")
    add(lines)
    add(
        lines,
        "I would like to express my sincere gratitude to my project advisor for the guidance, patience, and insights that shaped every stage of this work. "
        "The regular feedback sessions were essential in refining both the software and this written report.",
    )
    add(lines)
    add(
        lines,
        "I am grateful to the faculty of the Department of Computer Science at California State University, Sacramento, whose coursework in software engineering, "
        "data science, and related areas provided an important foundation for this project. I also thank the second reader and committee members for suggestions "
        "that improved the clarity of the manuscript.",
    )
    add(lines)
    add(
        lines,
        "Special thanks go to the open-source communities behind OpenCV, PyTorch, Streamlit, FastAPI, and FFmpeg, whose freely available tools made this project possible. "
        "Finally, I thank my family and friends for their encouragement and support throughout my graduate studies.",
    )
    add(lines)
    add(lines, r"\newpage")
    add(lines)

    for kind, text in extract_body(tex):
        if kind == "chapter":
            add(lines, f"# {text}")
            add(lines)
        elif kind == "section":
            add(lines, f"## {text}")
            add(lines)
        elif kind == "subsection":
            add(lines, f"### {text}")
            add(lines)
        elif kind == "paragraph":
            add(lines, text)
            add(lines)
        elif kind == "figure":
            add(lines, f"**Figure:** {text}")
            add(lines)
        elif kind == "table":
            add(lines, f"**Table:** {text}")
            add(lines)
        elif kind == "alt":
            add(lines, f"*Alternative text:* {text}")
            add(lines)

    add(lines, "# REFERENCES")
    add(lines)
    for i, ref in enumerate(extract_references(tex), start=1):
        add(lines, f"{i}. {ref}")
        add(lines)

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    MD_PATH.write_text(build_markdown(), encoding="utf-8")
    print(f"Wrote {MD_PATH}")
    if not PANDOC_EXE.exists():
        raise FileNotFoundError(f"Pandoc not found: {PANDOC_EXE}")
    import subprocess

    subprocess.run(
        [str(PANDOC_EXE), str(MD_PATH), "-o", str(DOCX_PATH)],
        check=True,
        cwd=str(ROOT),
    )
    print(f"Wrote {DOCX_PATH}")


if __name__ == "__main__":
    main()
