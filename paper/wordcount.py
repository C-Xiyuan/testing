#!/usr/bin/env python3
"""Count the main-text words of paper/main.tex.

The count excludes, per SPEC.md Sec. 12 item 5: the preamble, the abstract, all
float environments (figures and tables, hence all captions and table contents),
the appendices, the bibliography, and LaTeX comments. Section headings and
displayed equations are also excluded, since neither is prose.

Usage:  python3 paper/wordcount.py
"""

from __future__ import annotations

import re
from pathlib import Path

TEX = Path(__file__).resolve().parent / "main.tex"


def strip_to_words(s: str) -> list[str]:
    """Reduce LaTeX source to the tokens a word counter would see."""
    s = re.sub(r"\\(beta|sigma|rho|delta|Delta|times|langle|rangle)\b", " Z ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)
    s = re.sub(r"[{}$\\&]", " ", s)
    return [w for w in s.split() if any(c.isalnum() for c in w)]


def abstract_count(text: str) -> int:
    body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S).group(1)
    body = re.sub(r"(?<!\\)%.*", "", body)
    return len(strip_to_words(body))


def clean(text: str) -> str:
    """Strip everything the count excludes, leaving prose."""
    # Float environments.
    for env in ("figure*", "figure", "table*", "table"):
        text = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (re.escape(env), re.escape(env)),
                      " ", text, flags=re.S)

    # Displayed equations.
    for env in ("equation", "align", "gather", "eqnarray"):
        text = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env),
                      " ", text, flags=re.S)

    # Section headings.
    text = re.sub(r"\\(sub)*section\*?\{[^}]*\}", " ", text)
    text = re.sub(r"\\paragraph\*?\{[^}]*\}", " ", text)
    text = re.sub(r"\\label\{[^}]*\}", " ", text)
    # A citation renders as a superscript numeral and is not counted as a word;
    # a cross-reference renders as a number in running text and is counted.
    text = re.sub(r"\\cite\{[^}]*\}", " ", text, flags=re.S)
    text = re.sub(r"\\(onlinecite|ref|eqref)\{[^}]*\}", " X ", text, flags=re.S)

    # Remaining macros, braces and inline maths markers.
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)
    text = re.sub(r"[{}$\\&]", " ", text)
    return text


def per_section(body: str) -> list[tuple[str, int]]:
    """Word count of each top-level section, in document order."""
    marks = [(m.start(), m.group(1)) for m in
             re.finditer(r"\\section\*?\{([^}]*)\}", body)]
    out = []
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(body)
        words = [w for w in clean(body[pos:end]).split() if any(c.isalnum() for c in w)]
        out.append((name, len(words)))
    return out


def main():
    text = TEX.read_text()
    print(f"abstract words: {abstract_count(text)}")

    # Comments.
    text = re.sub(r"(?<!\\)%.*", "", text)

    # Everything up to and including \maketitle (preamble, title, abstract).
    text = text.split(r"\maketitle", 1)[1]

    # Everything from \appendix onward (appendices, floats, bibliography).
    text = text.split(r"\appendix", 1)[0]

    for name, n in per_section(text):
        print(f"  {name:<32s} {n:5d}")

    words = [w for w in clean(text).split() if any(c.isalnum() for c in w)]
    print(f"main-text words (Secs. 1-9, excluding abstract, floats, "
          f"appendices, references): {len(words)}")


if __name__ == "__main__":
    main()
