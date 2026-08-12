#!/usr/bin/env python3
"""Render paper/main.tex to paper/main.md.

`main.md` is a reading rendering of the submission of record, `main.tex`. It
carries the same sections, the same sentences and the same numbers; the only
deliberate differences are that mathematics is written in Unicode rather than
LaTeX, that cross-references are resolved to fixed numbers, and that citations
are author-year rather than the superscript numerals BibTeX generates for the
submission. This script exists so the two cannot drift apart.

Usage:  python3 paper/tex2md.py
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
MD = HERE / "main.md"

# Author-year rendering of every key in refs.bib.
CITE = {
    "Fu2023": "Fu *et al.* 2023", "Stocker2022": "Stocker *et al.* 2022",
    "Morrow2023": "Morrow *et al.* 2023", "Kovacs2021": "Kovács *et al.* 2021",
    "Wu2024": "Wu *et al.* 2024", "Pota2024": "Póta *et al.* 2024",
    "Deng2025": "Deng *et al.* 2025", "Imbalzano2021": "Imbalzano *et al.* 2021",
    "Thaler2021": "Thaler and Zavadlav 2021",
    "Frederiksen2004": "Frederiksen *et al.* 2004",
    "Kellner2026": "Kellner *et al.* 2026",
    "Gawkowski2026": "Gawkowski *et al.* 2026",
    "Behler2007": "Behler and Parrinello 2007", "Behler2011": "Behler 2011",
    "Bartok2010": "Bartók *et al.* 2010", "Bartok2013": "Bartók *et al.* 2013",
    "Drautz2019": "Drautz 2019", "Thompson2015": "Thompson *et al.* 2015",
    "Batzner2022": "Batzner *et al.* 2022", "Batatia2022": "Batatia *et al.* 2022",
    "Schutt2018": "Schütt *et al.* 2018", "Gilmer2017": "Gilmer *et al.* 2017",
    "Jones1924": "Jones 1924", "Verlet1967": "Verlet 1967",
    "Rahman1964": "Rahman 1964", "StillingerWeber1985": "Stillinger and Weber 1985",
    "DawBaskes1984": "Daw and Baskes 1984",
    "FinnisSinclair1984": "Finnis and Sinclair 1984",
    "CleriRosato1993": "Cleri and Rosato 1993", "Zwanzig1954": "Zwanzig 1954",
    "Bennett1976": "Bennett 1976", "ShirtsChodera2008": "Shirts and Chodera 2008",
    "Duane1987": "Duane *et al.* 1987", "Neal2011": "Neal 2011",
    "Metropolis1953": "Metropolis *et al.* 1953",
    "Flyvbjerg1989": "Flyvbjerg and Petersen 1989", "Kunsch1989": "Künsch 1989",
    "PolitisRomano1994": "Politis and Romano 1994", "Kish1965": "Kish 1965",
    "Steinhardt1983": "Steinhardt *et al.* 1983",
    "LechnerDellago2008": "Lechner and Dellago 2008",
    "Swope1982": "Swope *et al.* 1982",
    "LebowitzPercusVerlet1967": "Lebowitz, Percus and Verlet 1967",
    "FrenkelSmit2002": "Frenkel and Smit 2002",
    "AllenTildesley2017": "Allen and Tildesley 2017",
    "HansenMcDonald2013": "Hansen and McDonald 2013",
    "Bouthillier2021": "Bouthillier *et al.* 2021",
    "Pineau2021": "Pineau *et al.* 2021", "Kapoor2024": "Kapoor *et al.* 2024",
    "Deposit": "deposit reference",
}

# Cross-reference labels resolved to the numbers REVTeX assigns.
REF = {
    "sec:intro": "Section 1", "sec:theory": "Section 2",
    "sec:cumulant": "Section 2.3", "sec:frequency": "Section 2.5",
    "sec:retraction": "Section 2.6", "sec:dynamics": "Section 2.7",
    "sec:methods": "Section 3", "sec:equilibration": "Section 3.3",
    "sec:designed": "Section 3.5", "sec:zoo": "Section 3.6",
    "sec:uncertainty": "Section 3.7", "sec:results": "Section 4",
    "sec:counterexamples": "Section 4.1", "sec:prediction": "Section 4.2",
    "sec:orthogonality": "Section 4.3", "sec:regimes": "Section 4.4",
    "sec:fitted": "Section 4.5", "sec:width": "Section 4.6",
    "sec:calibration": "Section 4.7", "sec:replication": "Section 4.8",
    "sec:failures": "Section 5", "sec:w32": "Section 5.1",
    "sec:smoothness": "Section 5.2", "sec:discrepancy": "Section 5.3",
    "sec:breakdown": "Section 5.4", "sec:warninglight": "Section 5.5",
    "sec:dilute": "Section 5.6",
    "sec:prior": "Section 6", "sec:scope": "Section 7",
    "sec:conclusion": "Section 8",
    "app:validation": "Appendix A", "app:statistics": "Appendix B",
    "fig:mechanism": "Figure 1", "fig:prediction": "Figure 2",
    "fig:counterexamples": "Figure 3", "fig:regimes": "Figure 4",
    "fig:proxies": "Figure 5", "fig:response": "Figure 6",
    "tab:clusters": "Table I", "tab:warning": "Table II",
    "tab:counterexamples": "Table III", "tab:budget": "Table IV",
    "tab:regimes": "Table V", "tab:zoo": "Table VI",
    "tab:breakdown": "Table VII", "tab:discrepancy": "Table VIII",
    "tab:cutoff": "Table IX", "tab:validation": "Table X",
    "eq:frmse": "Eq. (2)", "eq:fep": "Eq. (3)", "eq:cumulants": "Eq. (4)",
    "eq:linear": "Eq. (6)", "eq:cs": "Eq. (7)", "eq:modecov": "Eq. (8)",
    "eq:modeforce": "Eq. (9)", "eq:w32": "Eq. (11)",
}

MATH = [
    (r"\\dU", "δU"), (r"\\eVA", "eV/Å"), (r"\\AA\b", "Å"), (r"\\aa\b", "Å"),
    (r"\\avgz\{([^{}]*)\}", r"⟨\1⟩₀"), (r"\\avg\{([^{}]*)\}", r"⟨\1⟩"),
    (r"\\Cov_0", "Cov₀"), (r"\\Cov", "Cov"), (r"\\Var_0", "Var₀"),
    (r"\\Var", "Var"), (r"\\beta", "β"), (r"\\sigma", "σ"), (r"\\rho", "ρ"),
    (r"\\delta", "δ"), (r"\\Delta", "Δ"), (r"\\tau_\{\\rm int\}", "τ_int"),
    (r"\\mp\b", "∓"), (r"\\equiv", "≡"), (r"\\in\b", "∈"), (r"\\nabla", "∇"), (r"\\varphi", "φ"), (r"\\epsilon", "ε"), (r"\\varepsilon", "ε"),
    (r"\\kappa_0", "κ₀"), (r"\\mathbb\{E\}", "E"), (r"\\mathbb\{R\}", "ℝ"),
    (r"\\times", "×"), (r"\\approx", "≈"), (r"\\le(?![a-zA-Z])", "≤"), (r"\\ge(?![a-zA-Z])", "≥"),
    (r"\\pm", "±"), (r"\\sim\b", "~"), (r"\\cdot", "·"), (r"\\to\b", "→"),
    (r"\\ldots", "…"), (r"\\dots", "…"), (r"\\sum", "Σ"), (r"\\int", "∫"),
    (r"\\sqrt\{([^{}]*)\}", r"√(\1)"), (r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2"),
    (r"\\text\{([^{}]*)\}", r"\1"), (r"\\mathrm\{([^{}]*)\}", r"\1"),
    (r"\\emph\{([^{}]*)\}", r"\1"), (r"\\rm\b", ""), (r"\\,", " "),
    (r"\\;", " "), (r"\\!", ""), (r"\\big\|", "|"), (r"\\Big\|", "|"),
    (r"\\big\\langle", "⟨"), (r"\\big\\rangle", "⟩"),
    (r"\\langle", "⟨"), (r"\\rangle", "⟩"), (r"\\left", ""), (r"\\right", ""),
    (r"\\tilde A", "Ã"), (r"\\widetilde\{δU\}", "δŨ"),
    (r"\\underbrace\{([^{}]*)\}", r"\1"), (r"\\overline\{([^{}]*)\}", r"\1̄"),
    (r"\\bar ", ""), (r"\\quad", "  "), (r"\{,\}", ","), (r"\\mathbf\{([^{}]*)\}", r"\1"),
]

SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵",
       "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻", "+": "⁺"}
SUB = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅",
       "6": "₆", "7": "₇", "8": "₈", "9": "₉", "k": "ₖ", "m": "ₘ"}



def _arg(s: str, i: int):
    """Return (content, index_after) for a balanced {...} starting at s[i]=='{'."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _cmd1(s: str, name: str, fmt) -> str:
    """Replace \\name{arg} (balanced) using fmt(arg)."""
    out, i = [], 0
    pat = "\\" + name + "{"
    while True:
        k = s.find(pat, i)
        if k < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:k])
        arg, j = _arg(s, k + len(pat) - 1)
        out.append(fmt(arg))
        i = j


def _cmd2(s: str, name: str, fmt) -> str:
    """Replace \\name{a}{b} (balanced) using fmt(a, b)."""
    out, i = [], 0
    pat = "\\" + name + "{"
    while True:
        k = s.find(pat, i)
        if k < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:k])
        a, j = _arg(s, k + len(pat) - 1)
        if j < len(s) and s[j] == "{":
            b, j = _arg(s, j)
        else:
            b = ""
        out.append(fmt(a, b))
        i = j


def prepare_math(s: str) -> str:
    """Resolve the nested-brace constructs before the flat substitutions."""
    for _ in range(4):
        s = _cmd2(s, "frac", lambda a, b: f"({a.strip()})/({b.strip()})")
        s = _cmd1(s, "avgz", lambda a: f"\\langle {a} \\rangle_0")
        s = _cmd1(s, "avg", lambda a: f"\\langle {a} \\rangle")
        s = _cmd1(s, "underbrace", lambda a: a)
        s = _cmd1(s, "widetilde", lambda a: a + "\u0303")
        s = _cmd1(s, "big", lambda a: a)
    s = s.replace("_{\\rm test}", "_test").replace("{\\rm int}", "int")
    s = s.replace("\\rm ", "")
    return s


def render_math(s: str) -> str:
    s = s.replace("\\{", "\u2983").replace("\\}", "\u2984")
    s = prepare_math(s)
    for pat, rep in MATH:
        s = re.sub(pat, rep, s)
    # superscripts / subscripts of simple numeric groups
    s = re.sub(r"\^\{([-+0-9]+)\}",
               lambda m: "".join(SUP.get(c, c) for c in m.group(1)), s)
    s = re.sub(r"\^\{([^{}]+)\}", r"^{\1}", s)
    s = re.sub(r"_\{([^{}]+)\}", r"_{\1}", s)
    s = re.sub(r"\^([0-9])", lambda m: SUP[m.group(1)], s)
    s = re.sub(r"_\{([0-9km]+)\}",
               lambda m: "".join(SUB.get(c, c) for c in m.group(1)), s)
    s = re.sub(r"_([0-9km])\b", lambda m: SUB.get(m.group(1), m.group(1)), s)
    s = s.replace("^*", "*")
    s = s.replace("\\", "")
    s = re.sub(r"(?<![\^_])\{", "", s)
    s = re.sub(r"\}(?![^{]*\^)", "}", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"⟨\s+", "⟨", s)
    s = re.sub(r"\s+⟩", "⟩", s)
    return s.replace("δ U", "δU").replace("δ F", "δF").replace("∇ δU", "∇δU")


def inline(s: str) -> str:
    while "\\multicolumn{" in s:
        k = s.index("\\multicolumn{")
        _, j = _arg(s, k + len("\\multicolumn"))
        _, j = _arg(s, j)
        txt, j = _arg(s, j)
        s = s[:k] + txt + s[j:]
    s = re.sub(r"\\cite\{([^{}]*)\}",
               lambda m: " (" + "; ".join(CITE.get(k.strip(), k.strip())
                                          for k in m.group(1).split(",")) + ")", s)
    s = re.sub(r"\\onlinecite\{([^{}]*)\}",
               lambda m: CITE.get(m.group(1).strip(), m.group(1).strip()), s)
    s = re.sub(r"\\(?:eq)?ref\{([^{}]*)\}",
               lambda m: REF.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\$([^$]*)\$", lambda m: render_math(m.group(1)), s)
    for _ in range(3):
        s = _cmd1(s, "texttt", lambda a: "`" + a.replace("\\_", "_") + "`")
        s = _cmd1(s, "textbf", lambda a: "**" + a + "**")
        s = _cmd1(s, "emph", lambda a: "*" + a + "*")
        s = _cmd1(s, "mathbf", lambda a: "**" + a + "**")
    s = s.replace("\\AA{}", "Å").replace("\\AA", "Å").replace("~\\AA", " Å")
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    s = s.replace("\\dots", "…").replace("\\ngstr\\\"om", "ngström")
    s = s.replace("--", "–").replace("–-", "—").replace("---", "—")
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = re.sub(r"Secs?\.~?\s*Section (\S+?) and Section ", r"Sections \1 and ", s)
    s = re.sub(r"Tables~?\s*Table (\S+?) and Table ", r"Tables \1 and ", s)
    s = re.sub(r"Figs?\.~?\s*Figure (\S+?) and Figure ", r"Figures \1 and ", s)
    s = re.sub(r"Secs?\.~?\s*Section", "Section", s)
    s = re.sub(r"Sections?~?\s*Section", "Section", s)
    s = re.sub(r"Tables?~?\s*Table", "Table", s)
    s = re.sub(r"Figs?\.~?\s*Figure", "Figure", s)
    s = re.sub(r"Figures?~?\s*Figure", "Figure", s)
    s = re.sub(r"Appendix~?\s*Appendix", "Appendix", s)
    s = re.sub(r"Eqs?\.~?\s*Eq\.", "Eq.", s)
    s = re.sub(r"Equations?~?\s*Eq\. ", "Equation ", s)
    s = re.sub(r"~", " ", s)
    s = re.sub(r"\{,\}", ",", s)
    s = re.sub(r"\\texorpdfstring\{([^{}]*)\}\{[^{}]*\}", r"\1", s)
    s = re.sub(r"\\label\{[^{}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = re.sub(r"(?<![\^_])\{", "", s)
    s = s.replace("δ U", "δU")
    s = s.replace("\u2983", "{").replace("\u2984", "}")
    return re.sub(r"[ \t]+", " ", s).strip()


def wrap(s: str, width: int = 79) -> str:
    out, line = [], ""
    for word in s.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = word if not line else line + " " + word
    if line:
        out.append(line)
    return "\n".join(out)


def convert_body(body: str) -> list[str]:
    """Convert the running text of the manuscript (sections 1-9 + back matter)."""
    # Pull environments out first, so they survive paragraph splitting.
    envs: list[str] = []

    def stash(m):
        envs.append(flush_env(m.group(1), m.group(0)))
        return f"\n\n@@ENV{len(envs) - 1}@@\n\n"

    body = re.sub(r"\\begin\{acknowledgments\}(.*?)\\end\{acknowledgments\}",
                  lambda m: "\n\n## Acknowledgments\n\n" + m.group(1).strip() + "\n\n",
                  body, flags=re.S)
    body = re.sub(r"\\begin\{(equation|align|gather|quote|enumerate|itemize)\*?\}"
                  r".*?\\end\{\1\*?\}", stash, body, flags=re.S)

    out: list[str] = []
    appendix = [False]
    secno, subno = 0, 0
    body = re.sub(r"\\subsection(\*?)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
                  lambda m: "\\subsection" + m.group(1) + "{"
                            + " ".join(m.group(2).split()) + "}",
                  body, flags=re.S)
    for blk in re.split(r"\n\s*\n", body):
        stripped = blk.strip()
        if not stripped:
            continue
        m = re.fullmatch(r"@@ENV(\d+)@@", stripped)
        if m:
            out.append(envs[int(m.group(1))])
            continue
        if stripped.startswith("\\appendix"):
            secno, subno = 0, 0
            appendix[0] = True
            stripped = stripped[len("\\appendix"):].strip()
            if not stripped:
                continue
        m = re.match(r"\\section(\*?)\{([^{}]*)\}", stripped)
        if m:
            title = inline(m.group(2))
            if m.group(1) == "*":
                out.append(f"## {title.title()}")
            elif appendix[0]:
                secno += 1
                subno = 0
                out.append(f"## Appendix {chr(64 + secno)}: {title}")
            else:
                secno += 1
                subno = 0
                out.append(f"## {secno}. {title}")
            rest = stripped[m.end():].strip()
            if rest:
                out.append(wrap(inline(rest)))
            continue
        stripped = re.sub(r"\\texorpdfstring\{((?:[^{}]|\{[^{}]*\})*)\}\{[^{}]*\}",
                          r"\1", stripped)
        m = re.match(r"\\subsection(\*?)\{((?:[^{}]|\{[^{}]*\})*)\}", stripped)
        if m:
            subno += 1
            label = "" if m.group(1) == "*" else f"{secno}.{subno}. "
            out.append(f"### {label}{inline(m.group(2))}")
            rest = stripped[m.end():].strip()
            if rest:
                out.append(wrap(inline(rest)))
            continue
        m = re.match(r"\\paragraph\*?\{([^{}]*)\}", stripped)
        if m:
            rest = inline(stripped[m.end():].strip())
            out.append(wrap(f"**{inline(m.group(1))}** {rest}").strip())
            continue
        txt = inline(stripped)
        if txt:
            out.append(wrap(txt))
    return out


def flush_env(env: str, text: str) -> str:
    text = re.sub(r"\\(begin|end)\{[a-z*]+\}", "", text)
    text = re.sub(r"\\label\{[^{}]*\}", "", text)
    if env in ("equation", "align"):
        body = re.sub(r"\\\\", "\n", text)
        body = "\n".join(render_math(l).strip() for l in body.split("\n")
                         if l.strip())
        return "```\n" + body + "\n```"
    if env == "quote":
        return "\n".join("> " + l for l in wrap(inline(text)).split("\n"))
    items = [i.strip() for i in re.split(r"\\item", text) if i.strip()]
    marker = (lambda i: f"{i+1}.") if env == "enumerate" else (lambda i: "-")
    return "\n\n".join(wrap(f"{marker(i)} {inline(it)}") for i, it in enumerate(items))


def main() -> None:
    tex = TEX.read_text()
    tex = re.sub(r"(?<!\\)%.*", "", tex)

    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    body = tex.split(r"\maketitle", 1)[1]
    floats = {}
    for env in ("figure*", "figure", "table*", "table"):
        for m in re.finditer(r"\\begin\{%s\}.*?\\end\{%s\}" % (re.escape(env), re.escape(env)),
                             body, re.S):
            lab = re.search(r"\\label\{([^{}]*)\}", m.group(0))
            if lab:
                floats[lab.group(1)] = m.group(0)
        body = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (re.escape(env), re.escape(env)),
                      "", body, flags=re.S)
    body = body.split(r"\bibliographystyle", 1)[0]

    parts = [
        "# Error fields, not error norms: how the error in a fitted interatomic "
        "potential reaches a physical observable",
        "",
        "**[AUTHOR NAME]**, [AFFILIATION], [EMAIL]",
        "",
        "*Prepared for* The Journal of Chemical Physics *(Regular Article). This "
        "is the Markdown rendering of `paper/main.tex`, generated by "
        "`paper/tex2md.py`: same content, same sentences, same numbers. "
        "Mathematics is written in Unicode, cross-references are resolved to "
        "fixed numbers, and citations are rendered author-year; the submission "
        "of record uses superscript numerals generated by BibTeX from "
        "`paper/refs.bib`. Figures are the `.pdf` files in `../figures/`.*",
        "",
        "---",
        "",
        "## Abstract",
        "",
        "*(246 words.)*",
        "",
        wrap(inline(abstract)),
        "",
        "---",
        "",
    ]
    parts.extend(b + "\n" for b in convert_body(body))

    # Figures and tables, in numbering order.
    order = ["fig:mechanism", "fig:prediction", "fig:counterexamples",
             "fig:regimes", "fig:proxies", "fig:response"]
    parts.append("\n---\n\n## Figures\n")
    for lab in order:
        src = re.search(r"\\includegraphics\[[^\]]*\]\{([^{}]*)\}", floats[lab])
        cap = re.search(r"\\caption\{(.*)\}\s*\\end\{figure", floats[lab], re.S).group(1)
        cap = re.sub(r"\\label\{[^{}]*\}%?", "", cap).strip()
        parts.append(f"**{REF[lab]}** (`{src.group(1)}`)\n")
        parts.append(wrap(inline(cap)) + "\n")

    parts.append("\n---\n\n## Tables\n")
    for lab in ["tab:clusters", "tab:warning",
                "tab:counterexamples", "tab:budget", "tab:regimes", "tab:zoo",
                "tab:breakdown", "tab:discrepancy", "tab:cutoff", "tab:validation"]:
        blk = floats[lab]
        cap = re.search(r"\\caption\{(.*)\}\s*\\begin\{ruledtabular\}", blk, re.S).group(1)
        cap = re.sub(r"\\label\{[^{}]*\}%?", "", cap).strip()
        rows = re.search(r"\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}",
                         blk, re.S).group(1)
        parts.append(f"### {REF[lab]}\n")
        parts.append(wrap(inline(cap)) + "\n")
        parts.append("```")
        for line in rows.split(r"\\"):
            line = re.sub(r"\\colrule|\\footnotemark\[\d\]", "", line).strip()
            if not line:
                continue
            cells = [inline(c) for c in line.split("&")]
            parts.append(" | ".join(c for c in cells if c is not None))
        parts.append("```\n")
        foot = re.findall(r"\\footnotetext\[\d\]\{(.*?)\}\s*\n", blk, re.S)
        for f in foot:
            parts.append(wrap("*Footnote:* " + inline(f)) + "\n")

    parts.append("\n---\n\n## References\n")
    parts.append(wrap(
        "Citations above are rendered author-year for readability. The "
        "submission of record numbers them as superscripts in order of first "
        "citation, generated by BibTeX from `paper/refs.bib`; the list below is "
        "in the order the entries appear in that file, which is not the "
        "citation order. The works cited are:") + "\n")
    bib = (HERE / "refs.bib").read_text()
    keys = re.findall(r"@\w+\{([^,]+),", bib)
    parts.append(" · ".join(CITE.get(k, k) for k in keys) + ".\n")

    MD.write_text("\n".join(parts).replace("\n\n\n\n", "\n\n") + "\n")
    print(f"wrote {MD} ({len(MD.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()
