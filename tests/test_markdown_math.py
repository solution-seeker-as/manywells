"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 5 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The math in the project's markdown renders the same on GitHub as in VS Code (AGENTS.md, Math in markdown). VS Code's
KaTeX sees the math as written. GitHub parses the markdown before MathJax sees the math, so a backslash escape, an
emphasis delimiter or an HTML tag in the math changes it first: `\\,` reaches MathJax as a comma, and in inline math
the `_` of `(x)_i` opens emphasis that the `_` of the next `c_{p}` closes. In display math GitHub passes `\\\\` through
and puts the `_` of any emphasis back, which repairs a `_` but turns a `*` into a `_`.
"""

import pathlib
import re
import string

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_DIRS = {'target', 'node_modules'}

# Code is not math: fenced blocks, HTML comments and code spans are matched first and left alone.
SEGMENT = re.compile(
    r'(?P<fence>^(?P<f>```|~~~).*?^(?P=f)[^\n]*$)'
    r'|(?P<comment><!--.*?-->)'
    r'|(?P<code>(?P<ticks>`+).+?(?P=ticks))'
    r'|(?P<display>\$\$.+?\$\$)'
    r'|(?P<inline>(?<![\\$])\$(?!\$)(?:[^$\n]|\n(?!\s*\n))+?(?<!\\)\$)',
    re.M | re.S)
TOKEN = re.compile(r'\\([A-Za-z]+|.)', re.S)
TEXT = re.compile(r'\\text\{[^{}]*\}')
# Markdown hazards in math, besides escapes: an asterisk; a `<` that can open an HTML tag; and, in inline math, an
# underscore that can open emphasis (after a character that is not a letter or digit, and before one that is not a
# space).
HAZARD = {True: re.compile(r'(?<!\\)\*|<(?=[A-Za-z/!?])'),
          False: re.compile(r'(?<!\\)\*|<(?=[A-Za-z/!?])|(?<![^\W_])(?<!\\)_(?=\S)')}


def markdown_files():
    for path in sorted(ROOT.rglob('*.md')):
        parts = path.relative_to(ROOT).parts[:-1]
        if not any(p.startswith('.') or p in SKIP_DIRS for p in parts):
            yield path


def math_spans(text):
    """(offset, body, display) of every math span in markdown `text`, with `display` true for `$$...$$`."""
    for m in SEGMENT.finditer(text):
        for kind, d in (('display', 2), ('inline', 1)):
            if m.group(kind) is not None:
                yield m.start() + d, m.group(kind)[d:-d], kind == 'display'


def problems(text):
    """(offset, what) of everything in the math of `text` that GitHub renders differently from VS Code."""
    for offset, raw, display in math_spans(text):
        # `\_` in `\text{...}` is safe: GitHub's MathJax prints the `_` that the markdown escape leaves.
        body = TEXT.sub(lambda m: m.group(0).replace('\\_', '_'), raw)
        for m in TOKEN.finditer(body):
            name = m.group(1)
            if (name in string.punctuation and not (display and name == '\\')) or name == 'operatorname':
                yield offset + m.start(), m.group(0)
        for m in HAZARD[display].finditer(body):
            yield offset + m.start(), body[max(0, m.start() - 1):m.end() + 1]
        line = text[text.rfind('\n', 0, offset) + 1:text.find('\n', offset)]
        if line.lstrip().startswith('|') and '|' in body:
            yield offset, '| in a table cell'
        end = offset + len(raw) + (2 if display else 1)
        if text[end:end + 1].isalnum():
            yield end, 'a letter or digit after the closing $'


def test_problems_finds_hazards_in_math_only():
    text = ('$a\\,b$ and $$\\operatorname{smax}(x)$$ but `$c\\,d$` and $\\text{cp\\_flux}$\n'
            '```\n$e\\,f$\n```\n'
            '$$\n\\begin{aligned}a &= (x)_i\\\\\nb &= c_{p}\\end{aligned}\n$$\n\n'
            '$p^*$, $$q^*$$, $(x)_i + c_{p}$, $a\\\\b$ and $a<b$ but $p^\\ast$, $(x)_ i$ and $a < b$; $k\\cdot$step\n\n'
            '| $\\lvert x\\rvert$ | $|y|$ |\n')
    assert [what for _, what in problems(text)] == [
        '\\,', '\\operatorname', '^*', '^*', ')_i', '\\\\', 'a<b', 'a letter or digit after the closing $',
        '| in a table cell']


def test_markdown_math_renders_on_github():
    found = []
    for path in markdown_files():
        text = path.read_text(encoding='utf-8')
        for offset, what in problems(text):
            line = text.count('\n', 0, offset) + 1
            found.append(f'{path.relative_to(ROOT)}:{line}: {what}')
    assert not found, 'math that GitHub renders differently from VS Code (AGENTS.md, Math in markdown):\n' + \
        '\n'.join(found)
