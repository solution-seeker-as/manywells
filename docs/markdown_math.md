# Math in markdown

The specs, docs and plans are read on GitHub and in VS Code. VS Code's KaTeX sees the math as written; GitHub parses the markdown first, so a backslash escape, an emphasis delimiter or an HTML tag in the math is changed before MathJax sees it. `tests/test_markdown_math.py` checks these rules.

- **Delimiters.** `$...$` inline, with no letter or digit right after the closing `$` (VS Code does not close the math there), and `$$...$$` for display, starting on a new line. Not ```` ```math ```` blocks or `` $`...`$ ``: GitHub renders them, but VS Code shows them as code.
- **No backslash before punctuation.** Markdown drops the backslash, so `\,` reaches MathJax as a comma. Write `\thinspace`, `\negthinspace`, `\ `, `\lbrace` and `\rbrace` for `\,`, `\!`, `\;`, `\{` and `\}`. Two exceptions render in both: `\\` in display math, which GitHub keeps, and `\_` in `\text{...}`.
- **No emphasis.** Write `\ast` for `*`, as in `p^\ast`: two of them make emphasis, which display math turns into `_`. In inline math, a `_` after a bracket or a brace can open emphasis that the next subscript closes, so put a space after it: `(\alpha\rho_g v_g)_ i`, `\Delta\text{MD}_ i`. TeX ignores the space. Display math needs no space, because GitHub puts the `_` back.
- **No HTML.** Put a space after `<`, as in `a < b`, or markdown may read a tag. In a table cell, a `|` ends the cell: write `\lvert x\rvert`.
- **Operators.** GitHub refuses `\operatorname` ("macros are not allowed"): write `\mathop{\mathrm{smax}}(x, y)`.
- **Thin spaces.** TeX spaces a product itself. Add `\thinspace` only where the symbols read differently without it: a symbol before a bracket (`4h\thinspace(T - T_a)`, so that it does not read as a function), next to `\text{...}`, before a differential (`c_{pl}\thinspace dT`) and between digit groups (`101\thinspace 325`).
