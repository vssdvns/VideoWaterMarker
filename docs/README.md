# Report

The Master's project report is in `REPORT.md`.

## Converting to PDF

### Option 1: Pandoc (recommended)

```bash
pandoc docs/REPORT.md -o docs/REPORT.pdf --pdf-engine=xelatex -V geometry:margin=1in
```

### Option 2: Markdown to PDF tools

- [md-to-pdf](https://www.npmjs.com/package/md-to-pdf): `npx md-to-pdf docs/REPORT.md`
- VS Code extension: "Markdown PDF"
- Online: paste REPORT.md into a Markdown-to-PDF converter

### Option 3: Print from browser

1. Open `REPORT.md` in a Markdown viewer (VS Code, GitHub, etc.)
2. Use "Print" and "Save as PDF"
