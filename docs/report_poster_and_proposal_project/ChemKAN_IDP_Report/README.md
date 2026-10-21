# ChemKAN IDP report — source package (21 October 2026)

This package contains the LaTeX source, bibliography, included figures, and the supplied 30-page `Bankoglu_Project_Report.pdf`.

## Files

- `main.tex` — LaTeX source
- `references.bib` — BibLaTeX bibliography
- `Bankoglu_Project_Report.pdf` — supplied compiled report
- `REPORT_CLAIM_EVIDENCE_MAP.md` — claim-by-claim literature and project-artifact provenance, with verification limits
- `ChemKAN_report_assets/` — figures used by the report
- `MANIFEST.sha256` — SHA-256 checksums for the non-hidden package files, excluding the manifest itself

## Compilation

Upload the directory contents to Overleaf, select `main.tex`, use pdfLaTeX and Biber, or compile locally with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Evidence and status

See `REPORT_CLAIM_EVIDENCE_MAP.md` for the exact basis of numerical statements and for claims supported only by historical development records. An identical repository copy is kept at `docs/REPORT_CLAIM_EVIDENCE_MAP.md`. The provided PDF should be regenerated after substantive edits to `main.tex` or `references.bib`.

Verify the package from this directory with `shasum -a 256 -c MANIFEST.sha256`.
Regenerate the manifest after any package edits, file renames or PDF compilation.
Checksums verify file integrity; they do not prove that the PDF was built from the current source.
