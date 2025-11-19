# StoryHash Paper - LaTeX Sources

This directory contains the LaTeX sources for the StoryHash arXiv paper.

## 📁 File Structure

```
paper/
├── main.tex              # Main LaTeX document (camera-ready)
├── references.bib        # BibTeX bibliography (5 citations)
├── figures/              # Publication-quality figures
│   ├── fig1_pipeline_placeholder.pdf
│   ├── fig2_ablation_summary.pdf
│   ├── fig3_similarity_delta.pdf
│   ├── fig4_flip_bias.pdf
│   ├── fig5_architecture.pdf
│   └── table1_per_transform.tex
└── README.md             # This file
```

## 🔨 Compilation Instructions

### Standard pdflatex + bibtex workflow:

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Or use latexmk (automated):

```bash
latexmk -pdf main.tex
```

This produces `main.pdf` (~8-12 pages).

## 📊 Figures

All figures are generated from **real experimental data** using `scripts/generate_paper_figures.py`:

- **fig1**: Pipeline architecture (placeholder - needs manual diagram)
- **fig2**: 3-way ablation study (Recall@1 bar chart)
- **fig3**: Similarity delta comparison (confidence boost)
- **fig4**: CLIP flip bias (flip_h vs flip_v)
- **fig5**: 527D vector composition (CLIP 97.2%, Action 2.8%)
- **table1**: Per-transformation results (14 transforms × 6 metrics)

## 📦 arXiv Submission Package

To create the arXiv-ready ZIP:

```bash
cd paper/
zip -r arxiv_submission.zip \
  main.tex \
  references.bib \
  main.bbl \
  figures/*.pdf \
  figures/*.tex
```

Upload `arxiv_submission.zip` to https://arxiv.org/submit

## ✅ Camera-Ready Checklist

Before submission, verify:

- [ ] All figures cited in text (\\ref{fig:...})
- [ ] All tables numbered correctly
- [ ] Bibliography compiled (no missing citations)
- [ ] No unverifiable claims (all numbers from experiments)
- [ ] No sensitive content (dataset is public DAVIS 2017)
- [ ] PDF compiles without errors
- [ ] File size < 50MB (arXiv limit)

## 📄 Document Metadata

- **Title**: Lightweight Multi-Modal Fusion for Robust Video Retrieval
- **Author**: Denis Billi (Independent Researcher)
- **Date**: November 2025
- **Categories**: cs.CV (Computer Vision), cs.MM (Multimedia)
- **Keywords**: video retrieval, multi-modal fusion, robustness, CLIP

## 🔗 Related Files

- **Source Code**: `/Users/denisbilli/Documents/Repos/StoryHash/`
- **Technical Report**: `docs/TECH_REPORT.md` (source material)
- **Experimental Data**: `docs/PHASE_6_NOTES.md` (raw results)
- **Roadmap**: `docs/ROADMAP_PAPER.md` (publication plan)

## 📝 Notes

- LaTeX class: `article` (11pt)
- Bibliography style: `plainnat` (natbib)
- Page limit: 12 pages (arXiv default)
- Figures: Vector PDF (preferred) with PNG fallbacks
- No author affiliations (independent research)

## 🚀 Next Steps (Phase B2-B3)

1. **Verify compilation** (this README)
2. **Check all references** (no broken links)
3. **Generate final PDF** (pdflatex sequence)
4. **Package for arXiv** (ZIP with sources)
5. **Mark Phase B complete** in ROADMAP_PAPER.md

---

**Version**: v0.5.0 (Paper Release)  
**Branch**: release-paper-v0.5.0  
**Last Updated**: November 17, 2025
