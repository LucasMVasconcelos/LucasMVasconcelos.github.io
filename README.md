# lucasmvasconcelos.github.io

Personal site: **[lucasmvasconcelos.github.io](https://lucasmvasconcelos.github.io/)**

Source for my personal site — software projects, notes on machine learning theory, and my master's research.

## Structure

- **About** (`/about/`) — who I am, how to reach me.
- **Projects** (`/projects/`) — software projects I've built, e.g. the [Financial AI Agent](https://github.com/LucasMVasconcelos/financial-agent) and the [ML Engineer Case](https://github.com/LucasMVasconcelos/ml-engineer-case).
- **Studies** (`/studies/`) — notes on machine learning theory and how to implement it (with LaTeX support for math).
- **Master's Research** (`/master/`) — NLP and machine learning applied to sentiment analysis in psychotherapy session transcripts.

## Built with

[Jekyll](https://jekyllrb.com/) + the [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) theme, hosted on GitHub Pages. Projects and studies are Jekyll collections (`_projects/`, `_studies/`), so adding a new entry to either section is just a new markdown file — no template or navigation changes needed.

## Running locally

```bash
bundle install
bundle exec jekyll serve
```

Then open `http://localhost:4000`.
