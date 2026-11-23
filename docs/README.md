# Project page

The research project page is a static site in [`index.html`](index.html). It
requires no Node packages or build step. The workflow and profile-format guides
remain in this directory as Markdown documentation.

The hosted project page is coming soon. All asset paths are relative, so the
page supports GitHub Pages under the `/DVP/` project path.

## Local preview

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000/` on that machine. For a remote server, forward the port
to your local machine before opening the URL. Stop the server with Ctrl+C.

## Content and assets

- `index.html`: paper metadata, abstract, figures, resources, and citation.
- `assets/css/site.css`: responsive layout and reduced-motion support.
- `assets/js/site.js`: mobile navigation, figure enlargement, and citation copying.
- `assets/figures/`: Figures 1, 3, 4, and 5 extracted from the paper. Figure
  contents are unchanged; captions are provided as HTML for accessibility.
- `assets/animations/profile-construction.gif`: a 15-second illustrative
  construction walkthrough. Its sentences, annotations, scores, and distributions
  are schematic. Visitors can hide the animation; it starts collapsed when
  reduced motion is preferred or JavaScript is unavailable.

The page presents the paper's 21 study groups, not the union of source groups
in the aggregate release bundle. Model-training and steering results describe
the paper.

The visual direction is inspired by the [VALUEFLOW project page](https://aidaslab.github.io/VALUEFLOW/).
Source-data terms and project licensing are documented in [`DATA.md`](../DATA.md).

## Availability

The public paper, full Hugging Face aggregate bundle, and model-training and
steering code are coming soon. The current code release covers data processing,
annotation, and profile construction.
