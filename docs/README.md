# Project page

The project page is [`index.html`](index.html). It needs no build step or Node
packages. This directory also contains the workflow and profile-format guides.

The hosted page is coming soon. Relative asset paths let it run under the
GitHub Pages `/DVP/` path.

## Local preview

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000/`. If the server is remote, forward port 8000 to your
computer first. Stop the preview with Ctrl+C.

## Content and assets

- `index.html`: paper metadata, abstract, figures, resources, and citation.
- `assets/css/site.css`: responsive layout and reduced-motion support.
- `assets/js/site.js`: mobile navigation, figure enlargement, and citation copying.
- `assets/figures/`: Figures 1, 3, 4, and 5 extracted from the paper. Figure
  contents are unchanged; captions are written in HTML.
- `assets/animations/profile-construction.gif`: a 15-second illustrative
  walkthrough with example sentences, annotations, scores, and profiles.
  It can be hidden and starts collapsed for reduced motion or without JavaScript.

The page describes the paper's 21 study groups. The aggregate bundle covers a
different set of source groups. Training and steering results are from the paper.

The design is inspired by the [VALUEFLOW project page](https://aidaslab.github.io/VALUEFLOW/).
See [`DATA.md`](../DATA.md) for licenses and source terms.

## Availability

The public paper, full Hugging Face aggregate bundle, and model-training and
steering code are coming soon. The current code release covers data processing,
annotation, and profile construction.
