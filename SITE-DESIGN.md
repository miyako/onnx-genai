# Site design

The canonical [local inference documentation design guideline](https://github.com/miyako/local-inference-site-design/blob/main/SITE-DESIGN.md)
is maintained in `miyako/local-inference-site-design`, not in this repository.

**Adopted version:** [v1.0.0](https://github.com/miyako/local-inference-site-design/blob/v1.0.0/SITE-DESIGN.md).

This is a documentation reference, not a submodule or build dependency.
Guideline updates are adopted explicitly alongside any corresponding local
implementation changes; the site does not fetch shared assets at build time.

## Local implementation

- Styles: [`assets/css/site.css`](assets/css/site.css) and
  [`assets/css/code.css`](assets/css/code.css).
- Shell: [`_layouts/default.html`](_layouts/default.html).
- Landing page: [`index.html`](index.html).

## Local exceptions

None beyond the guideline's permitted engine-specific branding and content.
ONNX retains its green accent, `.onnx` mark, model-directory explanation, and
capability-specific sections.
