# Speculators Branding Guide

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/speculators-logo-white.svg" />
    <source media="(prefers-color-scheme: light)" srcset="assets/branding/speculators-logo-black.svg" />
    <img alt="Speculators logo" src="assets/branding/speculators-logo-black.svg" style="height: 64px; max-width: 100%; display: inline-block;" />
  </picture>
</div>

Fast, practical guidance and assets for using the Speculators brand in docs, slides, blogs, papers, and social posts.

- Audience: engineers, researchers, technical writers, and maintainers
- Formats: SVG (preferred for quality and scaling) and PNG (fallback for slides, web)
- Location: all assets live in `docs/assets/branding`

## Quick Start

1) Pick the correct asset type:
   - Logos (wordmark + symbol) for full brand representation
   - Icons (square) for compact usage or avatars
   - Model Icons for architecture/diagram callouts
   - User Flow diagrams for product explaining visuals
2) Choose the right color for contrast:
   - Black on light backgrounds; White on dark backgrounds
   - Blue/White-Orange are accents
3) SVG is preferred for quality and scaling; use PNG for raster contexts (e.g., slides)
4) Maintain clear space and minimum sizes (see below)
5) Add descriptive alt text for accessibility
6) Cite the project where appropriate

## Project Summary

Speculators is an open, unified library for creating, evaluating, and storing speculative decoding algorithms for efficient LLM inference. It integrates with Hugging Face formats and pairs with vLLM for production serving.

## Asset Catalog

All files reside in `docs/assets/branding`.

| Category | Variant / Name | SVG | PNG | Notes |
| -------- | -------------- | --- | --- | ----- |
| Logo | Black | [`speculators-logo-black.svg`](assets/branding/speculators-logo-black.svg) | [`speculators-logo-black.png`](assets/branding/speculators-logo-black.png) | Default on light backgrounds |
| Logo | Blue | [`speculators-logo-blue.svg`](assets/branding/speculators-logo-blue.svg) | [`speculators-logo-blue.png`](assets/branding/speculators-logo-blue.png) | Accent / section headers |
| Logo | White | [`speculators-logo-white.svg`](assets/branding/speculators-logo-white.svg) | [`speculators-logo-white.png`](assets/branding/speculators-logo-white.png) | Use on dark imagery/backgrounds |
| Icon | Black | [`speculators-icon-black.svg`](assets/branding/speculators-icon-black.svg) | [`speculators-icon-black.png`](assets/branding/speculators-icon-black.png) | App tile / light UI |
| Icon | Blue | [`speculators-icon-blue.svg`](assets/branding/speculators-icon-blue.svg) | [`speculators-icon-blue.png`](assets/branding/speculators-icon-blue.png) | Accent |
| Icon | White | [`speculators-icon-white.svg`](assets/branding/speculators-icon-white.svg) | [`speculators-icon-white.png`](assets/branding/speculators-icon-white.png) | Dark backgrounds |
| Icon | White-Orange | [`speculators-icon-white-orange.svg`](assets/branding/speculators-icon-white-orange.svg) | [`speculators-icon-white-orange.png`](assets/branding/speculators-icon-white-orange.png) | Limited highlight |
| Model Icon | Black | [`speculators-model-icon-black.svg`](assets/branding/speculators-model-icon-black.svg) | [`speculators-model-icon-black.png`](assets/branding/speculators-model-icon-black.png) | Architecture visuals |
| Model Icon | Blue | [`speculators-model-icon-blue.svg`](assets/branding/speculators-model-icon-blue.svg) | [`speculators-model-icon-blue.png`](assets/branding/speculators-model-icon-blue.png) | Architecture visuals |
| Model Icon | White | [`speculators-model-icon-white.svg`](assets/branding/speculators-model-icon-white.svg) | [`speculators-model-icon-white.png`](assets/branding/speculators-model-icon-white.png) | Dark backgrounds |
| Model Icon | White-Orange | [`speculators-model-icon-white-orange.svg`](assets/branding/speculators-model-icon-white-orange.svg) | [`speculators-model-icon-white-orange.png`](assets/branding/speculators-model-icon-white-orange.png) | Highlight sparingly |
| Diagram | User Flow (Light) | [`speculators-user-flow-light.svg`](assets/branding/speculators-user-flow-light.svg) | [`speculators-user-flow-light.png`](assets/branding/speculators-user-flow-light.png) | Light docs / print |
| Diagram | User Flow (Dark) | [`speculators-user-flow-dark.svg`](assets/branding/speculators-user-flow-dark.svg) | [`speculators-user-flow-dark.png`](assets/branding/speculators-user-flow-dark.png) | Dark slides / sites |

## Usage Guidelines

### Logos (Full Lockups)

- Clear space: keep at least the full logo height around all sides
- Minimum size: 120 px width (PNG), no minimum for SVG beyond legible rendering
- Backgrounds: use Black on light; White on dark; Blue as an accent
- Do not: modify colors, add effects, stretch, skew, rotate, or outline the mark

### Icons (Square Mark)

- Use for small spaces (avatars, tiles, compact headers)
- Preserve 1:1 aspect ratio; do not crop or place in a container with rounded mismatch
- Minimum size: 24 px (prefer SVG; use PNG ≥ 64 px)
- Do not add shadows, strokes, gradients, or recolor arbitrarily

### Model Icons

- For diagrams referencing model abstractions; not a replacement for the primary logo
- Keep consistent scale, margins, and stroke weight with surrounding diagram elements

### User Flow Diagrams

- Use the Light or Dark theme as-is; do not edit labels, arrows, or spacing
- Prefer SVG for clarity and printing; scale uniformly

## Do and Avoid
| Do | Avoid |
| --- | ----- |
| Use supplied SVG for scalable quality | Recreating marks from scratch |
| Maintain clear space and aspect ratio | Distorting, squeezing, or rotating |
| Choose color for sufficient contrast | Recoloring to arbitrary hues |
| Use a single primary mark per surface | Layering multiple logos |
| Attribute the project when citing | Using assets to imply endorsement |

## License & Attribution

- License: assets are distributed under the [Apache License 2.0](https://github.com/neuralmagic/speculators/blob/main/LICENSE) for identifying the Speculators open-source project
- Do not use assets to misrepresent affiliation or create derivative brand identities

## Changelog

- v1: Initial public set of logos, icons, model icons, and user flow diagrams
