# Aperture Design System

**Aperture** converts every kind of enterprise data — PDFs, videos, screenshots, logs, emails, semi-structured exports — into one **Universal Intermediate Representation (UIR)**: a single, structured, chunked, embedded format that AI agents can consume without hallucinating or burning excess tokens.

## Mission

Unstructured, multimodal data is enterprises' biggest bottleneck and their biggest untapped treasure. Every company is drowning in PDFs, screenshots, videos, logs, emails, and semi-structured sludge. Models keep getting smarter but the inputs keep getting messier — causing RAG systems to hallucinate, agents to break in subtle, expensive ways, and critical workflows to still lean on human QA. Aperture is the normalization layer that fixes the input side of that equation: one pipeline, one schema, one clean representation — regardless of what the source document looked like.

## Sources this system was built from

- **`uploads/DESIGN-apple.md`** — a full visual-language teardown of Apple's web presence (colors, type, spacing, elevation, component grammar, do's/don'ts, responsive rules). Aperture's entire visual foundation is lifted directly from this spec, per the brief's request for "a premium product using DESIGN-apple.md." Re-read it if you need the original component rationale.
- **`uploads/logo.png`** — the one brand asset provided: an abstract compass/interlocking-arrows mark. Copied verbatim into `assets/logo/`. No wordmark, no color logo variant, and no separate icon set were supplied alongside it.
- **GitHub — [`BestCody/AMD-Developer-Hackathon`](https://github.com/BestCody/AMD-Developer-Hackathon)** — the reference implementation of the UIR pipeline (Python, PDF → layout → tables → chunk → enrich → embed → UIR JSON, with a Flask LAN test UI). This is where the product's real mechanics, vocabulary, and existing dev-facing UI (`templates/index.html`, `static/style.css`) came from — it grounds the "Console" UI kit's screens and copy in a working system rather than invented functionality. **Explore this repo further** if you want to extend the console with real pipeline states, the actual UIR JSON schema (`docs/uir.schema.json`), or additional stages (OCR, LayoutLMv3, Tier-3 Florence-2 captioning) — this system only recreates the surfaces that repo's `web.py` / `templates/index.html` already define, at premium fidelity.

## What Aperture is, concretely

A single pipeline (see the source repo) takes a document in any modality, runs it through ingestion → layout classification → table extraction → chunking → semantic enrichment → embedding, and emits **UIR v1.0 JSON**: a strict, versioned schema an agent can retrieve against with high confidence and low token cost. The product wraps that pipeline in two surfaces:

1. **Marketing site** — explains the mission (messy multimodal data → one clean representation) and converts visitors into pipeline users.
2. **Console** (the product itself) — upload a document, watch it move through the pipeline stages in real time, inspect the resulting UIR as either a UMR (agent-readable markdown) or raw JSON, and export or push it to a vector index.

## Brand name

No company name was supplied with this brief — only a mission statement, a logo mark, and a visual-language spec. **"Aperture"** was chosen for this design system (an aperture converges scattered light into a single focused image — the same operation this product performs on scattered data) and is easily renamed. **Ask if you'd like a different name** — every token, component, and screen here uses it only as a placeholder wordmark; nothing is hard-coded elsewhere.

---

## Components

No component library was defined in either source (`DESIGN-apple.md` is a visual-language spec, not a component inventory; the GitHub repo is a backend pipeline with one plain HTML page). Per the brand-guidelines-only path, Aperture authors a standard primitive set sized to its own console + marketing needs, styled entirely from the tokens above:

- **Forms** (`components/forms/`) — `Button` (primary / secondary-pill / dark-utility / pearl-capsule / store-hero), `IconButton`, `Input` (default / pill), `Select`, `Checkbox`, `Radio`, `Switch`
- **Feedback** (`components/feedback/`) — `Badge` (pipeline status pill), `Tag` (selectable option chip), `Toast` (frosted floating notification), `Tooltip`
- **Navigation** (`components/navigation/`) — `Tabs` (segmented toggle, modeled on the source repo's own UMR/JSON view switch)
- **Overlay** (`components/overlay/`) — `Dialog`
- **Data** (`components/data/`) — `Card` (the source's `store-utility-card` grammar — the one all-purpose content container)

### Intentional additions
Components with no counterpart in either source, added because the product genuinely needs them:
- `Badge`, `Toast` — pipeline job states (processing/complete/failed) have no equivalent in a static marketing site.
- `Select`, `Checkbox`, `Radio`, `Switch` — standard form controls for pipeline configuration (chunking mode, embeddings on/off) that a product page never needed.
- `Dialog`, `Tooltip` — general-purpose UI utility with no direct source precedent, styled to the same quiet, no-shadow-on-chrome rule as everything else.

---

## Content fundamentals

Copy voice is inferred from `DESIGN-apple.md`'s tone (confident, spare, product-first) applied to Aperture's own subject matter — data infrastructure, not consumer electronics. No existing marketing copy was supplied, so treat this as a starting posture, not gospel.

- **Second person, rarely first.** Address the reader directly ("Drop a PDF and watch it become structure your agent can trust") — never "we/our" chest-thumping. The one first-person moment allowed is a quiet, factual claim about the pipeline itself ("we chunk at ~256 tokens with overlap").
- **Declarative, not hypey.** Sentences state what the product does, plainly, the way Apple states "Aperture reads the page." No "revolutionary," "game-changing," "unlock," or "supercharge." The source repo's own README is a good tonal reference: *"Pipeline that ingests PDF documents and emits Universal Intermediate Representation (UIR v1.0) JSON."* — factual, specific, no adjectives doing the work nouns should do.
- **One idea per line.** Headlines are short noun phrases or short imperative clauses ("One format. Every modality." / "Stop feeding your agent garbage."). Taglines are a single sentence, never a list.
- **Precision over vibes.** When a number is available, use it plainly — "256-token chunks," "<10s per document," "384-dimension embeddings" — the way an engineer would write it, not rounded into marketing math.
- **Sentence case everywhere.** Headlines, buttons, nav labels — sentence case, not Title Case, matching the Apple source exactly ("Learn more," not "Learn More").
- **No emoji.** Neither the Apple source nor the reference codebase uses emoji anywhere in UI copy; keep it out of Aperture's copy too. The only glyphs are line icons and the ⬆ upload icon in the reference app (worth replacing with a proper icon, see Iconography).
- **Technical vocabulary is a feature, not a bug.** "UIR," "chunk," "modal_features," "confidence" are said plainly, unglossed, the way the console needs to speak to the technical users who'll actually read a JSON pane. Marketing copy can translate these once per concept, then use the real term.

**Example lines** (written for this system, not sourced from a document): "Every modality, one format." · "Feed your agent structure, not scans." · "From PDF to production-ready in under 10 seconds." · "Confidence scores on everything you didn't have to trust blindly before."

## Visual foundations

Aperture's visual system is `DESIGN-apple.md` applied without invention — read that file for full rationale; this section answers the standing "what does X look like" questions for quick reference.

- **Colors.** One interactive hue system-wide: Action Blue `#0066cc` (`--accent-primary`). No second brand color. Surfaces alternate between pure white, off-white "Parchment" `#f5f5f7`, and three near-black tile tones (`#272729` / `#2a2a2c` / `#252527`) — the surface change itself is the section divider, never a border. Aperture adds one thing the source has no equivalent for: a small status palette (`--status-success/warning/error/processing`) for pipeline job states, kept desaturated and quiet rather than traffic-light loud.
- **Type.** Inter stands in for SF Pro Display/Text (see Fonts note below). Display sizes (déjà ≥19px) carry negative letter-spacing for the "Apple tight" cadence; body runs at 17px/1.44, never 16px. Weight ladder is 300 / 400 / 600 / 700 — 500 is deliberately never used.
- **Spacing.** 8px base unit. Section (tile) padding is 80px vertical; card padding is 24px; button padding is 11×22px. Tiles stack edge-to-edge with zero gap.
- **Backgrounds.** No gradients anywhere as a decorative device — the only "gradient" feeling comes from product photography's own lighting. No textures, no patterns, no hand-drawn illustration. Full-bleed rectangular product tiles are the primary background device; imagery is photographic and realistic, not illustrative.
- **Animation.** Minimal by design. The one system-wide micro-interaction is a `scale(0.95)` press state on every button — no color shift, no shadow pop. Progress fills animate via `width` transition on `cubic-bezier(0.22,0.61,0.36,1)`. No elastic/bounce easing anywhere, no looping decorative animation.
- **Hover states.** The source documents Default and Active only — it explicitly avoids hover as a first-class state (touch-first mindset). Where a hover affordance is still needed for desktop web (links, dark-utility buttons), keep it a subtle background lightening, never a color hue change.
- **Press states.** `transform: scale(0.95)` on every interactive control, always — this is the single system-wide gesture. Never a border, shadow, or background-color change on press.
- **Borders.** Functional, not decorative. A single 1px hairline (`--border-hairline`, `#e0e0e0`) appears on utility cards and configurator chips; everywhere else, "border" is really a soft `rgba(0,0,0,0.08)` ring-shadow, not a hard line.
- **Shadows.** Exactly one shadow exists in the entire system — `3px 5px 30px rgba(0,0,0,0.22)` — and it is reserved for product photography resting on a surface. Never apply it to cards, buttons, dialogs, or text. Aperture's own Card/Dialog components use a 1px ring instead.
- **Protective gradients vs. capsules.** No protective scrim gradients are used over imagery; legibility instead comes from placing text on flat surfaces beside (not on top of) photography, or from the translucent gray capsule (`--surface-chip-translucent`, ~64% alpha) behind small icon controls floating over an image.
- **Layout rules.** Global nav is a fixed, ultra-thin (44px) black bar. A secondary frosted sub-nav (52px, blurred Parchment) can stick below it per-surface. No sidebars in the marketing site; the Console introduces a persistent left rail (an Aperture addition, functional necessity for a working tool the source's marketing pages don't need).
- **Transparency & blur.** Reserved for exactly two functional cases — a sticky sub-nav and a bottom sticky action bar — both `saturate(180%) blur(20px)` over ~80%-opacity Parchment. Never decorative.
- **Imagery color vibe.** Photographic, cool-neutral, high-key studio lighting on white/parchment tiles; moodier and warmer only on atmospheric/environment-style hero imagery. No grain, no duotone, no black-and-white treatment.
- **Corner radii.** Five-step scale, each reserved for one grammar: `0` full-bleed tiles, `5px` rare inline chips, `8px` compact utility buttons, `11px` the one "pearl" capsule button, `18px` utility/content cards, `9999px` (pill) every primary action and tappable chip. Never mix grammars — no "12px, kind of rounded" cards.
- **Cards.** White fill, 1px hairline border, 18px radius, 24px padding, no shadow. Contents stack image → strong title (17/600) → body price/meta (17/400) → text link. This is the one card grammar in the system; don't invent a second "elevated" card variant.

### Note on fonts — flagged substitution

SF Pro (Display + Text) is Apple's proprietary system font and isn't licensed for redistribution. Per `DESIGN-apple.md`'s own guidance, this system loads **Inter** (Google Fonts, variable, weights 300–700) as the open equivalent, with `font-feature-settings: "ss03"` to approximate SF Pro's rounded lowercase "a," tracking nudged tighter on display sizes, and body line-height tightened from 1.47→1.44 for Inter's taller x-height. **If real SF Pro webfont files become available, replace `tokens/fonts.css`'s Google Fonts `@import` with local `@font-face` rules** — every other token stays the same.

## Iconography

- **No icon set was supplied.** The reference codebase's own web UI uses exactly one glyph — a bare ⬆ Unicode character for its upload affordance (`templates/index.html`) — and otherwise has no icon system at all.
- **Recommendation: [Lucide](https://lucide.dev)** as the icon set, loaded from CDN (`https://unpkg.com/lucide@latest`) or via the `lucide-react` package in a real app build. Lucide's 1.5–2px stroke weight and rounded joins sit comfortably next to Inter's rounded terminals and the source's quiet, line-based control chrome (search glyph, chevrons, close ×). This is a **flagged substitution** — swap it out if Aperture licenses a proprietary icon set later.
- **Usage.** Icons are line-only (never filled), sized 14–18px inline with text or 44×44px inside `IconButton`'s circular translucent chip when floating over imagery. No emoji, no icon fonts, no colored/duotone icon treatments.
- **Where icons show up:** search input leading glyph, upload affordance, close/dismiss controls, carousel prev/next, nav hamburger, and Console-specific glyphs (file type, stage checkmarks, download/copy actions).

---

## Index

```
Aperture Design System/
├── readme.md                     ← you are here
├── SKILL.md                      ← Claude Code / Agent Skills entry point
├── styles.css                    ← global stylesheet entry (imports everything below)
├── tokens/
│   ├── fonts.css                 ← Inter + JetBrains Mono @import (flagged SF Pro substitution)
│   ├── colors.css                ← surfaces, text, accent, status, hairlines, shadow
│   ├── typography.css             ← full type ladder (hero → micro-legal)
│   ├── spacing.css                ← 8px-based spacing + radius scale
│   ├── motion.css                 ← press-scale + easing tokens
│   └── base.css                   ← resets, link colors (the only non-token global CSS)
├── assets/
│   └── logo/aperture-mark.png     ← the one supplied brand asset
├── guidelines/                    ← 21 foundation specimen cards (Design System tab: Colors, Type, Spacing, Shape, Motion, Brand)
├── components/
│   ├── forms/                     ← Button, IconButton, Input, Select, Checkbox, Radio, Switch
│   ├── feedback/                  ← Badge, Tag, Toast, Tooltip
│   ├── navigation/                ← Tabs
│   ├── overlay/                   ← Dialog
│   └── data/                      ← Card
└── ui_kits/
    ├── marketing/                 ← full-bleed marketing site (Nav, ProductTile, Footer, index.html)
    └── console/                   ← the product itself (Sidebar, UploadPanel, PipelineProgress, ResultViewer, index.html)
```

Every component and UI kit screen loads `styles.css` + the compiled `_ds_bundle.js` and reads components off `window.ApertureDesignSystem_0a9afd`.

## Caveats & how to help iterate

- **No brand name was supplied.** "Aperture" is a placeholder — tell me the real name and I'll thread it through every screen, the logo alt text, and the readme in one pass.
- **No wordmark or color logo variant was supplied** — only the single mark in `uploads/logo.png`. If you have an SVG, a wordmark lockup, or usage guidelines, drop them in and I'll wire them into `assets/logo/` and the Brand cards.
- **Fonts are a flagged substitution.** Inter stands in for SF Pro (proprietary, not redistributable) per `DESIGN-apple.md`'s own guidance. If real SF Pro webfont files ever become licensable for this use, hand them over and I'll swap `tokens/fonts.css` to local `@font-face`.
- **No icon set was supplied.** Lucide (CDN) is recommended and referenced above, but nothing is wired into the UI kits yet beyond a few inline SVGs — say the word and I'll standardize every icon on Lucide.
- **Components are a standard set, not a mined inventory** — `DESIGN-apple.md` is a visual-language spec, not a component library, and the GitHub repo is a backend pipeline with one plain HTML page. I sized the primitive set (Button, Input, Card, etc.) to what the marketing site + console actually need. If you have a real Figma file or component codebase for Aperture, attach it and I'll rebuild this list against the real inventory.
- **UI kit copy is written, not sourced** — no existing marketing copy or console UI copy was provided beyond the one reference repo's dev-tool page. Treat every headline/tagline as a draft.

**My ask:** tell me the real product name (or confirm "Aperture" sticks), and point me at any of — a real logo/wordmark, an icon set, or an existing Figma/codebase for the actual product UI — and I'll do another full pass to bring this from "premium placeholder" to "your actual brand."
