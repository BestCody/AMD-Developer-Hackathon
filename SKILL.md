---
name: aperture-design
description: Use this skill to generate well-branded interfaces and assets for Aperture, either for production or throwaway prototypes/mocks/etc. Contains essential design guidelines, colors, type, fonts, assets, and UI kit components for prototyping.
user-invocable: true
---

Read the README.md file within this skill, and explore the other available files.

Aperture converts every kind of enterprise data (PDFs, videos, screenshots, logs, emails) into one Universal Intermediate Representation (UIR) for AI agents — the visual language is a premium, Apple-derived system (single blue accent, alternating light/dark full-bleed tiles, near-invisible chrome, exactly one drop-shadow) applied to a data-infrastructure product.

If creating visual artifacts (slides, mocks, throwaway prototypes, etc), copy assets out of `assets/` and reference tokens from `styles.css` to create static HTML files for the user to view. If working on production code, copy assets and read `readme.md` + `tokens/*.css` to become an expert in designing with this brand.

If the user invokes this skill without any other guidance, ask them what they want to build or design, ask some questions, and act as an expert designer who outputs HTML artifacts _or_ production code, depending on the need.

Key things to know before you design anything:
- One accent color only (`--accent-primary`, Action Blue #0066cc) — never introduce a second brand hue.
- Exactly one shadow in the whole system (`--shadow-product`), reserved for product imagery — never on cards, buttons, or text.
- Body copy is 17px, never 16px. Weight ladder is 300/400/600/700 — 500 is never used.
- Full-bleed section tiles alternate light/parchment/dark — the color change IS the divider, never a border.
- Press state is always `scale(0.95)` — never a color or shadow change.
- "Aperture" is a placeholder brand name (no company name was supplied) — ask the user if they have a real name before shipping anything customer-facing.
