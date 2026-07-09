# Aperture — Console UI Kit

The product itself: an upload-first flow. Drop a document, watch a premium converging-rings loading animation while it's processed, and see it land as a file icon in the side "folder" — a green badge pops onto the icon's bottom-right corner the instant conversion completes. A left icon rail switches between **Upload** and **Copilot**, a grounded chat surface that answers questions using only the documents you've converted.

This intentionally does not mirror `BestCody/AMD-Developer-Hackathon`'s own dev-tool UI (single-page form, dark glassmorphism) — that repo is grounding for the product's real mechanics (pipeline stages, UIR schema), not a visual reference. The visual language here is 100% `DESIGN-apple.md`.

- `IconRail.jsx` — left icon rail (Upload / Copilot)
- `UploadStage.jsx` — dropzone empty state, the converging-rings converting animation, the side file "folder," and the badge-pop micro-interaction
- `CopilotChat.jsx` — chat tab grounded in converted documents
- `index.html` — click the dropzone (or drag anything onto it) to simulate a conversion; add a few files, then switch to Copilot
