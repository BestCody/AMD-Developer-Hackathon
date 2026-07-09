Centered modal — Aperture addition. Canvas surface, `radius-lg`, no shadow on the dialog itself (only the dim scrim separates it, per the source's "no shadow on chrome" rule).

```jsx
<Dialog open={open} title="Delete document?" onClose={() => setOpen(false)}
  actions={<><Button variant="secondary-pill" onClick={close}>Cancel</Button><Button variant="primary" onClick={confirm}>Delete</Button></>}>
  This removes the UIR JSON and its Weaviate chunks.
</Dialog>
```
