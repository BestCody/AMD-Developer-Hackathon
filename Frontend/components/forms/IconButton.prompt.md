Circular 44×44 control chip that floats over photography or dense toolbars — always pair with an `aria-label` via `label`.

```jsx
<IconButton icon={<XIcon size={18} />} label="Close" />
<IconButton icon={<ChevronLeftIcon size={18} />} label="Previous" translucent={false} />
```

- `translucent` (default) renders the `surface-chip-translucent` gray chip used over imagery.
- Set `translucent={false}` for a solid canvas + hairline-ring version used on plain surfaces.
- Always exactly 44×44 by default — Apple's documented minimum touch target.
