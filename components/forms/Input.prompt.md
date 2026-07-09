Text input with two grammars: the source's pill-shaped search field, and a standard rectangular field for general forms.

```jsx
<Input variant="pill" icon={<SearchIcon size={14} />} placeholder="Search accessories" />
<Input placeholder="Document title" />
```

- Focus state: border shifts to `--accent-primary-focus` with a soft blue glow — matches the source's 2px focus-ring token.
- Error/validation states are not defined in the source (documented gap); Aperture uses `--status-error` if you need one — flag it as an intentional addition.
