Aperture's action button — five variants covering every button grammar in the source system.

```jsx
<Button variant="primary">Buy</Button>
<Button variant="secondary-pill">Learn more</Button>
<Button variant="dark-utility">Sign In</Button>
<Button variant="pearl-capsule">Compare</Button>
<Button variant="store-hero">Get started</Button>
```

Notes:
- `primary` is the only button that should read as "the" action on a screen — full pill, Action Blue, `on-primary` text.
- `secondary-pill` pairs with `primary` when two CTAs appear together ("Learn more" / "Buy").
- `dark-utility` is for nav-bar actions (Sign In, Bag) — never a full-pill radius.
- `disabled` drops opacity to 0.45 and disables the press micro-interaction.
- Press state is always `transform: scale(0.95)` — never a color or shadow change, per the source's Do's/Don'ts.
