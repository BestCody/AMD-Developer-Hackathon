Segmented control for switching between 2-4 views — modeled directly on the source repo's own UMR / UIR JSON result toggle.

```jsx
<Tabs tabs={[{value:"umr",label:"UMR"},{value:"json",label:"UIR JSON"}]} active={view} onChange={setView} />
```
