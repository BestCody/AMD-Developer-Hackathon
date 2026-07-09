Floating frosted notification bar — reuses the source's `floating-sticky-bar` treatment for transient messages (job complete, export ready).

```jsx
<Toast action={<Button variant="primary">View</Button>} onDismiss={() => setShow(false)}>
  3 documents processed
</Toast>
```
