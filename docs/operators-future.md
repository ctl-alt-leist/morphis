# Future Operator Considerations

Potential operator assignments for GA operations:

| Operation | Operator | Notes |
|-----------|----------|-------|
| Wedge | `^` | Implemented. Metric-free. |
| Reverse | `~` | Implemented. Metric-free. |
| Inverse | `**(-1)` | Implemented. Requires metric. |
| Dot | `@` | Matches numpy convention |
| Hodge dual | `*` | Unary: `*u` |
| Geometric | ? | No clear candidate remaining |

## Open Question: Metrics

Wedge is metric-independent, but dot, Hodge, and geometric all require a metric.

Options to explore:
1. Operators assume Euclidean; use functions for other metrics
2. Store metric on blades via context
3. Thread-local default metric

No decision made yet.
