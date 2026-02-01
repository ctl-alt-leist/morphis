Claude,

So I'm thinking a little more on our `Operator` class. Consider the equation:

$$
b^{ab}_m = G^{ab}_{mn} q^n
$$

where, for a bit of context without digging into the details, $q$ is a source term, $G$ is a geometric transfer matrix,
and $b$ is a field. In our Morphis API, this can be simply done with:

```python
b: Vector
q: Vector
G: Operator

b = G * q
```

Now...my first thought when we added the `*` operator overloading for operators and vectors like this, was that this was
great and straight forward, but it turns out NOT so simple. Don't get me wrong, as it is, it is fine, but I realized
this leads to indexing objects a little differently than I intended.

Here, in the given context, what is most natural is to have G indexed by `G[m, n, a, b]` where `M, N = G.lot` and
`A, B = G.geo`, which in d=3 is just `A, B == 3, 3  # True`. For even more context here, M is the number of sensors and
N is the number of sources. It is more like (m, n) are the lot indices, the collection of things, and (a, b) are the
local geometric indices. We have a collection of (M, N) geometric bivectors `b[a, b]` so to say.

Where this is in a tad bit of friction with our operator class now, is that with the * operation, we run into the fact
that we have operators and vectors with multiple indices, but the contraction isn't being specified. Yet, for an
operator overloading with *, it has to assume something. So, therefore I believe this is where the spec comes into play.

So, when the code in Maxwell had to set up G, it had to set the indexing in as `G[a, b, m, n]`, but this puts the
geometric indices first, and I really always want them last.

But, it really is the case that I want to contract on the `n` in this case, and it really it should be that `b = G * q`
should be doing `b = einsum("mnab, n -> mab", G, q)` (although this would really be with numpy arrays, not morphis
objects).

So, I'm trying to think of elegant ways we can still define this same operator overloading, but be able to set up the
operator indexing in arbitrary ways.
