

&#x20; 1. \*\*Authority scorer never called in forward pass.\*\* v1 `jusdef.py` lines 133-136 had identical branches; `self.authority\_scorer` instantiated but unused. The `use\_authority` flag changes parameter count but

&#x20;  those parameters never receive gradients.



&#x20; 2. \*\*r2 edge is forward-only.\*\* Messages flow sec→conc, never back. Concept embeddings never reach section, document, or label embeddings during the forward pass. Verified by direct forward-pass test on

&#x20; 2026-05-17: randomising operators changes only `conc` embeddings (Δ=0.20), all other node types Δ=0.



&#x20; 3. \*\*Only 5 of 8 paper edge types are constructed.\*\* Missing: r3 (conc-conc ontology), r5 (auth-auth hierarchy), r6 (auth-conc link).



&#x20; 4. \*\*Authority features not on edges.\*\* Trainer's `edge\_attr\_dict` only carries `operator` and `priority`; no per-edge `auth\_type`/`level`/`recency`. Authority scorer cannot compute π(m) per edge without

&#x20; these.



&#x20; 5. \*\*Schedule paper-code mismatch.\*\* Paper §4.4 describes Stage 1 (0-4) / Stage 2 (5-14) / Stage 3 (15+); code defaults `stage1\_end=50`, `stage2\_end=100`. All canonical v1 runs early-stop in Stage 1.



&#x20; 





`## Phase 0 — F1 and F2 fixes (2026-05-21)`



You can of course tweak the tone, but this covers the causal story in enough detail.



\*\*\*



\## Phase 0 — F1 and F2 fixes (2026-05-21)



\### F1: Reverse mention edge and tuple-key filtering



The first failure mode (F1) was that section embeddings were \*\*operator-blind\*\*: randomising r2 operators (`sec → conc`) changed concept embeddings but left section embeddings unchanged. This contradicted the intended architecture, where defeasible reasoning at concept level should propagate back to sections via a reverse edge (`conc → sec`).



The root cause turned out to be a subtle interaction between PyTorch Geometric’s `HeteroConv` and how I filtered edge types. I originally derived the set of edge types per layer via:



```python

conv\_edge\_types = set(hetero\_conv.convs.keys())

```



and then filtered the graph’s `edge\_index\_dict` with:



```python

candidate\_edges = {

&#x20;   k: v

&#x20;   for k, v in edge\_index\_dict.items()

&#x20;   if k != r2\_key and v.size(1) > 0 and k in conv\_edge\_types

}

```



However, `edge\_index\_dict` uses \*\*tuple\*\* keys like `("conc", "mentions\_rev", "sec")`, whereas `hetero\_conv.convs.keys()` can be \*\*strings\*\* internally (e.g. `"conc\_\_mentions\_rev\_\_sec"`) depending on how PyG wraps the modules. The membership test `k in conv\_edge\_types` therefore always returned `False`, silently dropping all edges from `candidate\_edges`. The reverse edge existed in the graph schema, but message passing over it never actually ran. \[pytorch-geometric.readthedocs](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)



The fix was to \*\*store the tuple keys explicitly\*\* at construction time, instead of rediscovering them through PyG’s internal representation. In `JusDef.\_\_init\_\_`, I now build each layer’s `conv\_dict` as before, then cache its keys:



```python

self.hetero\_convs = nn.ModuleList()

self.hetero\_conv\_edge\_types = \[]

for \_ in range(num\_layers):

&#x20;   conv\_dict = {

&#x20;       ("doc", "has\_section", "sec"): SAGEConv(hidden\_dim, hidden\_dim),

&#x20;       ("label", "maps\_to", "conc"): SAGEConv(hidden\_dim, hidden\_dim),

&#x20;       ("label", "parent\_of", "label"): SAGEConv(hidden\_dim, hidden\_dim),

&#x20;       ("sec", "cites", "auth"): SAGEConv(hidden\_dim, hidden\_dim),

&#x20;       ("conc", "mentions\_rev", "sec"): SAGEConv(hidden\_dim, hidden\_dim),

&#x20;   }

&#x20;   if not use\_dmp:

&#x20;       conv\_dict\[("sec", "mentions", "conc")] = SAGEConv(hidden\_dim, hidden\_dim)



&#x20;   self.hetero\_conv\_edge\_types.append(set(conv\_dict.keys()))

&#x20;   self.hetero\_convs.append(HeteroConv(conv\_dict, aggr="sum"))

```



In `forward`, the filter uses this cached set:



```python

conv\_edge\_types = self.hetero\_conv\_edge\_types\[layer\_idx]

candidate\_edges = {

&#x20;   k: v

&#x20;   for k, v in edge\_index\_dict.items()

&#x20;   if k != r2\_key and v.size(1) > 0 and k in conv\_edge\_types

}

```



Now the membership test is tuple-vs-tuple and HeteroConv actually runs on all intended relations, including the reverse mention edge. The unit test `test\_F1\_operators\_reach\_sections` then passes: randomising r2 operators changes section embeddings (max |Δ| > 0), demonstrating that operator-aware concept updates are propagating back to sections as designed. \[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/149322011/9c3edef9-5bad-4bf8-b6fd-05f65e4c32c7/kg\_builder.py)



\### F2: Authority scorer on the gradient path



The second failure mode (F2) was more insidious: the authority scorer module existed and was even being instantiated, but its parameters received \*\*zero gradient\*\* during backprop. From a training perspective, this is equivalent to having no authority model at all.



F2 required several coordinated changes:



1\. \*\*Authority features on r2 edges (test graphs)\*\*  

&#x20;  The model expects each r2 edge to carry not only the operator and base priority, but also three authority features: `auth\_type`, `auth\_level`, and `auth\_recency`. On the laptop, these did not yet exist in the cached graphs. I introduced a patch script:



&#x20;  ```python

&#x20;  PATHS = \[

&#x20;      "data/processed/graphs/test\_graphs.pt",

&#x20;      # later extended to train/validation

&#x20;  ]

&#x20;  R2 = ("sec", "mentions", "conc")

&#x20;  R2\_REV = ("conc", "mentions\_rev", "sec")



&#x20;  for path in PATHS:

&#x20;      graphs = torch.load(path, map\_location="cpu")

&#x20;      rng = np.random.RandomState(42)



&#x20;      for g in graphs:

&#x20;          for et in (R2, R2\_REV):

&#x20;              if et not in g.edge\_types:

&#x20;                  continue

&#x20;              store = g\[et]

&#x20;              n = store.edge\_index.size(1)

&#x20;              if n == 0:

&#x20;                  store.auth\_type = torch.zeros(0, dtype=torch.long)

&#x20;                  store.auth\_level = torch.zeros(0, dtype=torch.float)

&#x20;                  store.auth\_recency = torch.zeros(0, dtype=torch.float)

&#x20;                  continue

&#x20;              store.auth\_type = torch.tensor(rng.randint(0, 6, size=n), dtype=torch.long)

&#x20;              store.auth\_level = torch.tensor(rng.uniform(0, 1, size=n), dtype=torch.float)

&#x20;              store.auth\_recency = torch.tensor(rng.uniform(0, 1, size=n), dtype=torch.float)

&#x20;  ```



&#x20;  This is explicitly a \*\*synthetic\*\* patch for test-time smoke; real features will be built in `kg\_builder.py` on Ampere. After patching, each r2 edge had the three required tensors, with shapes matching the r2 edge count. \[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/149322011/b9f19c60-fef0-488b-86bb-9bb652e26c3a/train\_jusdef.py)



2\. \*\*Plumbing authority features into the forward pass (tests + model)\*\*  

&#x20;  The test helper `\_forward` in `tests/test\_architecture\_fixes.py` initially only passed `operator` and `priority`. I updated it to opportunistically include the authority fields if present:



&#x20;  ```python

&#x20;  if r2\_key in g.edge\_types and g\[r2\_key].edge\_index.size(1) > 0:

&#x20;      r2 = g\[r2\_key]

&#x20;      attrs = {"operator": r2.operator, "priority": r2.priority}

&#x20;      for k in ("auth\_type", "auth\_level", "auth\_recency"):

&#x20;          if hasattr(r2, k):

&#x20;              attrs\[k] = getattr(r2, k)

&#x20;      edge\_attr\_dict = {r2\_key: attrs}

&#x20;  ```



&#x20;  In `JusDef.forward`, I replaced the dead branch that always set `r2\_pri = r2\_pri\_raw` with a real authority path:



&#x20;  ```python

&#x20;  r2\_ops = edge\_attr\_dict\[r2\_key]\["operator"]

&#x20;  r2\_pri\_raw = edge\_attr\_dict\[r2\_key]\["priority"]



&#x20;  if (

&#x20;      self.use\_authority

&#x20;      and r2\_key in edge\_attr\_dict

&#x20;      and "auth\_type" in edge\_attr\_dict\[r2\_key]

&#x20;      and "auth\_level" in edge\_attr\_dict\[r2\_key]

&#x20;      and "auth\_recency" in edge\_attr\_dict\[r2\_key]

&#x20;  ):

&#x20;      r2\_pri = self.authority\_scorer(

&#x20;          edge\_attr\_dict\[r2\_key]\["auth\_type"],

&#x20;          edge\_attr\_dict\[r2\_key]\["auth\_level"],

&#x20;          edge\_attr\_dict\[r2\_key]\["auth\_recency"],

&#x20;      )

&#x20;  else:

&#x20;      r2\_pri = r2\_pri\_raw



&#x20;  dmp\_out = self.dmp\_layers\[layer\_idx](

&#x20;      src\_embs, dst\_ids, r2\_ops, r2\_pri, concept\_ids, num\_conc

&#x20;  )

&#x20;  ```



&#x20;  This ensures the DMP layer sees either learned priorities (when authority features are present) or the original priorities (for backwards compatibility).



3\. \*\*Fixing autograd in the defeat mask (scatter\_reduce\_)\*\*  

&#x20;  The most subtle bug was inside `compute\_defeat\_mask` in `dmp\_layer.py`. The original implementation computed per-concept maxima using an \*\*in-place\*\* scatter:



&#x20;  ```python

&#x20;  defeat\_score = operators.float() \* 1000.0 + priorities

&#x20;  num\_groups = dst\_nodes.max().item() + 1

&#x20;  group\_max = torch.full((num\_groups,), -1e9, device=operators.device)

&#x20;  group\_max.scatter\_reduce\_(0, dst\_nodes, defeat\_score, reduce="amax")

&#x20;  priority\_gap = group\_max\[dst\_nodes] - defeat\_score - 0.001

&#x20;  ```



&#x20;  Here, `group\_max` is created without `requires\_grad`, then mutated in place by `scatter\_reduce\_`. PyTorch’s autograd does not retroactively tie that mutated tensor back to `defeat\_score`: `defeat\_score.requires\_grad` is `True`, but `group\_max.requires\_grad` stays `False`, and the gradient signal from the max operation is effectively lost. As a result, `priority\_gap` only depended on `-defeat\_score`, and for the max element in each group the two terms partially cancelled. The Straight-Through Estimator (STE) then had almost no meaningful gradient to propagate back to `priorities` and the authority scorer. \[stackoverflow](https://stackoverflow.com/questions/60949682/how-to-break-pytorch-autograd-with-in-place-ops)



&#x20;  The fix was to switch to the \*\*non-in-place\*\* scatter\_reduce API, which returns a new tensor that correctly participates in the graph:



&#x20;  ```python

&#x20;  defeat\_score = operators.float() \* 1000.0 + priorities

&#x20;  num\_groups = dst\_nodes.max().item() + 1

&#x20;  base = torch.full((num\_groups,), -1e9, device=operators.device)

&#x20;  group\_max = base.scatter\_reduce(

&#x20;      0, dst\_nodes, defeat\_score, reduce="amax", include\_self=True

&#x20;  )

&#x20;  priority\_gap = group\_max\[dst\_nodes] - defeat\_score - 0.001

&#x20;  ```



&#x20;  After this change, both `group\_max` and `priority\_gap` correctly report `requires\_grad=True`, and autograd can connect the defeat mask back to the authority priorities. \[glaringlee.github](https://glaringlee.github.io/autograd.html)



4\. \*\*Softening the defeat mask usage to preserve gradients\*\*  

&#x20;  Finally, the way the defeat mask was used inside the attention block also mattered for gradient flow. The initial pattern combined STE with hard boolean thresholding:



&#x20;  ```python

&#x20;  active\_msg = msg \* defeat\_mask.unsqueeze(-1)

&#x20;  attn\_raw = self.attn(active\_msg).squeeze(-1)

&#x20;  attn\_raw = attn\_raw.masked\_fill(defeat\_mask < 0.5, -1e9)

&#x20;  attn\_exp = torch.exp(attn\_stable)

&#x20;  attn\_exp = attn\_exp \* (defeat\_mask > 0.5).float()

&#x20;  ```



&#x20;  The STE already provides a surrogate gradient through `defeat\_mask`, but the downstream boolean masks (`< 0.5`, `> 0.5`) turn it back into a hard gate, effectively killing the continuous signal. To keep the model faithful to the defeasible semantics while retaining a usable gradient, I switched to a \*\*soft-mask\*\* design: \[discuss.pytorch](https://discuss.pytorch.org/t/how-does-applying-a-mask-to-the-output-affect-the-gradients/126520)



&#x20;  ```python

&#x20;  defeat\_mask = compute\_defeat\_mask(operators, priorities, concept\_ids, self.temperature)



&#x20;  # Soft-mask messages

&#x20;  active\_msg = msg \* defeat\_mask.unsqueeze(-1)



&#x20;  # Compute raw attention from msg (not already-masked active\_msg)

&#x20;  attn\_raw = self.attn(msg).squeeze(-1)



&#x20;  # Group-wise normalization

&#x20;  attn\_base = torch.full((num\_dst,), -1e9, device=device)

&#x20;  attn\_max = attn\_base.scatter\_reduce(

&#x20;      0, dst\_node\_ids, attn\_raw, reduce="amax", include\_self=True

&#x20;  )

&#x20;  attn\_stable = attn\_raw - attn\_max\[dst\_node\_ids]



&#x20;  # Exponentiate and softly weight by defeat\_mask

&#x20;  attn\_exp = torch.exp(attn\_stable) \* defeat\_mask

&#x20;  attn\_sum = torch.zeros(num\_dst, device=device)

&#x20;  attn\_sum.scatter\_add\_(0, dst\_node\_ids, attn\_exp)

&#x20;  attn\_sum = attn\_sum.clamp(min=1e-8)



&#x20;  attn\_weights = attn\_exp / attn\_sum\[dst\_node\_ids]

&#x20;  weighted\_msg = active\_msg \* attn\_weights.unsqueeze(-1)

&#x20;  out = torch.zeros(num\_dst, self.out\_dim, device=device)

&#x20;  out.scatter\_add\_(0, dst\_node\_ids.unsqueeze(-1).expand(-1, self.out\_dim), weighted\_msg)

&#x20;  ```



&#x20;  This keeps the STE mask in the differentiable path: defeated messages get downweighted rather than obliterated by an additional hard mask. Conceptually, this is still a defeasible aggregation, but now the gradient with respect to priorities (and therefore the authority scorer) is rich enough for learning.



With these changes, the test `test\_F2\_authority\_scorer\_in\_gradient\_path` passes: after a backward pass on a simple scalar loss, at least one authority scorer parameter has non-zero gradient. Taken together, F1 and F2 mark the end of “Phase 0”: the architecture now behaves as intended both in forward semantics (operators and authority reaching sections) and in backward semantics (gradients flowing through the defeasible machinery to the learnable parameters.) \[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/149322011/9c3edef9-5bad-4bf8-b6fd-05f65e4c32c7/kg\_builder.py)



\*\*\*





