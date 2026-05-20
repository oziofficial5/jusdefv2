

&#x20; 1. \*\*Authority scorer never called in forward pass.\*\* v1 `jusdef.py` lines 133-136 had identical branches; `self.authority\_scorer` instantiated but unused. The `use\_authority` flag changes parameter count but

&#x20;  those parameters never receive gradients.



&#x20; 2. \*\*r2 edge is forward-only.\*\* Messages flow sec→conc, never back. Concept embeddings never reach section, document, or label embeddings during the forward pass. Verified by direct forward-pass test on

&#x20; 2026-05-17: randomising operators changes only `conc` embeddings (Δ=0.20), all other node types Δ=0.



&#x20; 3. \*\*Only 5 of 8 paper edge types are constructed.\*\* Missing: r3 (conc-conc ontology), r5 (auth-auth hierarchy), r6 (auth-conc link).



&#x20; 4. \*\*Authority features not on edges.\*\* Trainer's `edge\_attr\_dict` only carries `operator` and `priority`; no per-edge `auth\_type`/`level`/`recency`. Authority scorer cannot compute π(m) per edge without

&#x20; these.



&#x20; 5. \*\*Schedule paper-code mismatch.\*\* Paper §4.4 describes Stage 1 (0-4) / Stage 2 (5-14) / Stage 3 (15+); code defaults `stage1\_end=50`, `stage2\_end=100`. All canonical v1 runs early-stop in Stage 1.



&#x20; ## v2 Phase 0 fixes (in progress)



&#x20; - \[ ] F1: Add reverse mention edge ('conc','mentions\_rev','sec') with operator labels copied from forward edge.

&#x20; - \[ ] F2: Wire authority scorer into forward pass; add authority features to edges in kg\_builder.

&#x20; - \[ ] F3: Add missing edge types r3, r5, r6 to graph and model.

&#x20; - \[ ] F4: Make label embeddings refine via concept-aware C\&S step.

&#x20; - \[ ] F5: Unit tests verifying each fix actually executes during forward pass.

