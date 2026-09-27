# Method and membership functions

Source: `backend/fuzzy_smartphone_mamdani.py`, functions `fuzzify`, `build_rule_base`, `infer_mamdani_score` and `fallback_score`.

`tri(a,b,c)` is triangular; `trap(a,b,c,d)` is trapezoidal. Values outside a set's support have zero membership. These values are implementation choices, not empirically fitted thresholds.

| Feature | Low / small / old / entry | Medium / recent / mid | High / large / new / flagship |
| --- | --- | --- | --- |
| RAM, GB | trap(0,0,2,4) | tri(3,6,8) | trap(6,8,16,24) |
| Battery, mAh | trap(0,0,3000,4000) | tri(3500,4500,5500) | trap(5000,6000,8000,10000) |
| Rear camera, MP | trap(0,0,12,32) | tri(24,50,64) | trap(50,108,200,250) |
| Launch price, USD | trap(0,0,250,500) | tri(350,700,1000) | trap(800,1200,2000,3000) |
| Launch year | trap(2014,2014,2018,2020) | tri(2019,2022,2024) | trap(2023,2024,2025,2026) |
| Processor heuristic | trap(0,0,35,50) | tri(40,60,75) | trap(70,85,100,100) |
| Storage, GB | trap(0,0,64,128) | tri(64,128,256) | trap(128,256,512,1024) |
| Output score | trap(0,0,30,45) | tri(35,55,75) | trap(65,80,100,100) |

For each rule, antecedent membership gives its firing strength. The output set is clipped at that strength. Maximum aggregation combines the clipped sets. The final score is `sum(y * membership(y)) / sum(membership(y))`, rounded to two decimals.

An example rule is: high RAM AND flagship processor implies a high score. Budget and selected priorities add rules to the base rule collection. Duplicate priorities are normalized so selecting the same feature twice does not repeat the rules.

Budget candidate bands: low ≤500 USD; medium 300–1,000 USD; high ≥700 USD. If the band leaves fewer than 20 phones, the implementation falls back to all storage-qualified phones. This is deliberately documented and covered by a regression test; it is not a hard spending guarantee.

Storage uses the explicit `Storage_GB` column, then a GB/TB suffix in the model name. RAM alone cannot establish storage. Missing required numeric features exclude a row. The data loader is cached, so restart the server after replacing the dataset.
