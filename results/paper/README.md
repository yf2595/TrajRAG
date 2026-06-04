# Published paper results (verification snapshots)

Aggregate numbers transcribed from [TrajRAG_ADS_ICDM2026.pdf](../../TrajRAG_ADS_ICDM2026.pdf) for the **controlled benchmark** (Tables I, III–V). Use them to verify local reruns when flight data or API access is limited.

| File | Paper reference |
|------|-----------------|
| `table1_baselines.json` | Table I — TrajRAG vs. baselines |
| `table3_llm_sensitivity.json` | Table III — LLM backbone sweep |
| `table4_route_generalization.json` | Table IV — S1 vs. S2 memory (Δ MHE, km) |
| `table5_ablations.json` | Table V — component ablations |

```bash
cd iTransformer-DLinear-PatchTST
python compile_results.py --compare-paper
```

After a full TrajRAG eval, save per-flight errors to `data/MULTI_ROUTE/RESULTS/MEAN_HAVERSINE_RAG.json` and run `compile_results.py` again to compare **TrajRAG** against `table1_baselines.json`.
