# Trajectory Reconstruction — Baseline Comparison

Mean Haversine Error (km), mean +/- std over seeds.

| Model                  |     Takeoff (km) |      Cruise (km) |     Landing (km) |     Overall (km) |
|------------------------|------------------|------------------|------------------|------------------|
| DLinear                |   43.99 +/- 0.18 |   21.01 +/- 0.03 |   35.47 +/- 0.06 |   25.59 +/- 0.00 |
| KNN (k=5)              |   66.19 +/- 0.00 |   32.49 +/- 0.00 |   16.92 +/- 0.00 |   38.69 +/- 0.00 |
| PatchTST               |   55.77 +/- 1.80 |   23.04 +/- 0.90 |   38.17 +/- 2.28 |   28.89 +/- 0.61 |
| iTransformer           |   47.89 +/- 0.08 |   19.79 +/- 0.40 |   37.58 +/- 1.35 |   25.41 +/- 0.14 |

## Comparison vs. published Table I (`results/paper/table1_baselines.json`)

| Model | Phase | Rerun mean | Paper mean | Delta (km) |
|-------|-------|---------|---------|--------|
| DLinear | takeoff | 43.99 | 43.99 | -0.00 |
| DLinear | cruise | 21.01 | 21.01 | -0.00 |
| DLinear | landing | 35.47 | 35.47 | +0.00 |
| KNN (k=5) | takeoff | 66.19 | 66.19 | +0.00 |
| KNN (k=5) | cruise | 32.49 | 32.49 | +0.00 |
| KNN (k=5) | landing | 16.92 | 16.92 | +0.00 |
| PatchTST | takeoff | 55.77 | 55.77 | -0.00 |
| PatchTST | cruise | 23.04 | 23.04 | -0.00 |
| PatchTST | landing | 38.17 | 38.17 | -0.00 |
| iTransformer | takeoff | 47.89 | 47.89 | +0.00 |
| iTransformer | cruise | 19.79 | 19.79 | +0.00 |
| iTransformer | landing | 37.58 | 37.58 | -0.00 |

*Large delta expected if sample data, different LLM snapshot, or TSLib fixed `pred_len` protocol. TrajRAG/KNN should match when full data and `MEAN_HAVERSINE_RAG.json` are present.*
