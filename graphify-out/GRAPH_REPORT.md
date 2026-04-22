# Graph Report - C:\Users\sufiy\Downloads\DP Scam\dontredeem  (2026-04-22)

## Corpus Check
- 24 files · ~1,583,998 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 95 nodes · 117 edges · 21 communities detected
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 11 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]

## God Nodes (most connected - your core abstractions)
1. `run_pipeline()` - 11 edges
2. `main()` - 8 edges
3. `load_sms()` - 7 edges
4. `load_dailydialog()` - 7 edges
5. `score()` - 6 edges
6. `load_ftc()` - 6 edges
7. `ScamDataset` - 6 edges
8. `SessionStateManager` - 6 edges
9. `ScamResult` - 5 edges
10. `ScamScorer` - 5 edges

## Surprising Connections (you probably didn't know these)
- `text_model()` --calls--> `score()`  [INFERRED]
  C:\Users\sufiy\Downloads\DP Scam\dontredeem\src\text.py → C:\Users\sufiy\Downloads\DP Scam\dontredeem\models_local\DistillBertini\files\infer.py
- `Benchmark all audio files through the pipeline and collect results.` --uses--> `SessionStateManager`  [INFERRED]
  C:\Users\sufiy\Downloads\DP Scam\dontredeem\utils_and_tests\benchmark_all.py → C:\Users\sufiy\Downloads\DP Scam\dontredeem\src\analytics.py
- `run_pipeline()` --calls--> `SessionStateManager`  [INFERRED]
  C:\Users\sufiy\Downloads\DP Scam\dontredeem\src\main.py → C:\Users\sufiy\Downloads\DP Scam\dontredeem\src\analytics.py
- `run_pipeline()` --calls--> `transcribe()`  [INFERRED]
  C:\Users\sufiy\Downloads\DP Scam\dontredeem\src\main.py → C:\Users\sufiy\Downloads\DP Scam\dontredeem\src\text.py
- `run_pipeline()` --calls--> `audio_model()`  [INFERRED]
  C:\Users\sufiy\Downloads\DP Scam\dontredeem\src\main.py → C:\Users\sufiy\Downloads\DP Scam\dontredeem\src\audio.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.19
Nodes (19): balance(), clean_text(), contains_scam_pattern(), load_dailydialog(), load_ftc(), load_sms(), main(), make_record() (+11 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (12): audio_model(), extract_audio_features(), final_decision(), fuse_scores(), Fuses the scores continuously for each window.     S_t = W_a*A_t + W_t*T_t + W_, If S_final > 0.7 -> Likely Scam     If 0.4 < S_final <= 0.7 -> Suspicious, load_diarization(), run_pipeline() (+4 more)

### Community 2 - "Community 2"
Cohesion: 0.26
Nodes (9): _confidence_label(), main(), ============================================================================= S, Wrapper around the fine-tuned DistilBERT model.      Parameters     ---------, _resolve_device(), ScamResult, ScamScorer, score() (+1 more)

### Community 3 - "Community 3"
Cohesion: 0.2
Nodes (5): Asymmetric EMA Ratchet with Peak-Hold Floor.                  - alpha_rise: we, Records the window and returns the temporally smoothed EMA score., RiskAggregator, SessionStateManager, Benchmark all audio files through the pipeline and collect results.

### Community 4 - "Community 4"
Cohesion: 0.25
Nodes (7): Dataset, evaluate(), get_device(), main(), ============================================================================= D, ScamDataset, train_epoch()

### Community 5 - "Community 5"
Cohesion: 0.5
Nodes (1): EndpointHandler

### Community 6 - "Community 6"
Cohesion: 1.0
Nodes (0): 

### Community 7 - "Community 7"
Cohesion: 1.0
Nodes (0): 

### Community 8 - "Community 8"
Cohesion: 1.0
Nodes (0): 

### Community 9 - "Community 9"
Cohesion: 1.0
Nodes (0): 

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (1): Score a single text and return a ScamResult.

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (1): Score a list of texts efficiently in mini-batches.

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (0): 

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **17 isolated node(s):** `============================================================================= S`, `Wrapper around the fine-tuned DistilBERT model.      Parameters     ---------`, `Score a single text and return a ScamResult.`, `Score a list of texts efficiently in mini-batches.`, `============================================================================= S` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 6`** (2 nodes): `localize_models.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 7`** (2 nodes): `localize_pyannote.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 8`** (1 nodes): `cudatest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 9`** (1 nodes): `example_usage.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (1 nodes): `Score a single text and return a ScamResult.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 11`** (1 nodes): `Score a list of texts efficiently in mini-batches.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (1 nodes): `debug_hf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `debug_hf2.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `debug_hf3.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `generateaudio.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `get_auto_map.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `get_auto_map_to_file.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `test_load.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `test_load2.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `test_pyannote.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_pipeline()` connect `Community 1` to `Community 3`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **Why does `text_model()` connect `Community 1` to `Community 2`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `score()` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `run_pipeline()` (e.g. with `SessionStateManager` and `parse_metadata()`) actually correct?**
  _`run_pipeline()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `============================================================================= S`, `Wrapper around the fine-tuned DistilBERT model.      Parameters     ---------`, `Score a single text and return a ScamResult.` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._