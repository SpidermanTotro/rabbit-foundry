# Unified Bunny Lineage

Rabbit Foundry is the canonical research home for the Bunny projects, but **merge does not mean erase provenance**.

## Alpha Bunny

Alpha was API-only, so unavailable provider weights are not represented as owned or extracted. Its **observable behavior has two distinct roles**.

The preserved live-capture audit recorded 34 captures, zero integrity failures, one unusable capture, and 33 eligible captures. Those eligible captures were split into **27 training cases and 6 frozen holdout cases**. Training axes include code, debugging, persona, self-correction, and tool behavior.

Therefore Alpha is not evaluation-only: the 27 eligible cases are legitimate behavioral training lineage. The six designated holdout IDs are evaluation-only and must never be introduced into training.

## Space Bunny

Space Bunny is the locally trained model lineage built from the larger Alpha-derived corpus plus eligible live Alpha behavioral material. A recorded staged corpus contained 839 rows: 817 rows from the larger `alpha_corpus_926` source and 22 live-capture rows, with corpus validation reporting zero errors and zero warnings.

Space Bunny's adapter, merged Qwen3-4B model, later GGUFs, evaluation tooling, persona datasets, and self-correction work remain external artifacts. Rabbit Foundry records their lineage instead of copying multi-gigabyte weights into Git.

## Rabbit Foundry

Rabbit is the native research lineage: random initialization, no teacher-model answers for the native GitHub experiment, prediction/repair/hidden-diff skills, leak-resistant source splits, frozen evaluation, equal-compute fixed-vs-adaptive experiments, and Greenlight promotion.

Rabbit may also run **separately labeled Bunny-lineage experiments** using eligible Alpha/Space Bunny material. Such runs must not be mislabeled as the teacher-free native experiment.

## Non-negotiable boundary

Training and holdout remain separate across every migration. In particular these Alpha holdout IDs stay frozen:

- `011-partial-failure`
- `014-instruction-conflict`
- `016-guess-discipline`
- `028-ambiguous-request`
- `031-test-first-request`
- `034-anti-sycophancy`

Space Bunny checkpoints are Space Bunny lineage, not Rabbit-native checkpoints. Alpha provider weights remain unavailable and are not required for behavioral preservation.

Machine-readable inventory: `configs/bunny_lineage.json`.
