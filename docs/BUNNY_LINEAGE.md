# Unified Bunny Lineage

Rabbit Foundry is the canonical research home for the Bunny projects, but **merge does not mean erase provenance**.

## Alpha Bunny

Alpha contributes a frozen behavioral benchmark and preserved observable trajectories. Alpha was API-only, so unavailable weights are not represented as owned or extracted. `ALPHA-FINGERPRINT-v1` stays frozen and must remain separate from training examples used to optimize Rabbit.

## Space Bunny

Space Bunny contributes locally preserved model/training lineage, GGUF deployment references, and behavior baselines. These artifacts remain external to Git because they are large model files. Rabbit Foundry records their paths and provenance rather than copying weights into the repository.

## Rabbit Foundry

Rabbit remains the native research lineage: random initialization, no teacher-model answers, GitHub-derived prediction/repair/hidden-diff skills, leak-resistant source splits, frozen evaluation, equal-compute fixed-vs-adaptive experiments, and Greenlight promotion.

## Rule

A result may compare or learn from eligible Bunny-derived material, but every report must retain its lineage. Alpha benchmark cases designated as holdout must never become Rabbit training examples. Space Bunny checkpoints must never be labeled Rabbit-native checkpoints.

Machine-readable inventory: `configs/bunny_lineage.json`.
