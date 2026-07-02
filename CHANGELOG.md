# Changelog

Toutes les modifications notables apportées à ce projet sont documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/), et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [Unreleased]

### Added

- Configuration Poetry complète (`pyproject.toml`) : dépendances (gymnasium <1.3 pour
  Taxi-v3, numpy, torch, matplotlib, seaborn, pandas, scipy), groupe dev (pytest, ruff,
  black, mypy), extra optionnel `trackmania`, script console `taxi-driver`
- Pipeline CI GitHub Actions (ruff, black, mypy, pytest + couverture ≥ 70 %)
- Hooks pre-commit (black, ruff, hygiène de fichiers)
- Arborescence du package `src/` (agents, environments, training, evaluation,
  benchmarking, visualization, cli, utils) — remplace le dossier `src/environnements`
  (faute de frappe)
- Dataclass `Config` : chargement YAML, surcharge CLI, validation explicite,
  hash de configuration pour le benchmarking, décroissance d'epsilon paramétrée
  en fraction d'horizon (`decay_frac`)
- Utilitaires de reproductibilité (`src/utils/seeding.py`) : flux RNG indépendants,
  listes de seeds d'évaluation et de sondes disjointes
- Configuration par défaut commentée (`configs/default.yaml`)
- Extension TrackMania (bonus deep RL) : `TMEnvProtocol` + `TrackManiaEnvWrapper`
  (observations LIDAR aplaties/normalisées Box(83,), actions continues bornées),
  script SAC autonome `scripts/train_trackmania.py`, guide `docs/TRACKMANIA.md` —
  code livrable testé sur environnement factice (le jeu n'est pas exécutable en CI)

### Changed

- `.gitignore` : les modèles finaux (`models/final/`) et les résultats agrégés/figures
  (`results/aggregated/`, `results/figures/`) sont désormais versionnés (livrables)

## [0.1.0]

### Added

- Structure initiale du répertoire (`src/`, `tests/`, `configs/`, `docs/`, `models/`, `results/`)
- Documentation de cadrage (`docs/CADRAGE.md`)
- Architecture logicielle avec diagrammes Mermaid (`docs/ARCHITECTURE.md`)
- Backlog produit avec priorisation WSJF (`docs/BACKLOG.md`)
- README professionnel (`README.md`)
- Guide de contribution (`CONTRIBUTING.md`)

---

> Ce changelog sera mis à jour à chaque PR mergée sur `dev` ou `main`.
