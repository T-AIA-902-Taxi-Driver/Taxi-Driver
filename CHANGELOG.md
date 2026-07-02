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
- Agents RL tabulaires : `BaseAgent` (contrat terminated-vs-truncated), stratégies
  d'exploration enfichables (ε-greedy exp/linéaire, Boltzmann, UCB), `TabularAgent`
  (argmax greedy déterministe, save/load npz), Q-Learning, SARSA, Expected SARSA,
  Double Q-Learning, Monte Carlo first-visit, BruteForce ; factory `create_agent()`
- `TaxiEnvWrapper` : encapsulation Gymnasium Taxi-v3 (render ansi, `n_states`/`n_actions`,
  `decode_state()`, hook de reward shaping avec `info["raw_reward"]`, seeding premier-reset,
  graine explicite par épisode pour l'évaluation) + factory `create_env()`

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
