# Changelog

Toutes les modifications notables apportées à ce projet sont documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/), et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [1.0.0] - 2026-07-03

### Added

- Campagne expérimentale complète E0-E7 : 822 runs, 10 seeds par configuration,
  R* = 8.05 établi par value iteration ; 5 algorithmes sur 6 atteignent la
  politique optimale (8.05, 12.95 pas, 100 % succès) — Monte Carlo censuré 10/10
- Résultats versionnés : agrégats par bloc, tests statistiques H1-H9
  (Welch/Mann-Whitney, Holm, tailles d'effet), figures F1-F12, modèles finaux
- `configs/optimized.yaml` définitif : α=0.30, γ=0.95 (départage des 15 configs
  optimales ex æquo par vitesse de convergence, seuil en 980 épisodes)
- Rapport scientifique final (`docs/RAPPORT.md`, français) et support de
  soutenance (`docs/SLIDES.md`, Marp)
- Scripts d'analyse : `run_stats.py` (hypothèses), `make_figures.py` (figures
  régénérables depuis les données brutes), pack de chiffres reproductible

### Changed

- README aligné sur l'implémentation réelle (commandes vérifiées), corrections
  des documents de cadrage (14 400 états multi-passagers, seuil 90 %, early
  stopping opt-in)

### Fixed

- Seuil de convergence référencé sur l'optimum du jeu de sondes (7.77) et non
  de l'évaluation (8.05) — évite la censure erronée de runs optimaux

### Added (développement, PRs #59-#67)

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
- Agents RL tabulaires : `BaseAgent` (contrat terminated-vs-truncated), stratégies
  d'exploration enfichables (ε-greedy exp/linéaire, Boltzmann, UCB), `TabularAgent`
  (argmax greedy déterministe, save/load npz), Q-Learning, SARSA, Expected SARSA,
  Double Q-Learning, Monte Carlo first-visit, BruteForce ; factory `create_agent()`
- `TaxiEnvWrapper` : encapsulation Gymnasium Taxi-v3 (render ansi, `n_states`/`n_actions`,
  `decode_state()`, hook de reward shaping avec `info["raw_reward"]`, seeding premier-reset,
  graine explicite par épisode pour l'évaluation) + factory `create_env()`

- Agent DQN (extension deep RL) : QNetwork (MLP one-hot 2×couches cachées),
  ReplayBuffer en anneau numpy préalloué, cibles Double-DQN par défaut, mise à jour
  douce du réseau cible (Polyak), Huber + clipping de gradient, détection
  automatique du device (CUDA), sauvegarde .pt avec optimiseur et métadonnées
- Environnement multi-passagers (bonus du sujet) : `MultiPassengerTaxiEnv`
  à 14 400 états (25×6²×4², corrige le décompte 10 000 de CADRAGE.md qui omettait
  le statut « livré »), capacité 2, règles pickup/dropoff déterministes
  (plus petit indice), dépose hors destination interdite, TimeLimit 500 ;
  analyse d'ordre de route (`route_analysis.py`) ; protocole `EnvWrapper` partagé
- Pipeline d'entraînement : `Trainer` générique (contrat terminated-vs-truncated,
  sondes greedy périodiques), callbacks (Logging, EarlyStopping opt-in, Checkpoint,
  TimeBudget à horloge injectable), `Evaluator` (politique greedy stricte, seeds
  d'évaluation explicites par épisode, affichage d'épisodes aléatoires), métriques
  (moyennes/écarts-types/médianes, IC 95 %, percentiles, temps moyen par partie)
- CLI `taxi-driver` : sous-commandes train/eval/play (benchmark/compare à venir),
  mode utilisateur interactif (saisie des hyperparamètres et des nombres d'épisodes
  au lancement), mode time-limited (config optimisée, budget temps 90/10),
  auto-détection de l'algorithme depuis les métadonnées du modèle ; smoke test CI

- Benchmarking scientifique : runner d'expériences idempotent (hash de config,
  reprise après crash), module statistique (Shapiro→Welch/Mann-Whitney, Holm,
  Hedges g, Cliff δ), reward shaping (potentiel/naïf/pénalité), R* par value
  iteration, sous-commandes `benchmark`/`compare`, campagne E0-E7
  (`scripts/run_campaign.py`), protocole expérimental (`docs/PROTOCOLE.md`)
- Visualisation : générateurs de figures F1-F12 pilotés par les données
  (`scripts/make_figures.py`), GIF d'épisode, instrumentation max-Q des sondes

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
