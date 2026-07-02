# Protocole expérimental — Taxi Driver (T-AIA-902)

> Ce document fixe le protocole scientifique du projet : conventions, hypothèses,
> matrice d'expériences, méthodologie statistique et gestion des données. Toute
> figure du rapport doit être régénérable depuis `results/raw/` en appliquant
> ce protocole. Implémentation : `src/benchmarking/` et `scripts/run_campaign.py`.

## 1. Conventions globales

| Élément | Valeur | Justification |
|---|---|---|
| Seeds d'entraînement | 42..51 (n=10 par configuration) | CADRAGE §6.5 |
| Évaluation finale | 100 épisodes, seeds figés `10042+i` identiques pour tous les agents | comparaisons appariées équitables |
| Sondes de convergence | 30 épisodes greedy, seeds `20042+i` (disjoints de l'éval), toutes les 100 épisodes d'entraînement | mesurer la vraie politique sans contaminer l'évaluation ; le reward d'entraînement est confondu avec ε |
| Politique de test | greedy stricte (ε=0), argmax déterministe (plus petit indice) | comparabilité |
| Récompense rapportée | toujours la récompense **native** (`info["raw_reward"]`), même si le shaping est actif | le shaping ne doit pas gonfler les métriques |
| Budget tabulaire / DQN / multi | 15 000 / 5 000 / 150 000 épisodes | dimensionné sur la convergence observée |
| Early stopping | **désactivé** pour toutes les expériences (réservé au mode time-limited du CLI) | arrêter certains algos plus tôt biaise toutes les métriques de convergence |
| Décroissance de ε | paramétrée en **fraction d'horizon** : ε atteint ε_min à 60 % de N | découple la forme de la décroissance du nombre d'épisodes |
| Config de référence | α=0.15, γ=0.99, ε 1.0→0.01 (exp, frac 0.6) | point central des analyses de sensibilité |
| R\* (étalon) | value iteration sur le modèle exact de Taxi-v3 (`env.P`), évalué sur les mêmes seeds | uniquement un instrument de mesure ; les agents restent model-free |
| Seuil de convergence | reward sonde ≥ 0,9·R\* maintenu 3 checkpoints consécutifs ; censuré à N sinon | CADRAGE §6.5 (le 95 % de §6.2 est reporté en annexe) |
| Unité statistique | le **run** (une seed), jamais l'épisode | éviter la pseudo-réplication |
| Chronométrage | uniquement le bloc E7 (séquentiel, machine au repos) ; médiane rapportée en plus de la moyenne (variance WSL2) | mesures de temps fiables |

## 2. Hypothèses testées

| ID | Hypothèse (prédiction) | Métriques | Test |
|---|---|---|---|
| H1 | γ=0.99 converge plus lentement que γ=0.9 mais donne une meilleure politique finale (contre-prédiction discutée : épisodes courts ⇒ γ=0.9 peut suffire) | épisodes-au-seuil, reward final | Welch/M-W + Holm |
| H2 | Q-Learning atteint le seuil plus tôt ; SARSA est plus stable pendant l'exploration | épisodes-au-seuil ; σ du reward d'entraînement | Welch/M-W + Holm |
| H3 | Les résultats sont plus sensibles aux hyperparamètres qu'au choix d'algorithme | étendues intra vs inter-algos, η² descriptif | effect sizes + bootstrap |
| H4 | Décroissance linéaire ≈ exponentielle en final ; Boltzmann converge au moins aussi vite ; UCB couvre mieux (s,a) au début mais sensible à c | épisodes-au-seuil, first-success, couverture (s,a) | paires vs référence + Holm |
| H5 | Le shaping **basé potentiel** (Ng et al. 1999) accélère sans dégrader le reward natif final ; le bonus naïf dégrade ; la sur-pénalité de step est neutre à négative | épisodes-au-seuil et reward final **natifs** | paires vs native + Holm ; CI de la différence pour la non-infériorité |
| H6 | Le tabulaire domine le DQN en efficacité échantillon (≥5×), temps (≥50×), mémoire, latence, à performance finale comparable | courbes vs env-steps cumulés, temps E7, mémoire | Welch/M-W + ratios descriptifs |
| H7 | Double Q-Learning réduit le biais de surestimation (E[max Q] − retour réel), converge un peu plus lentement | gap de surestimation, épisodes-au-seuil | Welch/M-W + Holm |
| H8 | Monte Carlo est plus lent et plus variable que les méthodes TD (runs censurés possibles) | épisodes-au-seuil censuré, taux de convergence | Mann-Whitney (rang pire pour censurés) |
| H9 (exploratoire) | Multi-passagers (14 400 états) : croissance sur-linéaire du coût de convergence ; le classement QL>SARSA persiste | épisodes-au-seuil relatif | descriptif |

## 3. Matrice d'expériences (~840 runs, ≈4 h 30 – 5 h 30 sur 6 cœurs)

| Bloc | Contenu | Runs | Alimente |
|---|---|---|---|
| E0 | BruteForce, caps 200 et 2000 steps, 1000 épisodes d'éval | 20 | T1, ratio ~20×, artefact de troncature |
| E1a | Grid QL : α∈{0.05,0.1,0.15,0.2,0.3} × γ∈{0.9,0.95,0.99,0.999} | 200 | H1, H3, F5/F6, `optimized.yaml` |
| E1b | Grille croisée : {SARSA, ExpSARSA, DoubleQL, MC} × 3α × 3γ | 360 | H3 |
| E2 | Face-à-face : 6 tabulaires + DQN à la config de référence | 70 | T1, H2/H6/H7/H8, F1-F4, F9 |
| E3 | Exploration : ε-exp, ε-lin, Boltzmann, UCB | 40 | H4, F7 |
| E4 | Shaping : native, potentiel, naïf, step-penalty | 40 | H5, F8 |
| E5 | Sensibilité DQN : lr × hidden (5 seeds) | 30 | annexe DQN |
| E6 | Multi-passagers : QL, SARSA × 150 000 épisodes | 20 | H9, F12 |
| E7 | **Chronométrage séquentiel** : 6 algos × 10 seeds + DQN cuda/cpu × 3 | 66 | colonnes temps/mémoire de T1, ablation GPU |

Parallélisation : E0–E6 via `multiprocessing.Pool(6)` (la parallélisation n'affecte
pas reward/steps/épisodes ; « pas de parallélisation » du CADRAGE ne concerne que
les mesures de temps → E7). Ordre de coupe si dépassement : E5 → E6-SARSA → E1b →
seeds E3/E4 10→5. **E2 n'est jamais coupé.**

## 4. Équité des comparaisons

Trois axes rapportés, aucun algorithme n'est arrêté plus tôt qu'un autre :
1. **épisodes égaux** (tabulaires entre eux : 15 000) ;
2. **env-steps cumulés égaux** (tabulaire vs DQN, axe principal de H6) ;
3. **wall-clock égal** (annexe, extrait de E7).

La politique évaluée est celle de **fin d'entraînement** (pas de « meilleur
checkpoint », qui favoriserait les algorithmes bruités). BruteForce : la moyenne
(~−700) est un artefact de troncature à 200 steps — médiane, taux de succès et
variante 2000 steps sont rapportés.

## 5. Méthodologie statistique

1. Échantillons = agrégats par run (n=10).
2. Shapiro-Wilk (α=0.05) sur chaque groupe → deux normaux : **Welch** ; sinon
   **Mann-Whitney U**. À n=10 la puissance de Shapiro-Wilk est faible : si les
   deux tests divergent sur la significativité, les deux p sont rapportés.
3. Comparaisons multiples : **Holm-Bonferroni par famille d'hypothèses** ;
   p brut ET p ajusté rapportés.
4. Tailles d'effet : **Hedges g** (Welch) et **Cliff's δ** (M-W).
5. IC 95 % de Student : μ ± t₀.₉₇₅,ₙ₋₁·σ/√n.
6. Format : « QL : 7.92 ± 0.31 (IC 95 % [7.70 ; 8.14], n=10) vs SARSA : … ;
   Welch, p=0.008 (p_Holm=0.016), g=1.28 (effet fort) ».
7. Non-significativité : rapporter l'IC de la différence, jamais « pas de
   différence » (absence de preuve ≠ preuve d'absence).
8. Métriques censurées (épisodes-au-seuil non atteint) : M-W avec rang pire +
   taux de convergence rapporté.

## 6. Gestion des données

```
results/
├── r_star.json                      # étalon optimal (value iteration)
├── raw/<bloc>/<algo>__<hash8>__s<seed>__<horodatage>/
│   ├── config.json                  # config résolue + versions des libs
│   ├── train_episodes.csv           # 1 ligne / épisode d'entraînement
│   ├── probes.csv                   # 1 ligne / checkpoint de sonde
│   ├── eval_episodes.csv            # 1 ligne / épisode d'évaluation (seed figé)
│   └── summary.json                 # agrégats
├── aggregated/<bloc>__runs.csv      # 1 ligne / run (consommé par les figures)
└── figures/                         # PNG 300 dpi versionnés
```

`hash8` = SHA-256 tronqué de la configuration canonique **hors seed** : une même
configuration garde le même hash à travers les seeds. Les runs sont idempotents
par `(bloc, hash, seed)` : relancer la campagne reprend après un crash.

## 7. Spécifications machine (T5 du rapport)

CPU 6 cœurs, 27 Go RAM, GPU NVIDIA RTX 2060 6 Go (torch 2.5.1+cu124), WSL2,
Python 3.10.12, gymnasium 1.2.3 (Taxi-v3 ; retiré de gymnasium ≥1.3, remplacé
par Taxi-v4 identique aux paramètres par défaut — d'où le pin `<1.3`).
Versions exactes loguées dans chaque `config.json`. Les temps sous WSL2 ont une
variance élevée : médianes rapportées en complément des moyennes.
