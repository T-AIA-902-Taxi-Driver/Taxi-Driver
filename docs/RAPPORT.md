# Rapport scientifique — Taxi Driver (T-AIA-902)

> **Module** : T-AIA-902 — Intelligence Artificielle / Apprentissage par Renforcement
> **Organisation GitHub** : T-AIA-902-Taxi-Driver
> **Environnement** : Taxi-v3 (Gymnasium)
>
> Toute figure et tout tableau de ce rapport sont régénérables depuis
> `results/raw/` en appliquant le protocole décrit en section 4 (voir
> `docs/PROTOCOLE.md`, `scripts/run_campaign.py`, `scripts/run_stats.py` et
> `scripts/make_figures.py`).

---

## Table des matières

1. [Introduction](#1-introduction)
2. [État de l'art](#2-état-de-lart)
3. [Formalisation du problème](#3-formalisation-du-problème)
4. [Méthodologie](#4-méthodologie)
5. [Résultats expérimentaux](#5-résultats-expérimentaux)
6. [Discussion](#6-discussion)
7. [Limites et perspectives](#7-limites-et-perspectives)
8. [Conclusion](#8-conclusion)
9. [Références bibliographiques](#9-références-bibliographiques)

---

## 1. Introduction

### 1.1 Contexte

Ce rapport présente les travaux réalisés dans le cadre du module T-AIA-902 du
cursus Epitech, consacré à l'apprentissage par renforcement (*Reinforcement
Learning*, RL). Le sujet impose de résoudre l'environnement **Taxi-v3** de la
bibliothèque Gymnasium (Towers et al., 2024) à l'aide d'algorithmes
**model-free** et **épisodiques** : un taxi évolue sur une grille 5×5, doit
localiser un passager, le prendre en charge et le déposer à sa destination, le
tout en minimisant le nombre de déplacements et les manœuvres illégales.

Taxi-v3 est un problème jouet au sens noble du terme : suffisamment petit pour
que la politique optimale soit calculable exactement, suffisamment structuré
(sous-objectifs successifs, pénalités asymétriques, troncature temporelle) pour
faire apparaître les phénomènes qui traversent tout le champ du RL — dilemme
exploration/exploitation, biais de surestimation, sensibilité aux
hyperparamètres, artefacts de mesure. Chaque comportement observé peut ainsi
être confronté à la théorie.

### 1.2 Problématique

La consigne du module est explicite : *« un bon projet IA, c'est un bon
rapport, pas juste un bon code »*. La problématique n'est donc pas seulement
d'atteindre la performance optimale sur Taxi-v3 — objectif atteint de longue
date par le Q-Learning tabulaire — mais de **comprendre et d'expliquer les
comportements observés** : pourquoi tel algorithme converge-t-il plus vite
qu'un autre alors que tous atteignent la même politique finale ? Les écarts
mesurés sont-ils significatifs ou relèvent-ils du bruit d'échantillonnage ?
Le choix de l'algorithme importe-t-il davantage que celui des
hyperparamètres ? Que coûte le deep RL sur un problème que la tabulation
résout exactement ?

### 1.3 Objectifs

Les objectifs opérationnels, dérivés du cahier des charges et du document de
cadrage (`docs/CADRAGE.md`), sont les suivants :

1. **Résoudre Taxi-v3** avec une performance finale comparable à la politique
   optimale (8 à 13 pas par épisode en moyenne selon le sujet), et quantifier
   le gain par rapport à une base aléatoire (*brute force*).
2. **Implémenter et comparer plusieurs algorithmes** : Q-Learning, SARSA,
   Expected SARSA, Double Q-Learning, Monte Carlo first-visit, et un DQN en
   extension deep RL.
3. **Tester des hypothèses formulées a priori** (H1 à H9, section 4.3) selon
   une démarche scientifique contrôlée : seeds multiples, tests statistiques,
   correction des comparaisons multiples, tailles d'effet.
4. **Explorer trois extensions** : stratégies d'exploration alternatives
   (Boltzmann, UCB), *reward shaping* basé potentiel, et un environnement
   multi-passagers à 14 400 états conçu pour ce projet.

### 1.4 Annonce du plan

La section 2 situe les algorithmes retenus dans l'état de l'art du RL
tabulaire et profond. La section 3 formalise Taxi-v3 comme processus de
décision markovien et présente l'extension multi-passagers. La section 4
détaille le protocole expérimental — conventions, hypothèses, matrice
d'expériences, méthodologie statistique — dont la section 5 présente les
résultats, de la base aléatoire à la politique optimale puis aux extensions.
La section 6 en propose une lecture critique transversale, la section 7
recense les limites et les pistes ouvertes, et la section 8 conclut.

---

## 2. État de l'art

### 2.1 RL tabulaire et RL profond

L'apprentissage par renforcement (Sutton & Barto, 2018) étudie des agents qui
apprennent une politique de décision par interaction : observer un état,
choisir une action, recevoir une récompense, et maximiser l'espérance du
retour cumulé actualisé. Deux grandes familles de méthodes *model-free* — qui
n'apprennent pas le modèle de transition — coexistent. Les méthodes
**tabulaires** représentent la fonction de valeur action-état Q(s, a) par un
tableau explicite de taille |S|×|A| : garanties de convergence fortes, mais
espace d'états nécessairement fini et visitable exhaustivement. Les méthodes
**profondes** (deep RL) remplacent la table par un approximateur — typiquement
un réseau de neurones — pour traiter des espaces immenses ou continus, au prix
des garanties de convergence et d'une instabilité d'entraînement bien
documentée. Taxi-v3, avec ses 500 états, relève naturellement de la première
famille ; nous y appliquons néanmoins un DQN afin de mesurer précisément le
coût de cette sur-machinerie (hypothèse H6).

### 2.2 Méthodes par différence temporelle

Les méthodes TD (*temporal difference*) mettent à jour Q(s, a) après chaque
transition, en s'appuyant sur l'estimation courante du successeur
(*bootstrapping*).

**Q-Learning** (Watkins, 1989) est la méthode TD *off-policy* de référence :

```
Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

La cible utilise la meilleure action suivante *indépendamment* de l'action
réellement exécutée par la politique d'exploration : l'algorithme apprend
directement la fonction de valeur de la politique gloutonne optimale. Watkins
et Dayan (1992) prouvent la convergence vers Q\* sous les conditions de
Robbins-Monro : chaque paire (s, a) visitée infiniment souvent, et un taux
d'apprentissage vérifiant Σα = ∞ et Σα² < ∞.

**SARSA** (Rummery & Niranjan, 1994) est son pendant *on-policy* :

```
Q(s,a) ← Q(s,a) + α [ r + γ · Q(s',a') − Q(s,a) ]
```

où a′ est l'action **réellement exécutée** au pas suivant. SARSA apprend la
valeur de la politique effectivement suivie, exploration comprise : tant que
ε > 0, il intègre le risque des actions exploratoires, ce qui produit
classiquement des politiques plus prudentes pendant l'apprentissage.

**Expected SARSA** (van Seijen et al., 2009) remplace l'échantillon unique a′
par l'espérance sous la politique :

```
Q(s,a) ← Q(s,a) + α [ r + γ · Σ_a' π(a'|s')·Q(s',a') − Q(s,a) ]
```

Cette espérance supprime la variance introduite par le tirage de a′ tout en
restant on-policy ; van Seijen et al. démontrent qu'Expected SARSA domine SARSA
en variance à politique égale.

**Double Q-Learning** (van Hasselt, 2010) attaque le **biais de
surestimation** du Q-Learning : l'opérateur max appliqué à des estimations
bruitées surestime systématiquement la valeur du successeur
(E[max Q̂] ≥ max E[Q̂]). Deux tables Q_A et Q_B sont entretenues ; à chaque
transition une pièce équitable choisit la table mise à jour, l'argmax est pris
dans celle-ci mais **évalué dans l'autre** :

```
Q_A(s,a) ← Q_A(s,a) + α [ r + γ · Q_B(s', argmax_a' Q_A(s',a')) − Q_A(s,a) ]
```

La décorrélation entre sélection et évaluation supprime le biais, au prix
d'une vitesse d'apprentissage réduite (chaque table ne voit que la moitié des
transitions).

### 2.3 Méthodes Monte Carlo

Les méthodes **Monte Carlo** renoncent au bootstrapping : elles attendent la
fin de l'épisode pour mettre à jour Q(s, a) vers le retour réellement observé
G. La variante *first-visit* ne crédite que la première occurrence de chaque
paire (s, a) dans la trajectoire. L'estimation est non biaisée (en l'absence
de troncature) mais de variance élevée : le retour agrège le bruit de toutes
les décisions ultérieures. Sur des environnements à limite de temps, les
épisodes tronqués injectent de surcroît un retour biaisé — une limitation
structurelle discutée en section 5.3.

### 2.4 Deep Q-Network

Le **DQN** (Mnih et al., 2015) a rendu l'apprentissage de fonctions Q par
réseaux de neurones stable grâce à deux mécanismes : le **replay buffer**, qui
mémorise les transitions et échantillonne des mini-lots décorrélés (brisant la
corrélation temporelle des données), et le **réseau cible**, copie retardée du
réseau principal qui fixe la cible de régression le temps de quelques mises à
jour (brisant la poursuite d'une cible mouvante). Notre implémentation ajoute
les raffinements devenus standards : mises à jour douces du réseau cible
(moyennage de Polyak), perte de Huber et écrêtage du gradient — utiles face
aux récompenses aberrantes de −10 de Taxi.

### 2.5 Stratégies d'exploration

Le dilemme exploration/exploitation admet plusieurs réponses classiques.
**ε-greedy** joue l'action gloutonne avec probabilité 1−ε et une action
uniforme sinon, ε décroissant au fil de l'entraînement (décroissance
exponentielle ou linéaire). L'exploration de **Boltzmann** (softmax) tire les
actions proportionnellement à exp(Q(s,a)/τ) : elle gradue l'exploration selon
les valeurs estimées au lieu d'explorer uniformément. **UCB** (*Upper
Confidence Bound*, Auer, Cesa-Bianchi & Fischer, 2002), issu de la théorie des
bandits, sélectionne argmax_a [Q(s,a) + c·√(ln t / N(s,a))] : le bonus
d'incertitude force la visite des actions peu essayées, produisant une
exploration dirigée plutôt qu'aléatoire.

### 2.6 Reward shaping basé potentiel

Modifier la récompense pour guider l'apprentissage est tentant mais dangereux :
un bonus mal conçu peut changer la politique optimale (l'agent apprend à
collecter le bonus au lieu de résoudre la tâche). Ng, Harada et Russell (1999)
caractérisent la famille de modifications sûres : les récompenses de la forme

```
F(s, s') = γ·Φ(s') − Φ(s)
```

pour toute fonction potentiel Φ : S → ℝ **préservent la politique optimale**
(théorème d'invariance) — la somme télescopique de F s'annulant sur tout
cycle, aucune boucle ne peut accumuler de récompense fictive. Nous testons ce
théorème en opposant un shaping basé potentiel à un bonus de distance naïf,
structurellement proche mais non potentiel (hypothèse H5).

---

## 3. Formalisation du problème

### 3.1 Taxi-v3 comme processus de décision markovien

L'environnement Taxi-v3 se formalise comme un MDP fini (S, A, P, R, γ) :

- **S** : 500 états, produit cartésien de la position du taxi (5 lignes ×
  5 colonnes = 25 cases), du statut du passager (aux quatre repères R, G, Y, B
  ou dans le taxi, soit 5 valeurs) et de la destination (4 repères), soit
  **25 × 5 × 4 = 500**. L'état est encodé en un entier unique
  (base mixte `((row·5 + col)·5 + passager)·4 + destination`).
- **A** : 6 actions discrètes — Sud (0), Nord (1), Est (2), Ouest (3),
  Pickup (4), Dropoff (5).
- **P** : transitions **déterministes** dans la configuration par défaut ;
  les murs et les bords de la grille bloquent le déplacement (la position est
  inchangée mais le pas est décompté).
- **R** : +20 pour une dépose du passager à sa destination (fin d'épisode),
  −1 pour chaque pas, −10 pour un Pickup ou un Dropoff illégal.
- **γ** : facteur d'actualisation, hyperparamètre étudié (0,9 à 0,999).

Un épisode se termine à la livraison du passager (*terminated*) ou est
**tronqué à 200 pas** par le `TimeLimit` de Gymnasium (*truncated*). Cette
distinction, anodine en apparence, est centrale : un épisode tronqué doit
continuer à *bootstrapper* depuis l'état successeur (la tâche n'est pas
terminée, seule la mesure l'est), et la troncature crée un plancher artificiel
sur la récompense des mauvaises politiques (section 5.1). La récompense
minimale d'un épisode tronqué sans manœuvre illégale est de −200 ; les
pénalités de −10 peuvent l'abaisser bien davantage.

Le problème satisfait la propriété de Markov par construction : l'entier
d'état encode tout ce qui détermine la dynamique. La politique optimale résout
l'épisode en une vingtaine de pas au plus, pour une récompense optimale
moyenne mesurée à **R\* = 8,05** sur notre jeu d'évaluation (section 4.2).

### 3.2 Extension multi-passagers : 14 400 états

Le bonus du sujet demande un taxi qui prend en charge **2 passagers**, chacun
avec ses propres origine et destination parmi les 4 repères, l'objectif étant
d'**optimiser la tournée**. Nous avons implémenté cet environnement
(`src/environments/multi_passenger_env.py`) sur la grille exacte de Taxi-v3,
avec les mêmes actions et le même barème (−1 par pas, −10 pour une manœuvre
illégale, +20 par passager livré).

**Espace d'états.** Le document de cadrage estimait 25 × 5² × 4² = 10 000
états, par analogie avec les 5 statuts de Taxi-v3. Cette estimation était
**erronée** : dans Taxi-v3, la livraison termine l'épisode, si bien que l'état
« livré » n'a jamais besoin d'être représenté. Avec deux passagers, livrer le
premier ne termine pas l'épisode : « livré » doit être un **sixième statut**,
distinct des quatre repères et de « dans le taxi ». L'espace d'états correct
est donc :

```
25 (cases taxi) × 6² (statuts des 2 passagers) × 4² (destinations) = 14 400 états
```

soit un facteur ×28,8 par rapport à Taxi-v3 — l'encodage en base mixte est
`((((row·5 + col)·6 + s₀)·6 + s₁)·4 + d₀)·4 + d₁`.

**Règles de conception.** Trois choix méritent justification :

1. **Capacité 2** : les deux passagers peuvent occuper le taxi simultanément.
   C'est la condition d'une véritable optimisation de tournée — avec une
   capacité de 1, l'ordre des courses serait la seule liberté.
2. **Règle du plus petit indice** : lorsque les deux passagers sont éligibles
   à un Pickup (même repère) ou à un Dropoff (même destination), le passager
   d'indice le plus faible est servi. Cette règle rend la transition
   déterministe et l'environnement markovien sans état caché.
3. **Dépose hors destination interdite** : déposer un passager embarqué
   ailleurs qu'à sa destination est illégal (−10, le passager reste à bord),
   là où Taxi-v3 re-place le passager sur le repère pour −1. Cette divergence
   délibérée évite les allers-retours de dépose/reprise qui gonfleraient
   l'espace d'états utile sans intérêt pour l'optimisation de tournée.

Les états initiaux tirent la case du taxi uniformément, deux repères de départ
**distincts**, et pour chaque passager une destination différente de son
origine (les deux destinations peuvent coïncider). L'épisode se termine quand
les deux passagers sont livrés ; la troncature est portée à **500 pas** (les
tournées optimales sont environ deux fois plus longues que dans Taxi-v3 et
l'errance de début d'entraînement est plus coûteuse).

---

## 4. Méthodologie

Cette section reprend fidèlement le protocole expérimental figé avant la
campagne (`docs/PROTOCOLE.md`). Le principe directeur : **toute comparaison
doit être équitable, appariée et reproductible**, et toute affirmation du
rapport doit être adossée à un test statistique ou explicitement qualifiée de
descriptive.

### 4.1 Conventions globales

| Élément | Valeur | Justification |
|---|---|---|
| Seeds d'entraînement | 42 à 51 (**n = 10 runs** par configuration) | réplication minimale pour des tests à n=10 |
| Évaluation finale | 100 épisodes, seeds figés `10042+i`, identiques pour tous les agents | comparaisons appariées équitables |
| Sondes de convergence | 30 épisodes greedy, seeds `20042+i` (disjoints de l'évaluation), toutes les 100 épisodes d'entraînement | mesurer la vraie politique sans contaminer l'évaluation finale |
| Politique de test | greedy stricte (ε = 0), argmax déterministe (plus petit indice) | comparabilité inter-algorithmes |
| Récompense rapportée | toujours la récompense **native** (`info["raw_reward"]`), même sous shaping | le shaping ne doit jamais gonfler les métriques |
| Budget d'entraînement | 15 000 épisodes (tabulaire) / 5 000 (DQN) / 150 000 (multi-passagers) | dimensionné sur la convergence observée en pré-tests |
| Early stopping | **désactivé** pour toutes les expériences | arrêter certains algorithmes plus tôt biaiserait les métriques de convergence |
| Décroissance de ε | paramétrée en **fraction d'horizon** : ε atteint ε_min à 60 % de N | découple la forme de la décroissance du budget d'épisodes |
| Configuration de référence | α = 0,15, γ = 0,99, ε : 1,0 → 0,01 (exponentielle, fraction 0,6) | point central des analyses de sensibilité |
| Unité statistique | le **run** (une seed), jamais l'épisode | éviter la pseudo-réplication |
| Chronométrage | uniquement le bloc E7, séquentiel, machine au repos | mesures de temps fiables sous WSL2 |

Deux conventions méritent d'être soulignées. D'abord, la **récompense
d'entraînement est une mesure confondue** : elle mélange la qualité de la
politique et le niveau courant d'exploration (un agent excellent avec ε = 0,5
affiche un mauvais reward d'entraînement). Les **sondes greedy** périodiques
mesurent la vraie politique apprise, sur des seeds disjoints de l'évaluation
finale pour éviter toute fuite. Ensuite, la politique évaluée est celle de
**fin d'entraînement**, jamais le « meilleur checkpoint » — qui favoriserait
mécaniquement les algorithmes bruités (plus un processus fluctue, plus son
maximum observé est élevé).

### 4.2 L'étalon R\* : la politique optimale comme instrument de mesure

Taxi-v3 expose son modèle de transition exact (`env.P`). Nous en tirons la
politique optimale par **value iteration** (`src/benchmarking/value_iteration.py`)
et l'évaluons sur les mêmes seeds que les agents. Cet étalon sert
**uniquement d'instrument de mesure** — les agents entraînés restent
strictement model-free comme l'exige le sujet.

- **R\* = 8,05** sur les 100 épisodes d'évaluation (seeds `10042+i`) ;
- **R\* = 7,77** sur les 30 épisodes de sonde (seeds `20042+i`).

L'écart entre les deux valeurs illustre au passage la variance
d'échantillonnage des états initiaux. Le **seuil de convergence** est défini
comme : reward de sonde ≥ 0,9·R\*_sondes = **6,99**, maintenu **3 checkpoints
consécutifs** (l'exigence de maintien filtre les franchissements chanceux).
Les runs n'atteignant jamais le seuil sont **censurés** à N épisodes et
traités au rang le pire dans les tests (section 4.5).

### 4.3 Hypothèses testées

Chaque hypothèse a été formulée, avec sa prédiction, **avant** la campagne.

| ID | Hypothèse (prédiction) | Métriques principales |
|---|---|---|
| H1 | γ = 0,99 converge plus lentement que γ = 0,9 mais donne une meilleure politique finale ; contre-prédiction discutée : les épisodes étant courts, γ = 0,9 peut suffire | épisodes-au-seuil, reward final |
| H2 | Q-Learning atteint le seuil plus tôt ; SARSA est plus stable pendant l'exploration | épisodes-au-seuil ; σ du reward d'entraînement |
| H3 | Les résultats sont plus sensibles aux hyperparamètres qu'au choix d'algorithme | étendues intra- vs inter-algorithmes, η² descriptif |
| H4 | Décroissance linéaire ≈ exponentielle en performance finale ; Boltzmann converge au moins aussi vite ; UCB couvre mieux les paires (s, a) en début d'entraînement mais est sensible à c | épisodes-au-seuil, premier succès, couverture (s, a) |
| H5 | Le shaping basé potentiel accélère la convergence sans dégrader le reward natif final ; le bonus naïf dégrade ; la sur-pénalité de pas est neutre à négative | épisodes-au-seuil et reward final **natifs** |
| H6 | Le tabulaire domine le DQN : efficacité échantillon ≥ 5×, temps ≥ 50×, mémoire et latence moindres, à performance finale comparable | courbes vs env-steps cumulés, temps E7, mémoire |
| H7 | Double Q-Learning réduit le biais de surestimation (E[max Q] − retour réel) et converge un peu plus lentement | gap de surestimation, épisodes-au-seuil |
| H8 | Monte Carlo est plus lent et plus variable que les méthodes TD (runs censurés possibles) | épisodes-au-seuil censuré, taux de convergence |
| H9 (exploratoire) | Multi-passagers : croissance sur-linéaire du coût de convergence par rapport au facteur ×28,8 de l'espace d'états ; le classement QL > SARSA persiste | épisodes-au-seuil relatif (descriptif) |

### 4.4 Matrice d'expériences

La campagne compte **~840 runs** (≈ 4 h 30 – 5 h 30 sur 6 cœurs) :

| Bloc | Contenu | Runs | Alimente |
|---|---|---|---|
| E0 | BruteForce, troncatures 200 et 2000 pas, 1000 épisodes d'évaluation | 20 | baseline, artefact de troncature |
| E1a | Grille Q-Learning : α ∈ {0,05 ; 0,1 ; 0,15 ; 0,2 ; 0,3} × γ ∈ {0,9 ; 0,95 ; 0,99 ; 0,999} | 200 | H1, H3, `optimized.yaml` |
| E1b | Grille croisée : {SARSA, Expected SARSA, Double QL, MC} × 3α × 3γ | 360 | H3 |
| E2 | Face-à-face : 6 algorithmes tabulaires + DQN à la configuration de référence | 70 | H2, H6, H7, H8 |
| E3 | Exploration : ε-greedy exp., ε-greedy lin., Boltzmann, UCB | 40 | H4 |
| E4 | Shaping : natif, potentiel, naïf, sur-pénalité de pas | 40 | H5 |
| E5 | Sensibilité DQN : learning rate × taille cachée (5 seeds) | 30 | annexe DQN |
| E6 | Multi-passagers : QL et SARSA × 150 000 épisodes | 20 | H9 |
| E7 | **Chronométrage séquentiel** : 6 algos × 10 seeds + DQN cuda/cpu × 3 | 66 | temps, mémoire, ablation GPU |

Les blocs E0–E6 sont parallélisés sur 6 processus : la parallélisation
n'affecte ni les récompenses ni les trajectoires (générateurs indépendants par
run), la contrainte « pas de parallélisation » ne concernant que les mesures
de temps, isolées dans le bloc E7 exécuté séquentiellement sur machine au
repos.

**Équité des comparaisons.** Trois axes sont rapportés, et aucun algorithme
n'est arrêté plus tôt qu'un autre : (1) **épisodes égaux** — les tabulaires
entre eux, à 15 000 épisodes ; (2) **env-steps cumulés égaux** — axe principal
de la comparaison tabulaire/DQN (H6), un épisode DQN consommant le même type
de transitions qu'un épisode tabulaire mais leur nombre par épisode différant
selon la politique courante ; (3) **wall-clock égal** — en annexe, extrait
de E7.

### 4.5 Méthodologie statistique

1. Les échantillons sont les **agrégats par run** (n = 10 par groupe).
2. **Shapiro-Wilk** (α = 0,05) sur chaque groupe : si les deux groupes sont
   compatibles avec la normalité, **test t de Welch** (variances inégales) ;
   sinon **Mann-Whitney U**. À n = 10 la puissance de Shapiro-Wilk est faible :
   lorsque Welch et Mann-Whitney divergent sur la significativité, les deux
   p-valeurs sont rapportées.
3. Comparaisons multiples : correction de **Holm-Bonferroni par famille
   d'hypothèses** ; p brut **et** p ajusté rapportés.
4. Tailles d'effet : **g de Hedges** (avec Welch) et **δ de Cliff** (avec
   Mann-Whitney) — la significativité sans magnitude n'informe pas.
5. Intervalles de confiance à 95 % de Student : μ ± t₀.₉₇₅,ₙ₋₁ · σ/√n.
6. Format de rapport systématique : « QL : m ± σ (IC 95 % [a ; b], n=10) vs
   SARSA : … ; Welch, p = …, p_Holm = …, g = … ».
7. Non-significativité : l'IC de la différence est rapporté, jamais la
   conclusion « pas de différence » (absence de preuve ≠ preuve d'absence).
8. Métriques censurées (seuil non atteint) : Mann-Whitney avec **censure au
   rang le pire** + taux de convergence rapporté séparément.

### 4.6 Architecture logicielle et reproductibilité

L'infrastructure expérimentale repose sur quatre composants
(`docs/ARCHITECTURE.md`) : **BaseAgent** définit le contrat commun des agents
(patron Strategy : `select_action`, `learn`, `end_episode`), le **Trainer**
exécute la boucle d'entraînement avec sondes et callbacks, l'**Evaluator**
mesure la politique greedy sur les seeds figés, et le **Benchmarker**
orchestre la campagne. Chaque run est identifié par le triplet
`(bloc, hash8, seed)` où `hash8` est le SHA-256 tronqué de la configuration
canonique **hors seed** ; les runs sont **idempotents** : un répertoire de run
complet est réutilisé tel quel, ce qui rend la campagne reprenable après un
crash et chaque figure régénérable depuis `results/raw/`. Les versions
exactes des bibliothèques sont journalisées dans le `config.json` de chaque
run.

**Environnement matériel et logiciel** : CPU 6 cœurs, 27 Go de RAM, GPU
NVIDIA RTX 2060 (6 Go, torch 2.5.1+cu124), WSL2, Python 3.10.12,
**gymnasium 1.2.3** — épinglé `<1.3` car Taxi-v3 est retiré de
gymnasium ≥ 1.3.0, remplacé par un Taxi-v4 identique aux paramètres par
défaut. L'environnement s'exécute à ~56 000 steps/s, ce qui rend la campagne
tabulaire complète très abordable en temps de calcul.

---

## 5. Résultats expérimentaux

<!-- SQUELETTE : les valeurs {{...}} et les figures seront insérées après
     dépouillement de la campagne. Chaque sous-section suit le même plan :
     rappel de la prédiction, résultats, test statistique, analyse. -->

### 5.1 Baseline brute force : ce que « aléatoire » veut dire (E0)

Le sujet exige la comparaison à un agent *brute force* : une politique
uniforme sans apprentissage. Le sujet annonce ~350 pas par épisode en moyenne
pour un agent aléatoire contre ~13 pour un agent entraîné, soit un facteur ~20.

| Agent | Troncature | Reward moyen | Reward médian | Steps moyens | Taux de succès |
|---|---|---|---|---|---|
| BruteForce | 200 pas | {{E0_BF200_REWARD}} | {{E0_BF200_REWARD_MEDIAN}} | {{E0_BF200_STEPS}} | {{E0_BF200_SUCCESS}} |
| BruteForce | 2000 pas | {{E0_BF2000_REWARD}} | {{E0_BF2000_REWARD_MEDIAN}} | {{E0_BF2000_STEPS}} | {{E0_BF2000_SUCCESS}} |
| Politique optimale (R\*) | 200 pas | 8,05 | — | — | 100 % |

**L'artefact de troncature.** La moyenne de reward à 200 pas est un
**artefact de mesure**, pas une propriété de la politique aléatoire : la
plupart des épisodes sont coupés à 200 pas avant résolution, ce qui plafonne
la pénalité par épisode et écrase la distribution contre la borne. La
variante à 2000 pas révèle le vrai coût d'une marche aléatoire sur cette
grille : {{E0_BF2000_STEPS}} pas en moyenne pour résoudre un épisode (médiane
{{E0_BF2000_STEPS_MEDIAN}}), à confronter aux ~350 annoncés par le sujet. Le
ratio de pas entre l'agent aléatoire (mesuré sans troncature) et l'agent
entraîné s'établit à **{{E0_RATIO_STEPS}}×**.

<!-- FIGURE F4: results/figures/F4_barplot_steps.png -->

**Analyse.**
<!-- Questions guides :
  1. Le taux de succès à 200 pas est-il cohérent avec la probabilité qu'une
     marche aléatoire résolve la tâche en 200 pas (deux actions contextuelles
     sur 6, dont une seule séquence gagnante) ?
  2. Pourquoi la médiane de reward est-elle plus honnête que la moyenne pour
     décrire l'agent aléatoire à 200 pas ?
  3. Le facteur ~20× annoncé par le sujet est-il retrouvé, et sous quelle
     définition exacte (steps moyens tronqués ou non) ? -->

### 5.2 Grid search et sensibilité aux hyperparamètres (E1a, E1b — H1, H3)

**Prédictions.** H1 : γ = 0,99 converge plus lentement que γ = 0,9 mais
produit une meilleure politique finale — avec une contre-prédiction assumée :
les épisodes de Taxi-v3 étant courts (une vingtaine de pas optimaux),
l'horizon effectif requis est faible et γ = 0,9 pourrait suffire. H3 : la
variabilité induite par les hyperparamètres **au sein** d'un algorithme excède
celle observée **entre** algorithmes à configuration égale.

La grille E1a (Q-Learning, 5 valeurs de α × 4 valeurs de γ × 10 seeds
= 200 runs) donne la cartographie suivante :

{{T_E1_GRID}}

<!-- FIGURE F5: results/figures/F5_heatmap_grid.png -->
<!-- FIGURE F5b: results/figures/F5b_heatmap_convergence.png -->
<!-- FIGURE F6a: results/figures/F6a_sensibilite_alpha.png -->
<!-- FIGURE F6b: results/figures/F6b_sensibilite_gamma.png -->

**Test de H1** (famille Holm « H1 », γ = 0,99 vs γ = 0,9 à α = 0,15) :

- Épisodes-au-seuil : γ = 0,99 : {{H1_SEUIL_G99}} vs γ = 0,9 : {{H1_SEUIL_G90}} ; {{H1_TEST_SEUIL}}
- Reward final : γ = 0,99 : {{H1_REWARD_G99}} vs γ = 0,9 : {{H1_REWARD_G90}} ; {{H1_TEST_REWARD}}

Verdict H1 : {{H1_VERDICT}}

**Test de H3.** L'étendue des performances finales à travers la grille
d'hyperparamètres, au sein du seul Q-Learning, vaut {{H3_ETENDUE_INTRA}} ;
l'étendue entre les cinq algorithmes tabulaires à la configuration de
référence vaut {{H3_ETENDUE_INTER}}. La part de variance expliquée
(η² descriptif, bootstrap) attribue {{H3_ETA2}} aux hyperparamètres.
Verdict H3 : {{H3_VERDICT}}

La meilleure cellule de la grille — α = {{E1_BEST_ALPHA}},
γ = {{E1_BEST_GAMMA}}, reward final {{E1_BEST_REWARD}} — est promue dans
`configs/optimized.yaml` et alimente le mode time-limited (section 5.8).

**Analyse.**
<!-- Questions guides :
  1. La surface de réponse est-elle un plateau (large bassin de configurations
     quasi optimales) ou un pic étroit ? Qu'est-ce que cela implique pour le
     tuning en pratique ?
  2. α élevé (0,3) accélère-t-il la convergence au prix d'un bruit résiduel en
     fin d'entraînement (politique greedy instable entre checkpoints) ?
  3. γ = 0,999 pose-t-il un problème spécifique (propagation lente des valeurs,
     quasi-absence d'actualisation) visible dans les épisodes-au-seuil ? -->

### 5.3 Face-à-face des algorithmes tabulaires (E2 — H2, H7, H8)

**Prédictions.** H2 : le Q-Learning, off-policy et optimiste, atteint le seuil
de convergence plus tôt ; SARSA, on-policy et donc prudent tant que ε est
grand, présente un reward d'entraînement plus stable (σ plus faible). H7 : le
Double Q-Learning réduit le gap de surestimation E[max Q] − retour réel, au
prix d'une convergence légèrement plus lente (chaque table n'apprend que sur
la moitié des transitions). H8 : Monte Carlo, sans bootstrapping et exposé aux
retours tronqués, est plus lent et plus variable, avec des runs possiblement
censurés.

Les six algorithmes tabulaires et le DQN sont entraînés à la configuration de
référence (α = 0,15, γ = 0,99, ε : 1,0 → 0,01, n = 10 seeds) :

{{T1_FACE_A_FACE}}

<!-- FIGURE F1: results/figures/F1_courbes_apprentissage.png -->
<!-- FIGURE F2: results/figures/F2_courbes_steps.png -->
<!-- FIGURE F3: results/figures/F3_boxplots_rewards.png -->

**Test de H2** (famille Holm « H2 ») :

- Épisodes-au-seuil : QL : {{H2_SEUIL_QL}} vs SARSA : {{H2_SEUIL_SARSA}} ; {{H2_TEST_SEUIL}}
- Stabilité (σ du reward d'entraînement par run) : QL : {{H2_SIGMA_QL}} vs SARSA : {{H2_SIGMA_SARSA}} ; {{H2_TEST_SIGMA}}

Verdict H2 : {{H2_VERDICT}}

**Test de H7.** Le gap de surestimation (moyenne de max_a Q(s₀, a) sur les
états initiaux d'évaluation, moins le retour greedy réellement obtenu) vaut
{{H7_GAP_QL}} pour le Q-Learning contre {{H7_GAP_DQL}} pour le Double
Q-Learning ({{H7_TEST_GAP}}). Les épisodes-au-seuil du Double Q-Learning
s'établissent à {{H7_SEUIL_DQL}}. Verdict H7 : {{H7_VERDICT}}

<!-- FIGURE F10: results/figures/F10_surestimation.png -->

**Test de H8.** Monte Carlo atteint le seuil en {{H8_SEUIL_MC}} épisodes
(taux de convergence : {{H8_TAUX_CONV_MC}} des runs ; les runs censurés sont
traités au rang le pire) ; {{H8_TEST}}. Verdict H8 : {{H8_VERDICT}}

À titre de synthèse sur l'évaluation finale (100 épisodes figés) :
QL : {{E2_QL_REWARD}} ; SARSA : {{E2_SARSA_REWARD}} ;
Expected SARSA : {{E2_EXPSARSA_REWARD}} ; Double QL : {{E2_DQL_REWARD}} ;
MC : {{E2_MC_REWARD}} ; DQN : {{E2_DQN_REWARD}} — à comparer à R\* = 8,05.

<!-- FIGURE F11: results/figures/F11_heatmap_qvalues.png -->

**Analyse.**
<!-- Questions guides :
  1. Les différences entre algorithmes sont-elles cinétiques (vitesse) ou
     asymptotiques (politique finale) ? Tous les non-censurés rejoignent-ils
     R* à ±IC près ?
  2. Le σ d'entraînement plus faible de SARSA (s'il est confirmé) provient-il
     de l'évitement des −10 pendant l'exploration ? Vérifier via le taux de
     pénalités d'entraînement.
  3. Expected SARSA se comporte-t-il comme un SARSA débruité (même trajectoire
     moyenne, variance réduite), conformément à van Seijen et al. (2009) ?
  4. Les runs MC censurés partagent-ils une signature (boucles de politiques
     sous troncature, retours biaisés) visible dans les courbes de sonde ? -->

### 5.4 Stratégies d'exploration (E3 — H4)

**Prédictions.** H4 : à performance finale égale, la décroissance linéaire de
ε vaut l'exponentielle ; Boltzmann converge au moins aussi vite (exploration
graduée par les valeurs) ; UCB couvre mieux l'espace (s, a) en début
d'entraînement (bonus d'incertitude dirigé) mais sa performance est sensible
au coefficient c.

Les quatre stratégies sont comparées sur Q-Learning à configuration de
référence, seule l'exploration variant :

{{T_E3_EXPLORATION}}

- Épisodes-au-seuil : ε-exp : {{H4_SEUIL_EXP}} ; ε-lin : {{H4_SEUIL_LIN}} ;
  Boltzmann : {{H4_SEUIL_BOLTZ}} ; UCB : {{H4_SEUIL_UCB}}
- Couverture des paires (s, a) à 1000 épisodes : ε-greedy :
  {{H4_COUVERTURE_EPS}} vs UCB : {{H4_COUVERTURE_UCB}}
- Tests appariés contre la référence ε-exp (famille Holm « H4 ») : {{H4_TESTS}}

Verdict H4 : {{H4_VERDICT}}

<!-- FIGURE F7: results/figures/F7_exploration.png -->

**Analyse.**
<!-- Questions guides :
  1. La forme de la décroissance (lin vs exp) importe-t-elle moins que le
     moment où ε devient négligeable (fraction d'horizon commune de 0,6) ?
  2. La couverture précoce supérieure d'UCB (si confirmée) se convertit-elle
     en convergence plus rapide, ou l'exploration systématique des actions
     illégales (−10) la pénalise-t-elle ?
  3. Boltzmann évite-t-il mieux les −10 que ε-greedy (les actions à Q très
     négatif deviennent exponentiellement rares au lieu de rester à ε/6) ? -->

### 5.5 Reward shaping : le théorème de Ng et al. à l'épreuve (E4 — H5)

**Prédictions.** H5 : le shaping **basé potentiel** — F = γ·Φ(s′) − Φ(s) avec
Φ(s) = −(distance de Manhattan à l'objectif courant : le passager tant qu'il
n'est pas embarqué, la destination ensuite) — accélère la convergence sans
dégrader le reward **natif** final, conformément au théorème d'invariance
(Ng, Harada & Russell, 1999). Le **bonus naïf** (+0,5 dès que le taxi se
rapproche de l'objectif, sans structure de potentiel) constitue le contrôle
négatif : des cycles peuvent accumuler du bonus, la politique optimale n'est
plus garantie. La **sur-pénalité de pas** (−1 → −2 sur les déplacements) est
prédite neutre à négative : elle ne change pas l'ordre des politiques courtes
mais durcit le paysage pendant l'exploration.

Toutes les métriques ci-dessous sont **natives** (le shaping n'affecte que le
signal d'apprentissage, jamais la mesure) :

{{T_E4_SHAPING}}

- Épisodes-au-seuil : natif : {{H5_SEUIL_NATIVE}} ; potentiel :
  {{H5_SEUIL_POTENTIEL}} ; naïf : {{H5_SEUIL_NAIF}} ; sur-pénalité :
  {{H5_SEUIL_STEP}}
- Reward final natif : natif : {{H5_REWARD_NATIVE}} ; potentiel :
  {{H5_REWARD_POTENTIEL}} ; naïf : {{H5_REWARD_NAIF}}
- Non-infériorité du potentiel (IC 95 % de la différence de reward final
  vs natif) : {{H5_IC_NON_INF}}
- Tests appariés contre le natif (famille Holm « H5 ») : {{H5_TESTS}}

Verdict H5 : {{H5_VERDICT}}

<!-- FIGURE F8: results/figures/F8_reward_shaping.png -->

**Analyse.**
<!-- Questions guides :
  1. Le bonus naïf dégrade-t-il la politique finale (reward natif inférieur,
     trajectoires allongées) ou seulement la vitesse — et observe-t-on des
     boucles de collecte de bonus dans les replays ?
  2. Sur un MDP déterministe à horizon court où la récompense native est déjà
     dense (−1 par pas), le potentiel a-t-il encore une marge d'accélération
     mesurable ?
  3. L'IC de non-infériorité permet-il d'affirmer la préservation de la
     politique optimale, ou seulement de ne pas la rejeter ? -->

### 5.6 Tabulaire vs DQN : le coût du deep RL (E2, E5, E7 — H6)

**Prédictions.** H6 : sur un espace d'états de 500 entiers, le tabulaire
domine le DQN sur tous les axes de coût — efficacité échantillon (≥ 5×), temps
d'entraînement (≥ 50×), mémoire et latence de décision — à performance finale
comparable. Le DQN utilisé (Mnih et al., 2015) encode l'état en one-hot vers
un MLP à deux couches cachées, avec replay buffer, réseau cible à mises à jour
douces (Polyak), perte de Huber et écrêtage de gradient.

La comparaison principale se fait à **env-steps cumulés égaux** (axe 2 du
protocole), les budgets en épisodes différant (15 000 vs 5 000) :

{{T_H6_RESSOURCES}}

- Efficacité échantillon (env-steps pour atteindre le seuil, ratio
  DQN/tabulaire) : {{H6_RATIO_SAMPLE}}
- Temps d'entraînement (E7, séquentiel ; moyenne et médiane) : ratio
  {{H6_RATIO_TEMPS}}
- Mémoire (Q-table : 500×6 float64 = 24 ko ; DQN : paramètres + buffer) :
  ratio {{H6_RATIO_MEMOIRE}}
- Performance finale DQN : {{H6_DQN_REWARD}} vs R\* = 8,05 ; {{H6_TEST}}

Verdict H6 : {{H6_VERDICT}}

<!-- FIGURE F9: results/figures/F9_efficacite_echantillon.png -->

**Ablation CPU/GPU.** Le chronométrage E7 inclut le DQN sur cuda et sur cpu
(3 runs chacun) : {{E7_DQN_CUDA_TEMPS}} (cuda) contre {{E7_DQN_CPU_TEMPS}}
(cpu). {{T_E7_TEMPS}}

**Analyse.**
<!-- Questions guides :
  1. Sur des mini-lots minuscules et un MLP étroit, le GPU est-il réellement
     plus rapide que le CPU, ou le coût des transferts hôte-device domine-t-il ?
  2. La sensibilité du DQN à (lr, hidden) observée en E5 confirme-t-elle que le
     deep RL importe surtout de nouveaux hyperparamètres à régler ?
  3. Le DQN atteint-il exactement R* ou plafonne-t-il légèrement en dessous
     (approximation, cibles mouvantes) — et l'écart est-il significatif ? -->

### 5.7 Multi-passagers : passage à l'échelle tabulaire (E6 — H9)

**Prédiction.** H9 (exploratoire, descriptive) : l'espace d'états étant
multiplié par 28,8 (500 → 14 400) et les récompenses terminales étant plus
éparses (deux livraisons successives), le coût de convergence croît
**sur-linéairement** par rapport au facteur d'états ; le classement
QL > SARSA en vitesse observé en E2 persiste.

Q-Learning et SARSA sont entraînés 150 000 épisodes (10 seeds chacun) sur
l'environnement de la section 3.2 :

{{T_E6_MULTI}}

- Épisodes-au-seuil (seuil recalculé sur l'environnement multi) :
  QL : {{H9_SEUIL_QL_MULTI}} ; SARSA : {{H9_SEUIL_SARSA_MULTI}}
- Coût relatif de convergence (multi / simple, à algorithme égal) :
  {{H9_RATIO_COUT}} — à comparer au facteur ×28,8 de l'espace d'états
- Performance finale : QL : {{E6_QL_REWARD}} ; longueur des tournées greedy :
  {{E6_STEPS_GREEDY}}

Verdict H9 : {{H9_VERDICT}}

<!-- FIGURE F12: results/figures/F12_multi_passagers.png -->

**Analyse.**
<!-- Questions guides :
  1. La croissance sur-linéaire (si observée) s'explique-t-elle par la
     couverture (chaque état est visité ~29× moins souvent à budget égal) ou
     par l'allongement de la chaîne de crédit (deux +20 à propager) ?
  2. Les tournées greedy apprises embarquent-elles réellement les deux
     passagers simultanément quand c'est optimal (preuve d'optimisation de
     tournée, pas de simple enchaînement de courses) ?
  3. Le classement QL > SARSA persiste-t-il avec le même ordre de grandeur
     d'écart relatif qu'en E2 ? -->

### 5.8 Mode time-limited

Le mode time-limited du CLI (US-1.4) charge les hyperparamètres optimisés
issus de E1a (`configs/optimized.yaml`) et vise, selon ses critères
d'acceptation, une convergence en moins de 5 000 épisodes et un reward moyen
supérieur à 7,0 sur 100 épisodes de test. C'est le seul contexte où l'early
stopping est autorisé — il s'agit d'un mode produit, pas d'une expérience.

{{T_TIME_LIMITED}}

- Épisodes effectifs avant arrêt : {{TL_EPISODES}}
- Reward moyen sur 100 épisodes de test : {{TL_REWARD}}
- Temps total (entraînement + évaluation) : {{TL_TEMPS}}

**Analyse.**
<!-- Questions guides :
  1. Les critères d'acceptation (< 5000 épisodes, reward > 7,0) sont-ils tenus
     avec marge, et sur toutes les seeds essayées ?
  2. Quel est le compromis exact entre l'arrêt anticipé et la qualité finale
     par rapport au budget complet de 15 000 épisodes ? -->

---

## 6. Discussion

<!-- Trame rédactionnelle : chaque thème sera étayé par les valeurs de la
     section 5 une fois insérées. -->

### 6.1 Des différences cinétiques, rarement asymptotiques

Le fil conducteur des résultats tabulaires est la distinction entre **vitesse
de convergence** et **qualité asymptotique**. Sur un MDP fini, déterministe et
exhaustivement visitable, tous les algorithmes satisfaisant les conditions de
convergence apprennent la même politique optimale : les écarts de fin
d'entraînement relèvent de l'échantillonnage, pas de l'algorithme. H2, H7 et
H8 sont donc des hypothèses sur des **trajectoires d'apprentissage** — qui
atteint le seuil le premier, avec quelle variance, au prix de quel biais
transitoire — et non sur des plafonds de performance.
{{DISC_CINETIQUE_SYNTHESE}}

### 6.2 Hyperparamètres contre algorithmes

L'hypothèse H3 touche à une leçon générale du domaine : à famille d'algorithmes
égale, la configuration importe souvent plus que l'étiquette. Si la campagne
confirme que l'étendue intra-Q-Learning à travers la grille (α, γ) dépasse
l'étendue inter-algorithmes à configuration fixée, alors comparer des
algorithmes sans contrôler leurs hyperparamètres — pratique répandue — revient
à mesurer du bruit de tuning. C'est aussi un argument méthodologique en faveur
des grilles complètes (E1) préalables à tout face-à-face (E2).
{{DISC_H3_SYNTHESE}}

### 6.3 Le prix du deep RL sur un problème tabulaire

Le DQN sur Taxi-v3 est volontairement une expérience de **coût**, pas de
performance : 500 états tiennent dans 24 ko de Q-table, et aucune
généralisation entre états n'est nécessaire (l'encodage one-hot l'interdit
d'ailleurs par construction). Les ratios de H6 — échantillons, temps, mémoire,
latence — chiffrent ce que coûte l'approximation neuronale quand elle ne sert
à rien, et l'ablation CPU/GPU rappelle que le GPU n'accélère pas
mécaniquement de petits réseaux à petits lots. La valeur du deep RL est
ailleurs : elle commence là où la table s'arrête, ce que l'extension
TrackMania (section 7) matérialise sans avoir pu être exécutée.
{{DISC_H6_SYNTHESE}}

### 6.4 Ce que « à épisodes égaux » ne dit pas

Comparer à épisodes égaux avantage subtilement les politiques qui échouent
vite : un épisode raté de 200 pas consomme 15 fois plus de transitions qu'un
épisode optimal d'environ 13 pas, si bien que deux agents « à 15 000
épisodes » n'ont pas vu le même nombre d'interactions avec l'environnement.
C'est pourquoi le protocole rapporte trois axes (épisodes, env-steps,
wall-clock) et fait des env-steps l'axe principal de H6. De même, la
troncature à 200 pas fabrique un plancher de reward qui rend la moyenne du
brute force ininterprétable (section 5.1) : deux artefacts de mesure distincts
qui plaident pour une lecture toujours *instrumentée* des métriques RL.
{{DISC_EQUITE_SYNTHESE}}

### 6.5 Le reward shaping, boussole ou béquille ?

Sur Taxi-v3, la récompense native est déjà **dense** (−1 par pas structure le
gradient temporel) et l'horizon est court : l'espace laissé au shaping basé
potentiel pour accélérer est étroit, et son intérêt est surtout de vérifier
expérimentalement le théorème d'invariance — le contrôle négatif naïf jouant
le rôle de falsificateur. La leçon pratique attendue est double : (1) le
shaping sûr existe et se code en cinq lignes dès qu'on dispose d'un potentiel
raisonnable ; (2) le shaping intuitif mais non potentiel est un piège
silencieux, indétectable si l'on ne rapporte que la récompense façonnée —
d'où la convention stricte de ce protocole de ne jamais rapporter autre chose
que la récompense native. {{DISC_H5_SYNTHESE}}

---

## 7. Limites et perspectives

### 7.1 Limites de validité interne

- **Variance temporelle sous WSL2.** Les mesures de temps, même séquentielles
  et machine au repos (bloc E7), héritent de la variance de la couche de
  virtualisation WSL2 ; les médianes sont rapportées en complément des
  moyennes, mais les ratios de temps de H6 doivent être lus comme des ordres
  de grandeur, pas comme des constantes physiques.
- **Puissance statistique à n = 10.** Dix runs par configuration bornent la
  puissance des tests, et Shapiro-Wilk détecte mal les écarts à la normalité à
  cet effectif — le protocole rapporte les deux p-valeurs (Welch et
  Mann-Whitney) en cas de divergence, mais une non-significativité à n = 10
  reste peu informative (d'où le rapport systématique des IC de différence).
- **Un seul environnement de référence.** Toutes les conclusions
  algorithmiques sont établies sur un MDP déterministe à horizon court ; leur
  transfert à des dynamiques stochastiques n'est pas garanti (voir 7.3).

### 7.2 Extension TrackMania : livrée mais non exécutée

L'extension deep RL vers TrackMania 2020 (`docs/TRACKMANIA.md`,
`src/environments/trackmania_wrapper.py`, `scripts/train_trackmania.py`) est
**livrée et testée, mais n'a pas été exécutée en conditions réelles** : elle
exige une machine de jeu Windows avec TrackMania 2020 et le plugin OpenPlanet
(qui ne tournent ni sous Linux ni en CI), le tout en temps réel.
L'architecture a été conçue pour être validable sans le jeu : le wrapper ne
dépend que d'une interface structurelle (`TMEnvProtocol`), jamais de `tmrl`
directement, et a été testé unitairement sur un environnement factice en
mémoire (`FakeTMEnv`) — aplatissement et normalisation des observations
(vitesse + LIDAR) vers un espace continu `Box(-1, 1, (83,))`, consommé par un
agent SAC de Stable-Baselines3. L'exécution de ce pipeline sur machine
compatible ferait passer le projet d'un espace de 500 états discrets à un
espace continu en temps réel, où chaque transition coûte du temps d'horloge et
où l'efficacité échantillon, marginale sur Taxi, devient la contrainte
dominante.

### 7.3 Variantes stochastiques de Taxi

Gymnasium expose deux paramètres qui brisent le déterminisme de Taxi-v3 :
`is_rainy` (les déplacements réussissent avec probabilité 0,8, glissent
latéralement sinon) et `fickle_passenger` (le passager peut changer de
destination en cours de course). Ces variantes constituent la suite logique de
la campagne : elles réintroduisent précisément ce que notre MDP déterministe
neutralise — l'écart on-policy/off-policy de SARSA face au risque (le
« cliff walking » de Sutton & Barto), l'intérêt du lissage d'Expected SARSA,
et la surestimation du Q-Learning que le bruit de transition amplifie. Les
hypothèses H2 et H7, testées ici dans un cadre déterministe qui leur est
défavorable, mériteraient d'y être rejouées à protocole constant.

### 7.4 Autres pistes

- **Multi-passagers au-delà de 2** : l'encodage en base mixte se généralise,
  mais l'espace croît en 25·6^k·4^k — la frontière tabulaire/deep se
  franchirait autour de k = 3 (≈ 350 000 états), offrant une transition
  naturelle vers un DQN enfin justifié.
- **Sondes d'apprentissage plus riches** : suivre la norme ‖Q_t − Q\*‖∞ au fil
  de l'entraînement (Q\* étant disponible par value iteration) donnerait une
  mesure de convergence indépendante du reward, plus fine que le seuil 0,9·R\*.
- **Seuil de convergence alternatif** : le seuil à 95 % de R\* mentionné au
  cadrage, plus exigeant, est reporté en annexe ; une analyse de sensibilité
  du classement des algorithmes au choix du seuil consoliderait H2/H7/H8.

---

## 8. Conclusion

Ce projet a traité Taxi-v3 non comme un exercice de performance mais comme un
banc d'essai contrôlé : un étalon optimal calculé par value iteration
(R\* = 8,05), neuf hypothèses formulées a priori, environ 840 runs sous
protocole figé, et des conclusions systématiquement adossées à des tests
statistiques corrigés et à des tailles d'effet.

Du côté des résultats : la base aléatoire, une fois débarrassée de l'artefact
de troncature, établit le point de départ ({{E0_BF2000_STEPS}} pas par épisode) ;
{{CONCLUSION_MEILLEUR_ALGO}} fournit la meilleure trajectoire de convergence
vers la politique optimale, atteinte à {{CONCLUSION_REWARD_FINAL}} de reward
moyen en évaluation pour {{CONCLUSION_STEPS_FINAL}} pas par épisode ; les
hypothèses H1 à H8 sont tranchées comme suit : {{CONCLUSION_VERDICTS}}.

Sur les extensions : le DQN confirme que le deep RL est un instrument de
généralisation, pas d'accélération — son coût sur un problème tabulaire est
chiffré par H6 ({{H6_RATIO_TEMPS}} en temps) ; l'environnement multi-passagers
à 14 400 états montre que la tabulation encaisse un facteur ×28,8 d'états au
prix d'un coût de convergence {{H9_RATIO_COUT}} ; et le pipeline TrackMania,
livré et testé hors jeu, trace la frontière au-delà de laquelle la Q-table
cède la place au réseau.

La conclusion méthodologique est peut-être la plus durable : sur ce problème,
{{CONCLUSION_H3_PHRASE}} — et la moitié des « résultats » qu'un protocole
naïf aurait rapportés (moyenne du brute force, reward d'entraînement, meilleur
checkpoint) étaient des artefacts de mesure que le protocole a dû neutraliser
un à un.

---

## 9. Références bibliographiques

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
   Introduction* (2ᵉ éd.). MIT Press.
2. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. Thèse de
   doctorat, King's College, Université de Cambridge.
3. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*,
   8(3–4), 279–292.
4. Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using
   connectionist systems*. Rapport technique CUED/F-INFENG/TR 166,
   Département d'ingénierie, Université de Cambridge.
5. van Seijen, H., van Hasselt, H., Whiteson, S., & Wiering, M. (2009).
   A theoretical and empirical analysis of Expected Sarsa. *Actes de l'IEEE
   Symposium on Adaptive Dynamic Programming and Reinforcement Learning
   (ADPRL 2009)*, 177–184.
6. van Hasselt, H. (2010). Double Q-learning. *Advances in Neural Information
   Processing Systems*, 23, 2613–2621.
7. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control
   through deep reinforcement learning. *Nature*, 518(7540), 529–533.
8. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward
   transformations: Theory and application to reward shaping. *Actes de la
   16ᵉ International Conference on Machine Learning (ICML 1999)*, 278–287.
9. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of
   the multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256.
10. Towers, M., Kwiatkowski, A., Terry, J., et al. (2024). Gymnasium: A
    standard interface for reinforcement learning environments.
    *arXiv:2407.17032*.
