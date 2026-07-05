# Journal de bord — Agent Monte Carlo Tree Search (Taxi)

## Contexte du choix technique

Dans le découpage des tâches du groupe, j'ai choisi de m'attaquer à l'algorithme Monte Carlo, pour pouvoir travailler sur un algorithme différent des autres membres du groupe, avoir un point de comparaison, et disposer d'un benchmark intéressant.

---

## 1. Recherche théorique

### 1.1 Algorithme Monte Carlo

Le but principal de l'algorithme Monte Carlo pour l'IA est d'essayer beaucoup de "runs" au hasard afin d'estimer un résultat. L'objectif n'est pas de chercher directement la route qui donne le meilleur résultat, mais de trouver celles qui donnent les meilleurs résultats **en moyenne**.

**Fonctionnement :**

```
répéter N fois :
    générer un scénario aléatoire
    observer le résultat
calculer la moyenne des résultats
```

**En résumé :** Monte Carlo permet d'estimer des résultats, mais il n'est pas conçu pour optimiser une suite de décisions. Il a peu de mémoire et ne permet pas de déterminer quelle action choisir à chaque étape.

**Problème identifié :** avec seulement une notation succès/échec, l'IA sait qu'il ne faut pas se prendre de mur, mais ne sait pas forcément comment trouver le chemin le plus rapide.

**Conclusion :** l'algorithme Monte Carlo est utile pour estimer des probabilités, estimer des résultats, et tester à grande échelle des scénarios aléatoires — mais il manque de précision car il reste très général.

### 1.2 Monte Carlo Tree Search (MCTS)

> À noter : c'est un algorithme très utilisé dans les jeux vidéo (Civilization, Total War).

Algorithme basé sur Monte Carlo, qui cherche à construire un arbre de décisions pour mémoriser les meilleurs résultats grâce à une notation plus fine des actions. Il donne des scores plus ou moins élevés selon que l'action mène à un succès **et** au chemin le plus rapide, ou non (score − nombre de pas).

**Fonctionnement en 4 étapes :**

| Étape | Rôle |
|---|---|
| **Sélection** | Choisir une action possible |
| **Expansion** | Ajout (ou non) d'une nouvelle possibilité d'action dans l'arbre |
| **Simulation** | Équilibrage meilleurs choix / aléatoire au fil des runs |
| **Backpropagation** | Propagation du résultat de la simulation dans tout l'arbre ; les nœuds gardent des stats (nombre de visites, score total, score moyen) |

**Conclusion :** grâce au système de score (reward), l'IA apprend non seulement à réussir, mais aussi à optimiser ses résultats. C'est cette solution que je retiens pour viser le résultat le plus performant.

---

## 2. Kick-off

Présentation des contraintes et de la philosophie du projet. J'ai appris qu'il allait falloir m'orienter sur deux notions pour l'expérimentation : le **grid search** et l'**epsilon decay**.

**Grid search :** l'idée est de construire un tableau avec une liste de paramètres, plusieurs lignes avec des variables différentes, et de tester chaque cas pour en tirer des résultats et des conclusions.

**Epsilon decay :** le paramètre qui joue sur l'exploration vs l'exploitation (0.99 serait très rapide, 0.997 serait visiblement lent).

---

## 3. Mise en place de l'agent

Une fois la base théorique posée, mise en place du premier agent et setup du projet.

Deux classes structurent l'agent :

- **`MCTSNode`** — sert à l'initialisation des nœuds, et a un rôle de mémorisation de ce qui se passe pendant les simulations (values, nombre de visites, etc.) ainsi que de structuration de l'arbre des actions. L'idée globale est de prendre une "snapshot" de l'état du jeu et de fournir des informations (`total_value`, `children`, `number of visits`).

- **`MonteCarloTreeSearchAgent`** — a un rôle de pilote : il a le pouvoir de décision sur l'action à jouer. Il se sert des snapshots et des informations associées pour décider quel chemin emprunter.

### Interface de suivi

Mise en place d'une interface pour avoir une vision plus lisible de l'entraînement du modèle. Sans avoir encore une visibilité complète sur tout ce qui serait utile pour améliorer efficacement l'entraînement, j'ai commencé avec :

- une courbe montrant le reward par épisode
- une vision des déplacements de l'IA
- la dernière action réalisée (pour voir si elle s'approche de la fin)
- le nombre d'états appris cumulés au fil des runs
- l'étape à laquelle l'IA est rendue

L'idée de ce premier setup est d'avoir une bonne base, pour ensuite pouvoir simplement ajuster les valeurs et augmenter l'efficacité de l'IA.

---

## 4. Historique des entraînements

| # | Épisodes | Simulations | Rollout max | Const. UCB | Discount | Rollout guidé ? | Reward moyen | Étapes moy. | Succès | États sauvegardés |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 200 | 20 | 1.4 | 0.99 | Non | -1101.20 | 200.00 | 0 % | 65 |
| 2 | 200 | 200 | 20 | 1.4 | 0.99 | Non | -988.58 | 200.00 | 0 % | 225 |
| 3 | 200 | 200 | 50 | 1.4 | 0.99 | Non | -1226.86 | 200.00 | 0 % | 207 |
| 4 | 400 | 400 | 20 | 1.4 | 0.99 | Non | -1293.12 | 200.00 | 0 % | 279 |
| 5 | 400 | 50 | 25 | 1.0 | 0.99 | **Oui** | -14.79 | 33.69 | 90.00 % | 352 |
| 6 | 800 | 50 | 25 | 1.0 | 0.99 | **Oui** | -6.38 | 25.96 | 93.25 % | 364 |
| 7 | 400 | 200 | 25 | 1.0 | 0.99 | **Oui** | 7.78 | 13.22 | 100.00 % | 340 |

### Analyse entraînement par entraînement

**Entraînement 1 (baseline)** — 20 épisodes, rollout 100 % aléatoire. Aucun succès, reward très négatif. Point de départ pour comparer.

**Entraînement 2** — même config, 200 épisodes au lieu de 20, sans toucher aux autres paramètres, pour voir si le modèle progresse seul avec plus d'expérience. Sur 200 épisodes, le modèle sauvegarde plus d'états en mémoire et le reward moyen s'améliore légèrement — mais toujours 0 % de succès.

**Entraînement 3** — augmentation du rollout à 50 (au lieu de 20) pour voir si prédire plus de coups à l'avance aide. Résultat : ça n'améliore pas l'efficacité. Augmenter le rollout revient surtout à injecter plus de bruit aléatoire, donc plus de risques de se tromper.

**Entraînement 4** — retour au rollout à 20, mais augmentation du nombre d'épisodes (400) et de simulations par décision (400), pour donner au modèle plus d'expérience et plus de profondeur de réflexion avant de décider. Le reward reste mauvais, toujours aucune run réussie, mais le nombre d'états sauvegardés progresse.

**Diagnostic du blocage :** le problème vient du fait que la politique du rollout est 100 % aléatoire — tant que le modèle ne tombe pas sur un premier succès par hasard, il n'a rien à exploiter pour s'améliorer. Décision de guider légèrement le rollout, pour mettre le modèle sur la voie de la réussite sans pour autant tout lui dicter, afin qu'il apprenne par lui-même.

**Entraînement 5 (premier rollout guidé)** — changement de plusieurs paramètres en même temps (simulations à 50, rollout à 25, constante UCB à 1). Résultat immédiat et spectaculaire : 90 % de succès, reward moyen à -14.79, 33.69 étapes en moyenne.

**Entraînement 6** — même configuration, doublement des épisodes (800). Amélioration progressive confirmée : reward moyen -6.38, 25.96 étapes, 93.25 % de succès, plus d'états sauvegardés. On observe une **courbe de progression claire** : le reward moyen augmente, le nombre d'étapes moyen baisse, et le taux de réussite augmente.

**Entraînement 7** — augmentation du nombre de simulations à 200 (au lieu de 50), 400 épisodes. Gain drastique : reward moyen positif (+7.78), 13.22 étapes en moyenne (proche de l'optimal), **100 % de succès**.

---

## 5. Conclusion

Le rollout 100 % aléatoire est le principal facteur bloquant des quatre premiers entraînements : sans jamais atteindre un succès, l'agent MCTS n'a aucun signal exploitable pour orienter sa recherche, quels que soient le nombre d'épisodes ou de simulations. Guider le rollout avec une heuristique — sans dicter directement la décision finale à l'agent — a permis de débloquer un vrai apprentissage : le taux de succès est passé de 0 % à 100 % en trois itérations, avec une progression mesurable et cohérente (reward ↑, nombre d'étapes ↓, taux de succès ↑) à chaque ajustement.
