# Extension TrackMania — Deep RL en espace continu

## Objectif pédagogique

Le cœur du projet (Taxi-v3) porte sur le RL **tabulaire** : espace d'états discret et fini,
Q-table explicite. Cette extension prolonge le sujet vers le **deep RL** : TrackMania 2020 expose
des observations continues (vitesse, LIDAR) et des actions continues (gaz, frein, direction),
impossibles à traiter avec une Q-table. On y remplace la table par un réseau de neurones
(politique + critiques SAC via Stable-Baselines3) et on découvre les contraintes d'un
environnement **temps réel** : le jeu tourne à vitesse réelle, chaque transition coûte du temps
d'horloge, l'échantillonnage est donc précieux.

## Prérequis (machine de jeu)

| Prérequis | Détail |
| --- | --- |
| OS | Windows 10/11 (le jeu et OpenPlanet ne tournent pas sous Linux/CI) |
| Jeu | TrackMania 2020 (Steam ou Epic), édition Standard suffisante |
| Plugin | [OpenPlanet](https://openplanet.dev/) installé pour TrackMania |
| Python | Python 3.10+ avec `pip install tmrl` (>= 0.6) |
| Piste | Piste avec **bordures noires** : le LIDAR simulé de tmrl ne fonctionne que sur ces pistes (il détecte les bords par contraste) |
| Affichage | Jeu en fenêtré ou plein écran **à la résolution configurée** dans tmrl (capture d'écran calibrée) |

## Installation

### Côté machine de jeu (Windows)

1. Installer TrackMania 2020, le lancer une fois, puis installer OpenPlanet.
2. `pip install tmrl` — l'installation crée le dossier `%USERPROFILE%\TmrlData\`
   (configuration, poids, plugin OpenPlanet à copier selon le README de tmrl).
3. Vérifier/adapter `~/TmrlData/config/config.json` : interface `LIDAR`, résolution de capture,
   touches. Suivre le guide officiel : <https://github.com/trackmania-rl/tmrl#installation>.
4. Charger une piste à bordures noires (des pistes d'exemple sont fournies dans `TmrlData`),
   mettre le jeu au premier plan à la bonne résolution.
5. Valider l'installation avec l'outil de diagnostic de tmrl (`python -m tmrl --check-environment`
   ou l'équivalent documenté dans le README de la version installée).

### Côté dépôt

```bash
poetry install -E trackmania   # installe stable-baselines3
pip install tmrl               # manuel, sur la machine de jeu uniquement
```

`tmrl` n'est **pas** verrouillé dans `poetry.lock` : ses dépendances (capture d'écran, entrées
clavier/manette) sont spécifiques à Windows et casseraient l'installation sur les machines de
développement et la CI Linux.

## Usage

```bash
python scripts/train_trackmania.py --timesteps 500000 --seed 42
# Options : --checkpoint-dir models/trackmania  --log-dir results/trackmania
```

Le script sauvegarde un checkpoint SAC tous les 50 000 pas dans `models/trackmania/` et le modèle
final dans `models/trackmania/sac_trackmania_final.zip`. Les épisodes (récompense, durée) sont
journalisés par le `Monitor` SB3 dans `results/trackmania/monitor.csv`.

## Architecture

```
TrackMania 2020 + OpenPlanet (Windows, temps réel ~20 FPS via rtgym)
        │
        ▼
tmrl.get_environment()            observations tuple ((1,), (4,19), (3,), (3,))
        │
        ▼
TMEnvProtocol                     interface structurelle (testable avec un faux env)
        │
        ▼
TrackManiaEnvWrapper              aplatissement + normalisation → Box(-1, 1, (83,))
        │
        ▼
Stable-Baselines3 SAC             MlpPolicy, replay buffer 200k, checkpoints
```

Le wrapper (`src/environments/trackmania_wrapper.py`) ne dépend que du **protocole**, jamais de
`tmrl` directement : `import tmrl` n'a lieu que dans `make_trackmania_env()`. Les tests unitaires
(`tests/environments/test_trackmania_wrapper.py`) utilisent un `FakeTMEnv` en mémoire.

## Observations et actions

Observation aplatie — vecteur `float32` de forme `(83,)` :

| Indices | Contenu | Normalisation |
| --- | --- | --- |
| `0` | Vitesse | `v / max_speed` (1000 par défaut), clip `[0, 1]` |
| `1–76` | 4 scans LIDAR empilés × 19 faisceaux | `d / max_lidar` (400 par défaut), clip `[0, 1]` |
| `77–79` | Action précédente | inchangée, dans `[-1, 1]` |
| `80–82` | Action précédent-précédente | inchangée, dans `[-1, 1]` |

Les deux dernières actions font partie de l'observation car l'environnement temps réel introduit
un délai action→effet : les inclure rend le processus à nouveau markovien.

Action — `Box(-1, 1, (3,))` : `[gaz, frein, direction]`. Le wrapper clippe toute action reçue
dans `[-1, 1]` avant de la transmettre au jeu.

## Pourquoi SAC plutôt que PPO ?

- **Efficacité d'échantillonnage** : l'environnement est temps réel et mono-instance
  (impossible de vectoriser ou d'accélérer le jeu). SAC est *off-policy* : chaque transition
  est rejouée de nombreuses fois depuis le replay buffer, alors que PPO (*on-policy*) jette
  ses données après quelques epochs.
- **Actions continues** : SAC est conçu pour les espaces d'actions continus bornés.
- **Exploration** : le bonus d'entropie de SAC maintient une exploration stochastique utile
  sur des pistes longues, sans schéma d'exploration manuel.
- **Référence amont** : le pipeline d'entraînement de référence de tmrl est lui-même basé sur
  SAC, ce qui rend nos résultats comparables.

## Limites

- **Non exécutable en CI** : pas de jeu, pas d'affichage sur les machines de dev/CI. Les tests
  ne couvrent que la logique du wrapper via `FakeTMEnv` ; le fichier est exclu de la couverture.
- Le LIDAR simulé exige des pistes à bordures noires ; les pistes standard nécessitent
  l'interface caméra (non couverte ici).
- L'entraînement dépend du temps réel : ~500 000 pas ≈ plusieurs heures de jeu effectif.
- Reproductibilité limitée : le jeu n'est pas déterministe et `--seed` ne fixe que SAC.

## Références

- tmrl : <https://github.com/trackmania-rl/tmrl>
- Real-Time Gym (rtgym) : <https://github.com/yannbouteiller/rtgym>
- Stable-Baselines3 SAC : <https://stable-baselines3.readthedocs.io/en/master/modules/sac.html>
- Soft Actor-Critic (Haarnoja et al., 2018) : <https://arxiv.org/abs/1801.01290>
- OpenPlanet : <https://openplanet.dev/>
