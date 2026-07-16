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
| Jeu | TrackMania 2020 (Steam/Epic/Ubisoft) + compte Ubisoft Connect |
| **Club Access** | **Requis en pratique** : le plugin `TMRL_GrabData` de tmrl n'est pas signé, et OpenPlanet n'autorise le mode signature « Developer » (nécessaire aux plugins non signés) qu'avec Club Access (~20 €/an). La mention « édition Standard suffisante » des docs tmrl est obsolète. |
| Plugin | [OpenPlanet](https://openplanet.dev/) (nécessite le runtime VC++ x64) |
| Manette virtuelle | Pilote **ViGEmBus** — installé par la dépendance `vgamepad` de tmrl (invite UAC). En cas de versions ViGEmBus multiples préexistantes (ex. HP Omen Gaming Hub embarque la sienne), nettoyer et installer la 1.22 officielle, sinon `vigem_target_add` échoue. |
| Python | Python 3.10 (la venv Poetry du projet) avec `pip install tmrl==0.7.1` |
| Piste | Piste à **bordures sombres** : le LIDAR simulé de tmrl détecte les bords par seuil de luminosité sur la capture d'écran. Les cartes `tmrl-test`/`tmrl-train` fournies dans `TmrlData\resources` conviennent (fichier de récompense fourni). |
| Affichage | Jeu **fenêtré 958×488, coin haut-gauche** (capture calibrée), **caméra 3** (voiture masquée), overlay OpenPlanet **fermé** (F3), ghosts masqués |

## Installation (ordre validé)

### Côté dépôt (une fois)

```bash
poetry install -E trackmania    # SB3 + tensorboard + rich
poetry run pip install tmrl==0.7.1
# Vérifier que les pins du projet ont survécu (numpy<2, gymnasium<1.3) :
poetry run pip check
# Si opencv-python 5.x a tiré numpy 2.x :
poetry run pip install "opencv-python==4.11.0.86" "numpy==1.26.4"
```

Le premier `import tmrl` crée `%USERPROFILE%\TmrlData\` (config, cartes, récompenses,
plugins, poids pré-entraînés). Appliquer le préréglage LIDAR :

```bash
python scripts/check_trackmania_setup.py --apply-config   # depuis configs/tmrl_config.json
```

### Côté machine de jeu (une fois, ~1 h)

1. Installer TrackMania (2020) via Steam, se connecter à Ubisoft Connect, lancer une fois,
   passer en mode **fenêtré**. Acheter **Club Access** (cf. prérequis).
2. Installer le runtime VC++ x64 puis **OpenPlanet pour Trackmania** ; relancer le jeu une fois
   (crée `%USERPROFILE%\OpenplanetNext`).
3. Copier les plugins et les cartes :
   - `TmrlData\resources\Plugins\TMRL_GrabData.op` → `OpenplanetNext\Plugins\`
   - `TmrlData\resources\tmrl-test.Map.Gbx` → `Documents\Trackmania\Maps\My Maps\`
4. En jeu : Créer → Éditeur de circuit → Modifier un circuit → My Maps → `tmrl-test`, drapeau
   vert (mode conduite). **Caméra 3**, ghosts masqués.
5. Overlay OpenPlanet (F3) : Developer → Signature mode → **Developer**, puis
   Developer → (Re)load plugin → **TMRL Grab Data**. Refermer l'overlay (F3).
6. Valider : `python scripts/check_trackmania_setup.py --steps 400` — la fenêtre est
   repositionnée automatiquement, un agent aléatoire conduit ~30 s, le script vérifie espaces,
   timing 20 Hz (violations rtgym), récompenses et observations, et écrit
   `results/trackmania/check_setup.json`. Sortie 0 = prêt à entraîner.

## Usage

```bash
# Entraînement (hyperparamètres : configs/trackmania.yaml, surchargés par CLI)
python scripts/train_trackmania.py --config configs/trackmania.yaml \
    --timesteps 500000 --seed 42 --run-name sac_500k_seed42

# Reprise après interruption/crash (--timesteps = pas SUPPLÉMENTAIRES)
python scripts/train_trackmania.py --resume --timesteps 200000

# Évaluation greedy + détection de tours complétés
python scripts/eval_trackmania.py --model models/trackmania/sac_trackmania_final.zip --episodes 10

# Suivi en direct
poetry run tensorboard --logdir results/trackmania/tb
```

Checkpoints SAC (~4 Mo) tous les 25 000 pas dans `models/trackmania/`, modèle final
`sac_trackmania_final.zip`, sauvegarde d'interruption `sac_trackmania_last.zip` (Ctrl+C sûr),
instantané du replay buffer (~700 Mo, un seul fichier tournant) tous les 100 000 pas.
Épisodes journalisés par `Monitor` SB3 dans `results/trackmania/monitor.csv` (append en mode
`--resume`) ; hyperparamètres résolus dans `models/trackmania/run_config.json`.

## Architecture

```
TrackMania 2020 + OpenPlanet TMRL_GrabData (télémétrie, socket localhost:9000)
        │            + capture d'écran 958×488 (LIDAR 19 rayons) + ViGEmBus (manette virtuelle)
        ▼
tmrl.get_environment()            rtgym temps réel 20 Hz — obs tuple ((1,), (4,19), (3,), (3,))
        │
        ▼
TMEnvProtocol                     interface structurelle (testable avec un faux env)
        │
        ▼
TrackManiaEnvWrapper              validation des espaces (échec précoce si mauvais préréglage),
        │                         aplatissement + normalisation → Box(-1, 1, (83,)),
        │                         wait()/close(), retry sur échec transitoire ViGEmBus
        ▼
Stable-Baselines3 SAC             MlpPolicy [256, 256], replay 1M, checkpoints + reprise
```

Le wrapper (`src/environments/trackmania_wrapper.py`) ne dépend que du **protocole**, jamais de
`tmrl` directement : `import tmrl` n'a lieu que dans `make_trackmania_env()`. Les tests unitaires
(`tests/environments/test_trackmania_wrapper.py`) utilisent un `FakeTMEnv` en mémoire.

## Observations, actions, récompense

Observation aplatie — vecteur `float32` de forme `(83,)` :

| Indices | Contenu | Normalisation |
| --- | --- | --- |
| `0` | Vitesse | `v / max_speed` (1000 par défaut), clip `[0, 1]` |
| `1–76` | 4 scans LIDAR empilés × 19 faisceaux | `d / max_lidar` (400 par défaut), clip `[0, 1]` |
| `77–79` | Action précédente | inchangée, dans `[-1, 1]` |
| `80–82` | Action précédent-précédente | inchangée, dans `[-1, 1]` |

Les deux dernières actions font partie de l'observation car l'environnement temps réel introduit
un délai action→effet : les inclure rend le processus à nouveau markovien.

Action — `Box(-1, 1, (3,))` : `[gaz, frein, direction]`, clippée avant transmission au jeu.

Récompense — **progression le long d'une trajectoire de démonstration** enregistrée par le
plugin (fichier `TmrlData\reward\reward.pkl`, fourni pour les cartes tmrl ; à enregistrer soi-même
pour une carte personnalisée). La trajectoire de `tmrl-test` compte 30 273 points, soit un
total de progression de ~302,7 (1 point = 0,01) ; le **franchissement de la ligne d'arrivée**
est signalé par le plugin lui-même (télémétrie du jeu) et vaut un bonus `END_OF_TRACK` de
**+100** sur le dernier pas, avec `terminated`. Autres fins d'épisode : `terminated` par
coupure `FAILURE_COUNTDOWN` (aucune progression, pas de bonus) ; `truncated` = plafond
`ep_max_length`. Nous avons relevé ce plafond de 1000 à **2000 pas** (50 s → 100 s) : à 50 s,
une politique en cours d'apprentissage n'atteint que ~60 % de la piste et **ne voit jamais ni
la fin du circuit ni le bonus** — la coupure anti-stagnation suffit à éliminer les épisodes
improductifs. Un **tour complété** se détecte par `terminated` avec récompense finale ≥ 50
(seuil `--lap-bonus-threshold` d'`eval_trackmania.py`) ; le dict `info` de tmrl est vide.
Attention à l'interprétation de `monitor.csv` : une récompense d'épisode ≥ 100 n'implique
**pas** un tour complété (la progression seule peut dépasser 100) ; seul le bonus final fait foi.

## Hyperparamètres (configs/trackmania.yaml)

Alignés sur le pipeline SAC de référence de tmrl (celui qui complète des tours) : lr 3e-5
(compromis entre les 1e-5/5e-5 acteur/critique de tmrl — SB3 n'a qu'un lr), `ent_coef` **fixe**
à 0,01 (tmrl n'apprend pas α ; l'« auto » de SB3 viserait une entropie cible inadaptée),
γ 0,995, τ 0,005, batch 256, replay 1M, `learning_starts` 5000 (~4 min d'exploration pure).
Cadence : 1 pas de gradient CPU (~2-5 ms) par pas d'environnement — validée sans violation de
timing rtgym sur la machine de jeu ; repli `--train-freq-episode` (rafales entre épisodes,
`wait_on_done` immobilise la voiture) si une autre machine n'y arrive pas.

## Pourquoi SAC plutôt que PPO ?

- **Efficacité d'échantillonnage** : l'environnement est temps réel et mono-instance
  (impossible de vectoriser ou d'accélérer le jeu). SAC est *off-policy* : chaque transition
  est rejouée de nombreuses fois depuis le replay buffer, alors que PPO (*on-policy*) jette
  ses données après quelques epochs.
- **Actions continues** : SAC est conçu pour les espaces d'actions continus bornés.
- **Exploration** : le bonus d'entropie de SAC maintient une exploration stochastique utile
  sur des pistes longues, sans schéma d'exploration manuel.
- **Référence amont** : le pipeline d'entraînement de référence de tmrl est lui-même basé sur
  SAC, ce qui rend nos résultats comparables (politique tmrl LIDAR : ~45,5 s sur `tmrl-test`).

## Dépannage

| Symptôme | Cause probable / correctif |
| --- | --- |
| `OpenPlanet stopped sending data since more than 10.0s` | Le plugin n'émet qu'en mode conduite : être **dans la voiture** ; recharger le plugin (F3 → Developer → (Re)load) après chaque redémarrage du jeu ; ne pas recharger le plugin pendant qu'un script tourne. |
| `The virtual device could not connect to ViGEmBus` persistant | Versions ViGEmBus multiples/anciennes (HP Omen…) : nettoyer les périphériques dupliqués, installer ViGEmBus 1.22, redémarrer. Les échecs *transitoires* sont réessayés automatiquement par le wrapper. |
| LIDAR à 0 / distances incohérentes | Mauvaise caméra (il faut la **3**, voiture masquée), overlay OpenPlanet ouvert (F3 pour fermer), fenêtre déplacée/recouverte, notification par-dessus le jeu. |
| Récompense plate à l'entraînement | Vérifier DANS L'ORDRE : fenêtre 958×488 visible et **au premier plan** ; fichier de récompense correspondant à la carte (`--check-environment`) ; violations de timing (TensorBoard `time/fps`) ; la voiture répond (ViGEmBus) ; plugin chargé. |
| `ValueError … RTGYM_INTERFACE` à la construction | La config TmrlData n'est pas sur le préréglage LIDAR : `python scripts/check_trackmania_setup.py --apply-config`. |
| Timing dégradé (violations rtgym) | Plan d'alimentation **Performances élevées**, secteur branché, fermer les applications lourdes, jamais de RDP/verrouillage pendant un run. |

## Limites

- **Non exécutable en CI** : pas de jeu, pas d'affichage sur les machines de dev/CI. Les tests
  ne couvrent que la logique du wrapper via `FakeTMEnv` ; le fichier est exclu de la couverture.
- Le LIDAR simulé exige des pistes à bordures sombres ; les pistes standard nécessitent
  l'interface caméra (non couverte ici).
- L'entraînement dépend du temps réel : 500 000 pas ≈ 8-10 h de jeu effectif (resets compris),
  machine dédiée (le jeu doit garder le focus).
- Reproductibilité limitée : le jeu n'est pas déterministe et `--seed` ne fixe que SAC.

## Références

- tmrl : <https://github.com/trackmania-rl/tmrl>
- Real-Time Gym (rtgym) : <https://github.com/yannbouteiller/rtgym>
- Stable-Baselines3 SAC : <https://stable-baselines3.readthedocs.io/en/master/modules/sac.html>
- Soft Actor-Critic (Haarnoja et al., 2018) : <https://arxiv.org/abs/1801.01290>
- OpenPlanet : <https://openplanet.dev/>
