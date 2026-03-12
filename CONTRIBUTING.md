# Contributing — Taxi Driver

Merci de contribuer au projet Taxi Driver ! Ce guide décrit les conventions et le processus à suivre pour garantir un développement fluide et une base de code de qualité.

## Table des matières

- [Prérequis](#prérequis)
- [Installation de l'environnement](#installation-de-lenvironnement)
- [Workflow Git](#workflow-git)
- [Conventions de commits](#conventions-de-commits)
- [Conventions de code](#conventions-de-code)
- [Tests](#tests)
- [Pull Requests](#pull-requests)
- [CI/CD](#cicd)

---

## Prérequis

- Python 3.10+
- [Poetry](https://python-poetry.org/) pour la gestion des dépendances
- Git

## Installation de l'environnement

```bash
git clone https://github.com/T-AIA-902-Taxi-Driver/Taxi-Driver.git
cd Taxi-Driver
poetry install
poetry shell
```

## Workflow Git

Le projet suit un **Git Flow simplifié** :

| Branche | Rôle |
|---|---|
| `main` | Code stable, releases uniquement |
| `dev` | Branche d'intégration, base pour les features |
| `feature/<nom>` | Nouvelles fonctionnalités |
| `fix/<nom>` | Corrections de bugs |

### Règles

1. **Toujours brancher depuis `dev`**, jamais depuis `main`
2. **Jamais de push direct** sur `main` ou `dev` — uniquement via Pull Request
3. **Supprimer la branche** après le merge

```bash
# Créer une feature
git checkout dev
git pull origin dev
git checkout -b feature/mon-agent

# Travailler, commiter, puis pousser
git push -u origin feature/mon-agent
```

## Conventions de commits

Le projet utilise les [Conventional Commits](https://www.conventionalcommits.org/) :

```
type(scope): description courte
```

### Types autorisés

| Type | Usage |
|---|---|
| `feat` | Nouvelle fonctionnalité |
| `fix` | Correction de bug |
| `docs` | Documentation uniquement |
| `style` | Formatage, pas de changement de logique |
| `refactor` | Refactoring sans changement de comportement |
| `test` | Ajout ou modification de tests |
| `chore` | Maintenance, CI, dépendances |
| `perf` | Amélioration de performance |

### Scopes courants

`agents`, `env`, `training`, `eval`, `bench`, `viz`, `cli`, `config`, `ci`

### Exemples

```
feat(agents): add Q-Learning agent with epsilon-greedy policy
fix(env): correct state decoding for passenger location
docs(readme): add usage examples for timed mode
test(agents): add unit tests for BruteForceAgent
chore(ci): add mypy to GitHub Actions pipeline
```

## Conventions de code

### Style

- **PEP 8** avec une longueur de ligne maximale de **100 caractères**
- **Type hints** obligatoires sur toutes les signatures de fonctions
- **Docstrings** au format Google pour les classes et fonctions publiques

```python
def train(self, episodes: int, verbose: bool = False) -> TrainingHistory:
    """Entraîne l'agent sur un nombre donné d'épisodes.

    Args:
        episodes: Nombre d'épisodes d'entraînement.
        verbose: Afficher les logs détaillés.

    Returns:
        Historique contenant rewards, steps et métriques.
    """
```

### Outils de qualité

| Outil | Rôle | Commande |
|---|---|---|
| **ruff** | Linting | `poetry run ruff check .` |
| **black** | Formatage | `poetry run black .` |
| **mypy** | Typage statique | `poetry run mypy src/` |
| **pytest** | Tests | `poetry run pytest` |

Lancer tous les checks en une commande :

```bash
poetry run ruff check . && poetry run black --check . && poetry run mypy src/ && poetry run pytest
```

### Structure des imports

Ordre des imports (appliqué par ruff) :

1. Standard library (`os`, `typing`, `pathlib`)
2. Third-party (`gymnasium`, `numpy`, `torch`)
3. Local (`src.agents`, `src.environments`)

## Tests

### Lancer les tests

```bash
# Tous les tests
poetry run pytest

# Avec couverture
poetry run pytest --cov=src

# Tests unitaires uniquement (exclure les tests lents)
poetry run pytest -m "not slow"
```

### Conventions de tests

- Un fichier de test par module : `tests/test_<module>.py`
- Nommer les fonctions : `test_<methode>_<scenario>_<resultat_attendu>`
- Utiliser des fixtures pytest pour les objets partagés (agents, environnements)
- Marquer les tests lents avec `@pytest.mark.slow`

```python
def test_q_learning_agent_choose_action_returns_valid_action():
    agent = QLearningAgent(state_size=500, action_size=6)
    action = agent.choose_action(state=0)
    assert 0 <= action < 6
```

## Pull Requests

### Processus

1. Créer la branche `feature/<nom>` ou `fix/<nom>` depuis `dev`
2. Développer, commiter avec les conventions
3. Pousser la branche et ouvrir une PR vers `dev`
4. Attendre la review d'au moins **1 membre** de l'équipe
5. Corriger les retours si nécessaire
6. **Squash merge** après approbation

### Template PR

```markdown
## Description
<!-- Résumé des changements -->

## Type de changement
- [ ] Nouvelle fonctionnalité (feat)
- [ ] Correction de bug (fix)
- [ ] Refactoring
- [ ] Documentation
- [ ] Tests
- [ ] CI/CD

## Checklist
- [ ] Le code suit les conventions (PEP 8, type hints)
- [ ] Les tests passent (`poetry run pytest`)
- [ ] Le linting passe (`poetry run ruff check .`)
- [ ] Le formatage est correct (`poetry run black --check .`)
- [ ] La documentation est mise à jour si nécessaire
- [ ] Les nouveaux fichiers ont des docstrings
```

## CI/CD

Le pipeline GitHub Actions s'exécute automatiquement sur chaque push vers `dev` et sur chaque PR vers `main`. Il vérifie :

1. **Linting** — `ruff check` + `black --check`
2. **Typage** — `mypy src/`
3. **Tests** — `pytest` avec rapport de couverture
4. **Smoke test** — entraînement rapide de 10 épisodes

La PR ne peut être mergée que si **tous les checks passent**.

---

Des questions ? Ouvre une issue ou contacte l'équipe sur le canal de communication du projet.
