# ═══════════════════════════════════════════════════════════════════════════
#  Taxi Driver — Makefile de démonstration (soutenance T-AIA-902)
#
#  Avant la soutenance (une fois, avec réseau) :
#      make check          checklist complète de l'environnement
#      make slides         exporte docs/SLIDES.html (télécharge marp-cli)
#      make trouve-seed    choisir une graine de rejeu courte puis figer SEED
#      make demo PAUSE=0   répétition chronométrée (~80 s de calcul)
#
#  Le jour J :
#      make demo           démo guidée complète (~5 min avec narration)
#      make demo-multi | demo-compare | demo-figures | demo-trackmania
#
#  Chaque acte est relançable seul (make demo-rejeu, make demo-deep, …).
#  La démo TrackMania tourne CÔTÉ WINDOWS : voir make demo-trackmania.
# ═══════════════════════════════════════════════════════════════════════════

SHELL         := /bin/bash
.SHELLFLAGS   := -eu -o pipefail -c
MAKEFLAGS     += --no-print-directory
.DEFAULT_GOAL := help
.NOTPARALLEL:

# ---- Interpréteur : python système par défaut (opérationnel sur cette machine).
# ---- Basculer vers Poetry : make demo PY="poetry run python"
PY   ?= python3
TAXI ?= $(PY) -m src.main

# ---- Paramètres de démo (surcharger : make demo SEED=7 PAUSE=0) ------------
DEMO_TMP ?= /tmp/taxi-demo
SEED     ?= 42
EPISODES ?= 2
DELAY    ?= 0.3
TIME     ?= 30
PAUSE    ?= 1
DEVICE   ?= cpu

# eval/play n'ont pas d'option --device : on masque le GPU via l'environnement
# (le DQN de Taxi est minuscule — le CPU évite l'init CUDA et ses warnings).
ifeq ($(DEVICE),cpu)
CUDA_ENV := CUDA_VISIBLE_DEVICES=
else
CUDA_ENV :=
endif

# ---- Artefacts commités dont dépend la démo --------------------------------
MODELS_DIR  := models/final
DEMO_MODELS := $(MODELS_DIR)/q_learning.npz $(MODELS_DIR)/sarsa.npz \
               $(MODELS_DIR)/expected_sarsa.npz $(MODELS_DIR)/double_q_learning.npz \
               $(MODELS_DIR)/monte_carlo.npz $(MODELS_DIR)/dqn.pt

MARP ?= npx -y @marp-team/marp-cli

# ---- Couleurs (chaînes vides si TERM absent ou NO_COLOR défini) ------------
ifndef NO_COLOR
BOLD   := $(shell tput bold    2>/dev/null)
RED    := $(shell tput setaf 1 2>/dev/null)
GREEN  := $(shell tput setaf 2 2>/dev/null)
YELLOW := $(shell tput setaf 3 2>/dev/null)
CYAN   := $(shell tput setaf 6 2>/dev/null)
RESET  := $(shell tput sgr0    2>/dev/null)
endif

LINE := ═══════════════════════════════════════════════════════════════════

# Bannière d'acte — PAS de virgule dans le texte ($(call) découpe dessus).
define banner
printf '\n%s\n%s\n%s\n\n' \
  "$(CYAN)$(BOLD)$(LINE)" \
  "  $(1)" \
  "$(LINE)$(RESET)"
endef

# Pause jury — sautée si PAUSE=0 ou stdin non-TTY (répétition chrono / CI).
define pause
if [ "$(PAUSE)" = "1" ] && [ -t 0 ]; then \
  printf '%s' "$(YELLOW)▶ Appuyez sur Entrée pour l'acte suivant… $(RESET)"; \
  read -r _ || true; \
fi
endef

.PHONY: help install check test _guard \
        demo demo-intro demo-baseline demo-rejeu demo-express \
        demo-time-limited demo-deep demo-repro demo-final \
        demo-multi demo-compare demo-figures demo-trackmania \
        trouve-seed demo-clean slides slides-pdf report

# ---- Garde silencieuse : échoue vite AVANT la démo, jamais pendant ---------
# (make demo passe GUARD_OK=1 à ses sous-makes : un seul contrôle par démo.)
_guard:
ifneq ($(GUARD_OK),1)
	@$(PY) -c 'import gymnasium' >/dev/null 2>&1 || { \
	  printf '%s\n' "$(RED)✘ $(PY) sans dépendances (gymnasium manquant)." \
	    "  Remèdes : make install (Poetry) puis make demo PY=\"poetry run python\"" \
	    "  ou installer les dépendances dans l'interpréteur choisi.$(RESET)"; exit 1; }
	@for f in $(DEMO_MODELS); do \
	  [ -f "$$f" ] || { printf '%s\n' "$(RED)✘ Modèle manquant : $$f (git checkout ?)$(RESET)"; exit 1; }; \
	done
endif

##@ Démo soutenance (chaîne complète ≈ 5 min avec narration)

demo: _guard ## Démo guidée — 6 actes suivant le fil des slides (PAUSE=0 : sans pause)
	@$(MAKE) GUARD_OK=1 demo-intro
	@$(call pause)
	@$(MAKE) GUARD_OK=1 demo-baseline
	@$(call pause)
	-@$(MAKE) GUARD_OK=1 demo-rejeu
	@$(call pause)
	@$(MAKE) GUARD_OK=1 demo-express
	@$(call pause)
	@$(MAKE) GUARD_OK=1 demo-time-limited
	@$(call pause)
	-@$(MAKE) GUARD_OK=1 demo-deep
	@$(call pause)
	@$(MAKE) GUARD_OK=1 demo-repro
	@$(MAKE) GUARD_OK=1 demo-final

##@ Actes individuels (chacun relançable isolément)

demo-intro: _guard ## Acte 0 — annonce du plan de la démo
	@$(call banner,TAXI DRIVER — Démo live · T-AIA-902)
	@echo "  Référence : R* = 8,05 (retour optimal moyen sur Taxi-v3 — étalon du rapport)"
	@echo ""
	@echo "  Acte 1  Baseline aléatoire      — le point zéro (reward ≈ -770)"
	@echo "  Acte 2  Agent entraîné          — rejeu pas à pas (~13 pas par course)"
	@echo "  Acte 3  Entraînement express    — 2 000 épisodes en direct (~2 s)"
	@echo "  Acte 4  Mode time-limited       — budget 30 s · early-stop (slide 15)"
	@echo "  Acte 5  Tabulaire vs DQN        — même optimum · coût ×31,6"
	@echo "  Acte 6  Reproductibilité        — figure et stats régénérées en direct"

demo-baseline: _guard ## Acte 1 — brute force : combien vaut le hasard ? (~5 s)
	@$(call banner,ACTE 1 — Baseline brute force : le point zéro)
	@echo "  À regarder : mean reward ≈ -770 et success rate ≈ 0 % sur 100 épisodes."
	@mkdir -p $(DEMO_TMP)
	$(TAXI) train --agent brute_force --non-interactive --seed $(SEED) \
	    --train-episodes 1 --test-episodes 100 --show-episodes 0 \
	    --save $(DEMO_TMP)/brute_force.npz

demo-rejeu: _guard ## Acte 2 — rejeu pas à pas du Q-Learning entraîné (~10 s)
	@$(call banner,ACTE 2 — L'agent entraîné joue sous vos yeux)
	@echo "  Modèle commité models/final/q_learning.npz — politique gloutonne."
	@echo "  À regarder : ~13 pas par course et retour cumulé ≈ +8."
	$(TAXI) play --model models/final/q_learning.npz \
	    --episodes $(EPISODES) --delay $(DELAY) --seed $(SEED)

demo-express: _guard ## Acte 3 — entraînement from scratch en direct : 2 000 épisodes (~2 s)
	@$(call banner,ACTE 3 — Entraînement from scratch en direct)
	@echo "  Q-Learning · 2 000 épisodes · objectif : s'approcher de R* = 8,05."
	@mkdir -p $(DEMO_TMP)
	$(TAXI) train --agent q_learning --non-interactive --seed $(SEED) \
	    --train-episodes 2000 --test-episodes 20 --show-episodes 0 \
	    --save $(DEMO_TMP)/ql_express.npz

demo-time-limited: _guard ## Acte 4 — LE mode du sujet : budget 30 s + early-stop (slide 15)
	@$(call banner,ACTE 4 — Mode time-limited : budget de $(TIME) s)
	@echo "  Exigence du sujet : apprendre sous contrainte de temps (configs/optimized.yaml)."
	@echo "  À regarder : arrêt anticipé (~14 s) · reward 8,05 = R* · succès 100 %."
	@mkdir -p $(DEMO_TMP)
	$(TAXI) train --mode time-limited --time $(TIME) --non-interactive \
	    --show-episodes 0 --save $(DEMO_TMP)/time_limited.npz

demo-deep: _guard ## Acte 5 — même optimum : tabulaire vs DQN (H6) (~10 s)
	@$(call banner,ACTE 5 — Tabulaire vs Deep : même optimum)
	@echo "  Deux modèles commités évalués sur 100 épisodes identiques (mêmes graines)."
	@echo "  Verdict H6 : rewards équivalents · coût d'entraînement ×31,6 pour le DQN."
	@printf '%s\n' "$(GREEN)-- Q-Learning (table 500×6 · 23 Ko) --$(RESET)"
	$(CUDA_ENV) $(TAXI) eval --model models/final/q_learning.npz --test-episodes 100 --seed $(SEED)
	@printf '%s\n' "$(GREEN)-- DQN (réseau de neurones · 2 Mo) --$(RESET)"
	$(CUDA_ENV) PYTHONWARNINGS=ignore::FutureWarning $(TAXI) eval --model models/final/dqn.pt --test-episodes 100 --seed $(SEED)

# NB : run_stats.py réécrit results/aggregated/stats_h6.csv à l'identique
# (pas d'option --out) — git ne voit aucune modification.
demo-repro: _guard ## Acte 6 — tout est reproductible : figure + test d'hypothèse (~10 s)
	@$(call banner,ACTE 6 — Reproductibilité : figures et statistiques)
	@echo "  La figure F1 du rapport est régénérée depuis les données commitées."
	@mkdir -p $(DEMO_TMP)/figures
	$(PY) scripts/make_figures.py --figures f1 --out $(DEMO_TMP)/figures
	@echo ""
	@echo "  Test d'hypothèse H6 (coût tabulaire vs DQN) recalculé en direct :"
	$(PY) scripts/run_stats.py --families h6

demo-final: ## Rideau — le message à retenir
	@$(call banner,CONCLUSION — Des différences cinétiques · pas asymptotiques)
	@echo "  Q-Learning 8,048 ± 0,006 · R* = 8,05 · brute force ≈ -770"
	@echo "  L'algorithme change la vitesse d'apprentissage — pas le plafond atteint."
	@printf '%s\n' "$(GREEN)  Merci — place aux questions (make demo-multi · demo-compare · demo-figures · demo-trackmania).$(RESET)"

##@ Bonus (pendant les questions du jury)

demo-multi: _guard ## Multi-passagers : 14 400 états — rendu ANSI (~20-30 s)
	@$(call banner,BONUS — Multi-passagers : 14 400 états)
	@echo "  Entraînement court (3 000 épisodes) : l'agent N'EST PAS convergé — c'est le propos."
	@echo "  ×28,8 d'états ⇒ ~150 000 épisodes nécessaires pour l'optimum (H9 · figure F12)."
	@mkdir -p $(DEMO_TMP)
	$(TAXI) train --env multi --agent q_learning --non-interactive --seed $(SEED) \
	    --train-episodes 3000 --test-episodes 20 --show-episodes 0 \
	    --save $(DEMO_TMP)/multi_ql.npz
	@echo ""
	@echo "  Rejeu dans la grille multi (T = taxi · 1/2 = passagers en attente) :"
	$(TAXI) play --model $(DEMO_TMP)/multi_ql.npz --episodes 1 --delay 0.12 --seed $(SEED)

demo-compare: _guard ## Face-à-face express brute force / QL / SARSA avec stats (~1 min)
	@$(call banner,BONUS — Face-à-face statistique en direct)
	@echo "  3 agents × 2 graines · mêmes épisodes de test · tests appariés corrigés (Holm)."
	@echo "  Version express de E2 (le rapport : 10 graines × 15 000 épisodes)."
	@mkdir -p $(DEMO_TMP)
	PYTHONWARNINGS=ignore::FutureWarning $(TAXI) compare --agents brute_force,q_learning,sarsa \
	    --train-episodes 3000 --test-episodes 50 --n-seeds 2 --workers 6 \
	    --out $(DEMO_TMP)/compare

demo-figures: ## Ouvre les figures clés dans VSCode (le GIF F11b s'anime dans l'aperçu)
	@if command -v code >/dev/null 2>&1; then \
	  echo "Ouverture dans VSCode : F1 (apprentissage) · F9 (échantillon) · F12 (multi) · F11b (GIF)"; \
	  code -r results/figures/F1_courbes_apprentissage.png \
	         results/figures/F9_efficacite_echantillon.png \
	         results/figures/F12_multi_passagers.png \
	         results/figures/F11b_episode.gif; \
	else \
	  echo "VSCode introuvable — figures disponibles :"; ls -1 results/figures/; \
	fi

demo-trackmania: ## Démo live TrackMania (CÔTÉ WINDOWS) — runbook + résultats de référence
	@$(call banner,BONUS — TrackMania : SAC en conditions réelles)
	@echo "  Cette démo tourne CÔTÉ WINDOWS (jeu + OpenPlanet + ViGEmBus) :"
	@echo ""
	@echo "  1. Relancer VSCode en mode Windows sur C:\\Coding\\Taxi-Driver"
	@echo "  2. Lancer TrackMania : piste tmrl-test en mode conduite · fenêtré 958×488"
	@echo "     coin haut-gauche · caméra 3 · ghosts masqués · overlay OpenPlanet fermé (F3)"
	@echo "  3. Dans le terminal VSCode Windows :"
	@echo "       powershell -ExecutionPolicy Bypass -File scripts\\demo_trackmania.ps1"
	@echo "     (1 tour ≈ 70-75 s · le script affiche LAP COMPLETED et le temps au tour)"
	@echo ""
	@echo "  Référence commitée (results/trackmania/eval.json) — secours si le live échoue :"
	@$(PY) -c "import json; d=json.load(open('results/trackmania/eval.json')); print(f\"    {d['n_laps']}/{d['n_episodes']} tours complétés · meilleur tour {d['best_lap_time_s']:.2f} s · reward moyen {d['mean_reward']:.1f}\")"

##@ Préparation et entretien

check: ## Checklist d'avant-soutenance (déps · modèles · figures · outils)
	@printf '%s\n' "$(BOLD)Vérification de l'environnement de soutenance$(RESET)"
	@ok=1; \
	if $(PY) -c 'import gymnasium, numpy, torch' >/dev/null 2>&1; then \
	  printf '  %s %s\n' "$(GREEN)✔$(RESET)" "dépendances Python OK ($$($(PY) --version 2>&1) · gymnasium · numpy · torch)"; \
	else \
	  printf '  %s\n' "$(RED)✘ dépendances manquantes pour $(PY) → make install puis PY=\"poetry run python\"$(RESET)"; ok=0; \
	fi; \
	for f in $(DEMO_MODELS) results/r_star.json results/figures/F11b_episode.gif docs/SLIDES.md; do \
	  if [ -f "$$f" ]; then printf '  %s %s\n' "$(GREEN)✔$(RESET)" "$$f"; \
	  else printf '  %s\n' "$(RED)✘ $$f manquant (git checkout ?)$(RESET)"; ok=0; fi; \
	done; \
	if command -v code >/dev/null 2>&1; then printf '  %s CLI VSCode (make demo-figures)\n' "$(GREEN)✔$(RESET)"; \
	else printf '  %s\n' "$(YELLOW)– CLI VSCode absente : make demo-figures listera les chemins$(RESET)"; fi; \
	if command -v npx >/dev/null 2>&1; then \
	  printf '  %s npx présent — lancez « make slides » une fois AVANT la soutenance (téléchargement)\n' "$(GREEN)✔$(RESET)"; \
	else printf '  %s\n' "$(YELLOW)– npx absent : make slides indisponible (utiliser l'extension Marp de VSCode)$(RESET)"; fi; \
	if [ $$ok -eq 1 ]; then printf '\n%s\n' "$(GREEN)$(BOLD)✔ Tout est prêt. Répétez avec : make demo PAUSE=0$(RESET)"; \
	else printf '\n%s\n' "$(RED)$(BOLD)✘ Corrigez les points ci-dessus avant la soutenance.$(RESET)"; exit 1; fi

trouve-seed: _guard ## Répétition : scanne des graines pour des rejeux courts (figer SEED= ensuite)
	@echo "Recherche d'une graine donnant $(EPISODES) rejeux courts (~13 pas) :"
	@for s in 1 2 3 5 7 11 13 21 42 123; do \
	  printf '%s\n' "$(BOLD)-- seed $$s$(RESET)"; \
	  $(TAXI) play --model models/final/q_learning.npz \
	      --episodes $(EPISODES) --delay 0 --seed $$s 2>/dev/null \
	    | grep -E "delivered|truncated" || true; \
	done
	@echo ""
	@echo "Choisir une graine aux courses courtes puis : make demo SEED=<s> (ou figer SEED= dans ce Makefile)."

install: ## Installe l'environnement Poetry (voie officielle du README — télécharge torch)
	poetry install
	@printf '%s\n' "$(GREEN)✔ Environnement installé. Utilisez : make demo PY=\"poetry run python\"$(RESET)"

test: _guard ## Tests rapides du projet (sous-ensemble « not slow »)
	$(PY) -m pytest

demo-clean: ## Supprime les artefacts de démo (/tmp/taxi-demo) — ne touche à rien de commité
	rm -rf $(DEMO_TMP)
	@printf '%s\n' "$(GREEN)✔ $(DEMO_TMP) supprimé (models/final et results/ intacts).$(RESET)"

##@ Supports de soutenance

slides: ## Exporte docs/SLIDES.md → docs/SLIDES.html (npx requis · téléchargement au 1er lancement)
	@command -v npx >/dev/null 2>&1 || { printf '%s\n' \
	  "$(RED)✘ npx introuvable — installez Node.js ou utilisez l'extension VSCode « Marp for VS Code ».$(RESET)"; exit 1; }
	$(MARP) --html docs/SLIDES.md -o docs/SLIDES.html
	@printf '%s\n' "$(GREEN)✔ docs/SLIDES.html généré (figures liées en relatif — ouvrir dans un navigateur).$(RESET)"

slides-pdf: ## Exporte docs/SLIDES.md → docs/SLIDES.pdf (Chrome requis — celui de Windows marche sous WSL)
	@command -v npx >/dev/null 2>&1 || { printf '%s\n' "$(RED)✘ npx introuvable.$(RESET)"; exit 1; }
	$(MARP) --pdf --allow-local-files docs/SLIDES.md -o docs/SLIDES.pdf
	@printf '%s\n' "$(GREEN)✔ docs/SLIDES.pdf généré. Si échec Chrome : CHROME_PATH='/mnt/c/Program Files/Google/Chrome/Application/chrome.exe' make slides-pdf$(RESET)"

report: ## Ouvre le rapport scientifique commité (docs/RAPPORT.pdf)
	@test -f docs/RAPPORT.pdf || { printf '%s\n' "$(RED)✘ docs/RAPPORT.pdf manquant.$(RESET)"; exit 1; }
	@if command -v code >/dev/null 2>&1; then code docs/RAPPORT.pdf; \
	elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$$(wslpath -w docs/RAPPORT.pdf)" || true; \
	else printf '%s\n' "→ docs/RAPPORT.pdf"; fi

##@ Aide

help: ## Affiche cette aide (cible par défaut)
	@awk 'BEGIN {FS = ":.*## "; printf "\nUsage : make $(CYAN)<cible>$(RESET)   (ex : make demo · make demo PAUSE=0 · make demo-rejeu SEED=7)\n"} \
	  /^[a-zA-Z0-9_-]+:.*## / { printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2 } \
	  /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
