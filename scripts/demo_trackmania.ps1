# =============================================================================
#  Démo TrackMania — soutenance T-AIA-902 (machine de jeu Windows uniquement)
#
#  À lancer depuis la racine du checkout Windows (ex. C:\Coding\Taxi-Driver),
#  dans un terminal VSCode « mode Windows » :
#      powershell -ExecutionPolicy Bypass -File scripts\demo_trackmania.ps1
#      powershell -ExecutionPolicy Bypass -File scripts\demo_trackmania.ps1 -Episodes 3
#
#  Prérequis (détail : docs/TRACKMANIA.md) : jeu lancé sur la piste tmrl-test
#  en mode conduite, fenêtré 958x488 coin haut-gauche, caméra 3, overlay fermé.
# =============================================================================
param(
    [int]$Episodes = 1,
    [string]$Model = "models/trackmania/sac_trackmania_final.zip"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "poetry introuvable sur le PATH — installer Poetry ou activer la venv du projet." -ForegroundColor Red
    exit 1
}

# Repli sur la copie commitée du modèle final si le dossier local n'existe pas
if (-not (Test-Path $Model)) {
    $fallback = "models/final/sac_trackmania_final.zip"
    if (Test-Path $fallback) {
        $Model = $fallback
    } else {
        Write-Host "Modèle SAC introuvable : $Model" -ForegroundColor Red
        exit 1
    }
}

$line = "=" * 74
Write-Host ""
Write-Host $line -ForegroundColor Cyan
Write-Host "  DÉMO TRACKMANIA — SAC final : $Episodes tour(s) jusqu'à l'arrivée" -ForegroundColor Cyan
Write-Host $line -ForegroundColor Cyan
Write-Host ""
Write-Host "Checklist avant lancement :" -ForegroundColor Yellow
Write-Host "  1. TrackMania 2020 lancé, piste tmrl-test ouverte en mode conduite (drapeau vert)"
Write-Host "  2. Fenêtre 958x488, coin haut-gauche, au premier plan (repositionnée automatiquement)"
Write-Host "  3. Caméra 3 (voiture masquée), ghosts masqués"
Write-Host "  4. Overlay OpenPlanet fermé (F3), plugin TMRL Grab Data (re)chargé"
Write-Host "  5. Plan d'alimentation « Performances élevées », aucune application lourde"
Write-Host ""
Read-Host "Appuyez sur Entrée pour lancer (1 tour = environ 70-75 s de conduite autonome)"

# --output explicite : ne PAS écraser le results/trackmania/eval.json commité
poetry run python scripts/eval_trackmania.py `
    --model $Model `
    --episodes $Episodes `
    --output results/demo_trackmania_eval.json

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Échec de l'évaluation — voir docs/TRACKMANIA.md (section Dépannage)." -ForegroundColor Red
    Write-Host "Secours : montrer results/trackmania/eval.json (9/10 tours, meilleur 61,15 s)." -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Attendu ci-dessus : [LAP COMPLETED] par épisode et le meilleur temps au tour." -ForegroundColor Green
