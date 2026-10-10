# DermaSense — PFA IAD 2025/2026

## Contenu
- `Rapport_PFA_DermaSense.docx` : rapport complet (3 volets)
- `dermasense_pipeline.py` : pipeline IA (EDA, modèles, évaluation)
- `dermasense_app.html` : application web (login/signup + recommandations)
- `cosmetics.csv` : dataset (1 472 produits Sephora)
- `figures/` : graphiques du rapport
- `clients_simules.csv`, `interactions_simulees.csv`, `results.json` : données générées

## Installation (Windows)
1. Installer Python 3.10+ depuis python.org (cocher "Add Python to PATH").
2. Ouvrir le dossier dans VS Code (File > Open Folder).
3. Dans le terminal intégré (Ctrl + ù) :
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

## Exécution
- Pipeline IA : `python dermasense_pipeline.py`
  (métriques dans le terminal, graphiques régénérés dans figures/)
- Application web : double-clic sur `dermasense_app.html`
  Compte démo : demo@dermasense.ma / demo123

## Reproductibilité
Seed fixée à 42. Les métriques peuvent varier de ±0,01-0,03 selon la version
de scikit-learn.
