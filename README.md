# Survival-Analysis-Oncologie
Dans ce projet, la Survival Analysis est utilisée pour analyser le temps de survie des patientes atteintes d’un cancer du sein, en tenant compte de la censure des données. L’objectif est d’identifier les facteurs influençant la survie et de construire des modèles prédictifs pour estimer le pronostic des patientes.

## Interface Gradio (RSF + SHAP)

Une interface interactive est disponible dans `gradio_app.py` avec:

- formulaire patient: age, T Stage, grade, taille tumorale, status estrogen/progesterone
- output 1: courbe de survie personnalisee
- output 2: score de risque a 5 ans
- output 3: graphique SHAP explicatif (waterfall)

### Lancement local

```bash
pip install -r requirements.txt
python gradio_app.py
```

Puis ouvrir:

```text
http://127.0.0.1:7860
```

### Deploiement rapide (Hugging Face Spaces)

1. Creer un Space Gradio.
2. Pousser ces fichiers dans le repo du Space:
	- `gradio_app.py`
	- `requirements.txt`
	- `seer_cancer.csv`
3. Si necessaire, renommer le script en `app.py` (ou configurer la commande de lancement du Space).

### Note clinique

Le formulaire est volontairement compact. Les variables non saisies (ex: N Stage, nodes examined/positive, A Stage) sont renseignees automatiquement avec des valeurs de reference issues de la cohorte d'entrainement.
