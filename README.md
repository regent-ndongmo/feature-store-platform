# 🗄️ Feature Store Platform — Tkinter

Interface graphique locale pour le prétraitement de données ML et la gestion d'un Feature Store, avec API REST intégrée.

---

## 📁 Structure du projet

```
feature_store_app/
├── app.py                    ← Point d'entrée (fenêtre Tkinter)
├── requirements.txt
├── core/
│   ├── cleaner.py            ← Moteur de nettoyage des données
│   └── engineer.py           ← Moteur de feature engineering
├── store/
│   └── feature_store.py      ← Backend Feature Store (SQLite + Parquet)
├── api/
│   └── server.py             ← Serveur API REST (HTTP, port 5050)
├── store/data/               ← Données persistées (créé automatiquement)
│   ├── feature_store.db      ← Métadonnées SQLite
│   └── *.parquet             ← Features sauvegardées
└── tests/
    └── test_all.py           ← Suite de tests complète
```

---

## ⚙️ Prérequis

- **Python 3.10+**
- `tkinter` (inclus avec Python sur Windows et Linux ; sur macOS : `brew install python-tk`)
- Les dépendances listées dans `requirements.txt`

---

## 🚀 Installation

```bash
# 1. Cloner / décompresser le projet
cd feature_store_app

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv

# Activer sur Windows :
.venv\Scripts\activate

# Activer sur Linux / macOS :
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## ▶️ Lancer l'application

```bash
# Depuis le dossier feature_store_app/
python app.py
```

La fenêtre s'ouvre immédiatement. L'API REST démarre automatiquement en arrière-plan sur **http://localhost:5050**.

---

## 🖥️ Guide d'utilisation (étapes dans l'interface)

### Onglet ① — Chargement
- Cliquez **"Charger CSV / Excel / Parquet"** pour charger votre propre fichier.
- Ou cliquez **"Charger données véhicules (exemple)"** pour générer un dataset de démonstration instantanément.
- Le tableau de droite affiche un aperçu, le panneau de gauche affiche les types et NaN par colonne.

### Onglet ② — Nettoyage
1. Choisissez une **action** dans le menu déroulant (ex : `fill_median`).
2. Sélectionnez la **colonne** cible.
3. Pour `fill_value`, entrez la valeur de remplacement.
4. Cliquez **"➕ Ajouter règle"** — la règle apparaît dans la queue.
5. Répétez pour enchaîner autant de règles que nécessaire.
6. Cliquez **"▶ Appliquer le nettoyage"** — le log s'affiche, l'aperçu se met à jour.

**Actions disponibles :**
| Action | Description |
|---|---|
| `drop_duplicates` | Supprime les lignes dupliquées |
| `drop_all_na_rows` | Supprime les lignes entièrement vides |
| `drop_column` | Supprime une colonne |
| `drop_na_rows` | Supprime les lignes avec NaN dans une colonne |
| `fill_mean` | Remplace NaN par la moyenne |
| `fill_median` | Remplace NaN par la médiane |
| `fill_mode` | Remplace NaN par le mode |
| `fill_value` | Remplace NaN par une valeur fixe |
| `clip_outliers` | Écrête les outliers (percentiles 1%-99%) |
| `cast_numeric` | Convertit une colonne en numérique |
| `lowercase_str` | Met une colonne texte en minuscules |

### Onglet ③ — Feature Engineering
1. Choisissez une **action de transformation**.
2. Sélectionnez la/les **colonne(s)** source.
3. Donnez un **nom** à la nouvelle feature (ex : `log_price`).
4. Si besoin, entrez un **paramètre** (seuil pour `binarize`, nombre de bins pour `bin`, expression pour `custom_expr`).
5. Cliquez **"➕ Ajouter transformation"**, puis **"▶ Appliquer"**.

**Transformations disponibles :**
| Action | Description |
|---|---|
| `log_transform` | log1p(colonne) |
| `normalize` | Normalisation min-max [0, 1] |
| `standardize` | Z-score (moyenne 0, écart-type 1) |
| `binarize` | 0/1 selon un seuil numérique |
| `label_encode` | Encodage catégoriel en entiers |
| `bin` | Découpage en N intervalles |
| `ratio` | col1 / col2 |
| `difference` | col1 − col2 |
| `product` | col1 × col2 |
| `age_from_year` | année_actuelle − colonne_année |
| `custom_expr` | Expression pandas eval() |

### Onglet ④ — Feature Store
1. Entrez un **nom** pour le groupe de features (ex : `vehicle_features_v1`).
2. Ajoutez une **description** (optionnel).
3. Sélectionnez la **colonne entité** (identifiant unique des lignes, ex : `vehicle_id`).
4. Dans la liste de droite, sélectionnez les **features à sauvegarder** (Ctrl+clic pour multi-sélection).
5. Cliquez **"💾 Sauvegarder dans le Store"**.
6. La liste des groupes se met à jour. Cliquez sur un groupe pour voir ses métadonnées et un aperçu.

Les données sont persistées dans `store/data/` (Parquet + SQLite) et survivent au redémarrage de l'application.

### Onglet ⑤ — API / Endpoint
L'API REST démarre automatiquement. Vous pouvez :
- Cliquer **"📡 GET /list"** pour lister les groupes.
- Sélectionner un groupe, entrer des features et IDs, puis cliquer **"📥 POST /fetch"** pour récupérer des données.
- Cliquer **"💊 GET /health"** pour vérifier l'état du serveur.

---

## 🌐 Utiliser l'API depuis votre code

L'API tourne sur **http://localhost:5050** pendant que l'application est ouverte.

### Lister les groupes
```bash
curl http://localhost:5050/api/feature-groups
```

### Voir les métadonnées d'un groupe
```bash
curl http://localhost:5050/api/feature-groups/vehicle_features_v1
```

### Récupérer des features (pour entraîner un modèle)
```bash
curl -X POST http://localhost:5050/api/feature-groups/vehicle_features_v1/fetch \
  -H "Content-Type: application/json" \
  -d '{"features": ["log_price", "norm_odo"], "entity_ids": ["0", "1", "2"]}'
```

### Depuis Python
```python
import urllib.request, json

url = "http://localhost:5050/api/feature-groups/vehicle_features_v1/fetch"
payload = json.dumps({"features": ["log_price", "norm_odo"]}).encode()
req = urllib.request.Request(url, data=payload, method="POST",
                              headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as r:
    data = json.loads(r.read())

import pandas as pd
df = pd.DataFrame(data["data"])
print(df.head())
```

### Depuis un notebook Jupyter
```python
import requests, pandas as pd

resp = requests.post(
    "http://localhost:5050/api/feature-groups/vehicle_features_v1/fetch",
    json={"features": ["log_price", "norm_odo", "vehicle_age"]}
)
df = pd.DataFrame(resp.json()["data"])
# → prêt pour X_train, y_train = df.drop("target", axis=1), df["target"]
```

---

## 🧪 Lancer les tests

```bash
# Depuis le dossier feature_store_app/
python tests/run_tests.py
```

La suite de tests couvre 41 cas répartis en 5 sections :
- **DataCleaner** (13 tests) — toutes les actions de nettoyage
- **FeatureEngineer** (11 tests) — toutes les transformations
- **FeatureStore** (11 tests) — CRUD complet, versioning, filtrage par entité
- **API REST** (5 tests) — health, list, fetch, erreurs 404
- **Pipeline End-to-End** (1 test) — pipeline complet clean → engineer → store → retrieve

Résultat attendu :
```
RÉSULTATS  :  41/41 tests passés
🎉  Tous les tests passent !
```

Si vous avez **pytest** installé, vous pouvez aussi lancer :
```bash
python -m pytest tests/run_tests.py -v
```

---

## 🔄 Flux complet recommandé

```
① Chargez votre CSV
       ↓
② Ajoutez des règles de nettoyage → Appliquez
       ↓
③ Ajoutez des transformations → Appliquez
       ↓
④ Sélectionnez les features → Sauvegardez dans le Store
       ↓
⑤ Récupérez via l'API depuis votre notebook d'entraînement
```

---

## 📝 Notes techniques

- **Stockage** : les features sont sauvegardées en format Parquet (efficace, compatible pandas/spark) et les métadonnées en SQLite.
- **API** : serveur HTTP standard Python (`http.server`), sans dépendance Flask. Fonctionne dès le démarrage de l'application.
- **Versioning** : chaque sauvegarde d'un groupe existant incrémente automatiquement la version.
- **Portabilité** : aucune dépendance cloud. Tout est local.
