# Seattle : prédire la consommation d'énergie et les émissions de CO₂ des bâtiments non résidentiels

Projet de Data Engineer pour la ville de Seattle, dont l'objectif est d'atteindre la neutralité carbone en 2050.
Les relevés énergétiques des bâtiments sont coûteux à obtenir : ce projet entraîne un modèle de machine learning qui **estime la consommation d'énergie et les émissions de CO₂ d'un bâtiment à partir de ses seules caractéristiques structurelles** (surface, usage, année de construction, localisation…), puis **expose ce modèle via une API déployée dans le cloud**, avec un formulaire web pour les citoyens.

```
Données 2016  →  nettoyage + features  →  Random Forest (énergie) + Random Forest (CO₂)
                                                      │
                          BentoML (API + validation Pydantic + formulaire web)
                                                      │
                               Image Docker  →  Cloud Build  →  Cloud Run
```

## Sommaire

1. [Résultats en bref](#1-résultats-en-bref)
2. [Contenu du dépôt](#2-contenu-du-dépôt)
3. [Partie 1 : analyse et modélisation](#3-partie-1--analyse-et-modélisation)
4. [Partie 2 : l'API BentoML](#4-partie-2--lapi-bentoml)
5. [Lancer le projet en local](#5-lancer-le-projet-en-local)
6. [Déploiement sur Google Cloud](#6-déploiement-sur-google-cloud)
7. [Difficultés rencontrées et enseignements](#7-difficultés-rencontrées-et-enseignements)
8. [Limites et pistes d'amélioration](#8-limites-et-pistes-damélioration)

---

## 1. Résultats en bref

| | Énergie (`SiteEnergyUse(kBtu)`) | CO₂ (`TotalGHGEmissions`, tonnes éq. CO₂) |
|---|---|---|
| Modèle retenu | Random Forest réglé par GridSearch | Random Forest réglé par GridSearch |
| R² en train | 0,79 | 0,61 |
| R² en validation croisée (5 plis) | 0,60 | 0,41 |
| R² sur le test (avec 7 bâtiments extrêmes) | 0,67 | 0,40 |
| Erreur absolue moyenne (MAE) sur le test | ≈ 3,3 M kBtu | ≈ 100 t |
| **Erreur relative médiane sur le test** | **39 %** | **56 %** |
| Variable la plus importante | Surface totale (≈ 55 % de l'importance), puis type de propriété (≈ 11 %) | Surface totale (≈ 43 %), puis type de propriété (≈ 20 %) |

**Lecture métier** : sur un bâtiment jamais vu, le modèle se trompe en médiane d'environ 39 % sur l'énergie et 56 % sur le CO₂. C'est un outil d'**ordre de grandeur** pour prioriser les bâtiments à relever, pas un substitut à la mesure, surtout pour le CO₂ (les sources d'énergie, qui l'expliquent le mieux, sont volontairement exclues des variables pour éviter la fuite de données).

---

## 2. Contenu du dépôt

```
.
├── README.md
├── projet6_modelisation_seattle.ipynb     # Partie 1 : analyse, features, modèles, GridSearch, sauvegarde BentoML
├── 2016_Building_Energy_Benchmarking.csv  # Données sources (Ville de Seattle, relevés 2016)
├── service.py                             # Partie 2 : API BentoML + validation Pydantic + feature engineering
├── Accueil_conso.html                     # Formulaire web (servi par le même service)
├── bentofile.yaml                         # Packaging du service (dépendances, fichiers, version de Python)
└── cloudbuild.yaml                        # Construction de l'image Docker dans Google Cloud Build
```

Source des données : [Seattle Building Energy Benchmarking](https://data.seattle.gov/) (relevés de la Ville de Seattle, année 2016).

---

## 3. Partie 1 : analyse et modélisation

Tout le détail (code commenté ligne par ligne et interprétations) est dans le notebook. Voici la démarche.

### 3.1 Périmètre et nettoyage

Le fichier brut contient 3 376 relevés et 46 colonnes. Seuls les bâtiments **non destinés à l'habitation** sont concernés. Chaque filtre est tracé (nombre de lignes avant et après) :

| Étape | Lignes restantes |
|---|---|
| Données brutes | 3 376 |
| Retrait des logements collectifs (`Multifamily LR / MR / HR`) | 1 668 |
| Conservation des seuls relevés `Compliant` (relevé conforme) | 1 548 |
| Retrait des cibles absentes ou ≤ 0 (CO₂) | 1 546 |
| Retrait des bâtiments à 0 étage | 1 530 |
| Retrait des bâtiments à 0 bâtiment | **1 479** |

Autres décisions :
- Les valeurs manquantes **n'ont pas été supprimées en bloc** : certaines sont *structurelles* (pas de 2ᵉ ou 3ᵉ usage, donc une information utile pour compter les usages), d'autres concernent des colonnes inutilisables (`Comments`, `Outlier`), d'autres enfin sont liées à la consommation (`ENERGYSTARScore`, supprimé).
- Les quartiers écrits avec des casses différentes sont harmonisés (`Delridge` / `DELRIDGE NEIGHBORHOODS` → `DELRIDGE`).
- **Deux cibles**, modélisées exactement de la même façon (mêmes bâtiments, mêmes variables, même découpage, mêmes métriques) : `SiteEnergyUse(kBtu)` et `TotalGHGEmissions`.

### 3.2 Insights de l'analyse exploratoire

- Les distributions des deux cibles sont **très asymétriques à droite** : médiane de l'énergie ≈ 2,7 M kBtu pour une moyenne ≈ 8,2 M kBtu, maximum à 448 M kBtu. Quelques très gros consommateurs (hôpitaux, laboratoires, grands bureaux) tirent la moyenne.
- La **surface** est le premier moteur de la consommation et des émissions.
- Le **type d'usage** change fortement les ordres de grandeur : un hôpital médian consomme ≈ 58 M kBtu, contre ≈ 1,3 M kBtu pour un entrepôt.
- L'**âge** du bâtiment n'a pas d'effet net à lui seul.

### 3.3 Fuite de données (data leakage)

Les colonnes `SiteEUI`, `SourceEUI`, `Electricity`, `NaturalGas`, `SteamUse`, `GHGEmissionsIntensity`, `SiteEnergyUseWN` et `ENERGYSTARScore` sont **calculées à partir de la consommation** : les donner au modèle reviendrait à lui donner la réponse. Elles sont exclues des variables. La consommation n'est pas non plus une variable pour prédire le CO₂, et inversement.

### 3.4 Feature engineering (uniquement à partir d'informations structurelles)

| Catégorie | Variables créées |
|---|---|
| Temporalité | `Age_batiments` (2016 − année de construction), `Tranche_age_batiment` (< 1920, 1920-45, 1945-70, 1970-90, 1990-2005, 2005+) |
| Structure | `GFAPerFloor` (surface par étage), `ParkingRatio` (part de parking) |
| Usages | `NumUseTypes` (nombre d'usages : un vide sur le 2ᵉ/3ᵉ usage signifie « pas d'autre usage ») |
| Localisation | `DistDowntownKm` (distance à vol d'oiseau au centre-ville, formule de haversine), `Neighborhood`, latitude, longitude |

Les variables corrélées à plus de 0,7 entre elles ont été retirées pour limiter la redondance (`PropertyGFABuilding(s)`, `GFAPerBuilding`, `HasParking`, `IsMultiUse`, `LargestPropertyUseTypeGFA`).

**Variables finales (15)** : 11 numériques (`Age_batiments`, `NumberofFloors`, `NumberofBuildings`, `PropertyGFATotal`, `PropertyGFAParking`, `GFAPerFloor`, `ParkingRatio`, `NumUseTypes`, `Latitude`, `Longitude`, `DistDowntownKm`) et 4 catégorielles (`BuildingType`, `PrimaryPropertyType`, `Neighborhood`, `Tranche_age_batiment`).

### 3.5 Préparation et protocole d'évaluation

- **Un seul découpage** train/test 80/20 (`random_state=42`), **stratifié** sur la consommation pour retrouver petits, moyens et gros bâtiments dans les deux jeux : 1 183 bâtiments en entraînement, 296 en test.
- **Outliers par quantiles** (1 % et 99 %) plutôt que par IQR, qui aurait supprimé trop de gros bâtiments. Les seuils sont calculés sur le train uniquement et **appliqués au train seulement**. Le test garde ses bâtiments extrêmes, pour une évaluation réaliste. (Une première version filtrait aussi le test, ce qui le rendait artificiellement facile : c'est corrigé.)
- **Pré-traitement** (dans un `Pipeline`, donc sans fuite entre train et validation) :
  - numériques : imputation par la médiane puis `StandardScaler` ;
  - catégorielles : `OneHotEncoder(max_categories=15, handle_unknown="ignore")`.
- **Validation croisée** à 5 plis (`cross_validate`, R², MAE, RMSE) sur le train, puis évaluation finale sur le test.

### 3.6 Comparaison des modèles (paramètres de base)

R² en validation croisée (MAE de validation entre parenthèses) :

| Modèle | Énergie | CO₂ |
|---|---|---|
| Dummy (moyenne) | 0,00 (6,82 M kBtu) | 0,00 (146,86 t) |
| Régression linéaire | 0,57 (3,5 M) | 0,29 (106,29 t) |
| SVR standard | −0,15 (5,79 M) | −0,07 (121,43 t) |
| SVR réglé (cible mise à l'échelle) | 0,57 (3,29 M) | 0,34 (99,04 t) |
| Random Forest standard | 0,58 (3,37 M), **train 0,95 : sur-apprentissage** | 0,38 (106,16 t), train 0,92 |
| **Random Forest réglé** | **0,59 (3,42 M)**, train 0,84 | **0,43 (105,11 t)**, train 0,77 |

Le **Random Forest** est retenu pour les deux cibles : meilleure MAE et meilleur R² en validation croisée, relations non linéaires (l'effet de la surface dépend de l'usage), importance des variables native. Le choix repose sur la **validation croisée** et non sur le test (296 bâtiments, dont 7 très gros) pour que le test reste une évaluation honnête. Le SVR standard est inadapté tel quel à une cible qui se compte en millions ; sa version réglée (cible standardisée) devient compétitive.

### 3.7 Optimisation (GridSearch)

`GridSearchCV` sur 108 combinaisons (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `criterion`), validation croisée à 5 plis, **critère unique : la MAE** (elle se lit dans l'unité du problème et résiste aux quelques très gros bâtiments qui dominent R² et RMSE).

| | Meilleurs paramètres |
|---|---|
| Énergie | `absolute_error`, `max_depth=8`, `max_features=0.6`, `min_samples_leaf=3`, `n_estimators=200` |
| CO₂ | `absolute_error`, `max_depth=8`, `max_features=1.0`, `min_samples_leaf=5`, `n_estimators=300` |

### 3.8 Interprétation

Importance native (`feature_importances_`, additionnée par variable d'origine) confirmée par la **permutation importance** sur le test :
- **Énergie** : la surface totale domine (≈ 55 %), puis le type de propriété (≈ 11 %), le nombre d'étages (≈ 10 %), la surface de parking et la surface par étage (≈ 4 % chacune). La permutation confirme l'ordre : mélanger la surface totale augmente la MAE d'environ 3,9 M kBtu.
- **CO₂** : la surface totale (≈ 43 %) et surtout le **type de propriété (≈ 20 %)**, qui joue le rôle d'indicateur de la technologie de chauffage puisque les sources d'énergie sont exclues. La permutation donne +60 t de MAE pour la surface et +22 t pour le type de propriété.

Plusieurs variables de taille étant corrélées, l'importance indique ce que le modèle utilise, pas une relation de cause à effet.

---

## 4. Partie 2 : l'API BentoML

Un propriétaire de bâtiment envoie quelques informations simples et reçoit en temps réel une estimation d'énergie **et** de CO₂.

```
JSON  →  validation Pydantic  →  feature engineering (mêmes formules que le notebook)  →  2 Random Forest  →  JSON
```

### 4.1 Choix de conception

| Choix | Pourquoi |
|---|---|
| Un seul endpoint `POST /predict` pour deux modèles | Les deux cibles partagent les mêmes variables : un appel renvoie énergie et CO₂. |
| L'utilisateur envoie des **données brutes** (année, surface, quartier…) | Il ne connaît pas nos transformations (âge, tranche d'âge, distance au centre, encodage…) : le service les recalcule. |
| Variables et **ordre des colonnes** sauvegardés avec le modèle (`custom_objects`) | À l'inférence, les colonnes doivent arriver dans l'ordre vu à l'entraînement. |
| Pipeline complet (pré-traitement + modèle) sauvegardé | L'encodage appliqué en production est exactement celui de l'entraînement. |
| Pas de `Runner` | Concept archivé par BentoML ; les modèles sont chargés dans `__init__`. |
| Validation Pydantic | Un Random Forest répond toujours, même à une entrée absurde : on rejette les valeurs incohérentes avant le modèle. |

### 4.2 Validation des données (Pydantic)

| Champ | Règle |
|---|---|
| `YearBuilt` | entier entre 1900 et 2016 |
| `NumberofFloors` | entier entre 1 et 80 |
| `NumberofBuildings` | entier entre 1 et 30 (1 par défaut) |
| `PropertyGFATotal` | surface totale, entre 10 000 et 2 500 000 sq ft (plage vue à l'entraînement) |
| `PropertyGFAParking` | ≥ 0 (0 par défaut), **strictement inférieure** à la surface totale |
| `NumUseTypes` | entier entre 1 et 3 |
| `Latitude` / `Longitude` | dans la boîte englobante de Seattle |
| `BuildingType`, `PrimaryPropertyType`, `Neighborhood` | listes fermées (valeurs vues à l'entraînement, logement exclu) |
| Cohérence croisée | surface par étage ≥ 500 sq ft |

Toute valeur invalide renvoie une erreur HTTP 400 avec le champ concerné.

### 4.3 Exemple d'appel

Le corps de la requête est enveloppé dans `building_data` (nom du paramètre de la méthode `predict`).

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "building_data": {
      "YearBuilt": 1985,
      "NumberofFloors": 12,
      "NumberofBuildings": 1,
      "PropertyGFATotal": 150000,
      "PropertyGFAParking": 30000,
      "NumUseTypes": 1,
      "Latitude": 47.6084,
      "Longitude": -122.3352,
      "BuildingType": "NonResidential",
      "PrimaryPropertyType": "Hotel",
      "Neighborhood": "DOWNTOWN"
    }
  }'
```

Réponse (structure ; les valeurs dépendent du modèle entraîné) :

```json
{ "energy_use_kbtu": 10341664, "ghg_emissions_tco2e": 413.7 }
```

### 4.4 Interfaces

| Adresse | Contenu |
|---|---|
| `/` | Documentation interactive **Swagger** (tester l'API depuis le navigateur) |
| `/predict` | Endpoint de prédiction |
| `/Accueil_conso/` | **Formulaire web en français** pour les citoyens (liste d'usages et de quartiers, surfaces en pi² ou m², position GPS pré-remplie par quartier, contrôles de saisie, résultat en MWh, kWh/m² et tonnes de CO₂, avec rappel de l'incertitude) |

Le formulaire est une simple page HTML servie par le même service (application Starlette montée avec `@bentoml.asgi_app`). Il appelle `/predict` sur le même serveur, sans dépendance supplémentaire. **Garder la barre oblique finale** dans l'adresse (`/Accueil_conso/`).

---

## 5. Lancer le projet en local

**Prérequis** : Python 3.13 (ou celui indiqué dans `bentofile.yaml`) et les versions utilisées pour entraîner le modèle.

```bash
pip install bentoml scikit-learn==1.9.1 pandas==2.3.3 numpy==2.3.5 pydantic==2.12.4
```

> La version de scikit-learn doit être **identique** à celle qui a sauvegardé le modèle, sinon le chargement échoue ou donne des résultats faux.

1. **Entraîner et sauvegarder les modèles** : exécuter le notebook jusqu'à la cellule de sauvegarde. Elle enregistre `seattle_energy_rf` et `seattle_co2_rf` dans le *Model Store* local de BentoML (`bentoml models list` pour vérifier).
2. **Lancer le service** depuis le dossier du projet :
   ```bash
   bentoml serve service:SeattleEnergyService
   ```
   Swagger : <http://localhost:3000/> · Formulaire : <http://localhost:3000/Accueil_conso/>

---

## 6. Déploiement sur Google Cloud

Architecture retenue : **Docker** (image construite par BentoML) → **Artifact Registry** (stockage de l'image) → **Cloud Run** (exécution de l'image, service généraliste, pas de service spécialisé ML).

### 6.1 Le `bentofile.yaml`

```yaml
service: "service:SeattleEnergyService"   # fichier service.py, classe SeattleEnergyService
labels:
  project: seattle-energy-benchmarking
description: "Estimation de la consommation d'énergie et des émissions de CO2 d'un bâtiment non résidentiel de Seattle"
include:
  - service.py             # logique de l'API
  - Accueil_conso.html     # page du formulaire (le nom doit être identique à celui lu dans service.py)
python:
  packages:                # dépendances de production, versions épinglées (celles du notebook)
    - scikit-learn==1.9.1
    - pandas==2.3.3
    - numpy==2.3.5
    - pydantic==2.12.4
docker:
  python_version: "3.13"
```

Seuls les fichiers nécessaires sont inclus pour garder une image légère. Les modèles sont embarqués automatiquement (ils sont référencés dans `service.py`).

### 6.2 Procédure complète

Les commandes de la section A se lancent **sur le PC** (dans le dossier du projet), celles des sections B et C **dans Cloud Shell**. Remplacer `<PROJECT_ID>` par l'identifiant du projet Google Cloud.

**A. Sur le PC**

```bash
# 1. Construire le bento (service + modèles + dépendances)
bentoml build

# 2. Vérifier que le formulaire est bien dans le bento (dossier "src")
bentoml get seattle_energy_service:latest -o path

# 3. Exporter le bento dans un seul fichier
bentoml export seattle_energy_service:latest seattle.bento

# 4. Se connecter à Google Cloud et choisir le projet
gcloud auth login
gcloud config set project <PROJECT_ID>

# 5. Envoyer le fichier dans un bucket de transfert
gcloud storage buckets create gs://<PROJECT_ID>-transfer --location=europe-west9   # une seule fois
gcloud storage cp seattle.bento gs://<PROJECT_ID>-transfer/seattle.bento
```

**B. Dans Cloud Shell : construire l'image**

```bash
gcloud config set project <PROJECT_ID>
export PATH="$HOME/.local/bin:$PATH"
pip install bentoml                    # si la commande bentoml est absente

# Récupérer et importer le bento
gcloud storage cp gs://<PROJECT_ID>-transfer/seattle.bento ~/seattle.bento
bentoml delete seattle_energy_service -y       # supprime une éventuelle ancienne version
bentoml import ~/seattle.bento

# Préparer le contexte de build AVEC les modèles (voir section 7)
rm -rf ~/ctx
python3 - <<'EOF'
import shutil
from pathlib import Path
import bentoml
from bentoml._internal.container import construct_containerfile

bento = bentoml.get("seattle_energy_service:latest")
with construct_containerfile(bento, enable_buildkit=False) as (ctx, dockerfile):
    shutil.copytree(ctx, Path.home() / "ctx")
print("contexte prêt")
EOF
ls ~/ctx/src ~/ctx/models              # doit afficher service.py, Accueil_conso.html et les 2 modèles

# Activer les services et créer le dépôt d'images
gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com run.googleapis.com
gcloud artifacts repositories create seattle-repo --repository-format=docker --location=europe-west9
```

`cloudbuild.yaml` (à placer dans le dossier personnel de Cloud Shell) :

```yaml
steps:
  - name: gcr.io/cloud-builders/docker
    args: ['build', '-f', 'env/docker/Dockerfile', '-t',
           'europe-west9-docker.pkg.dev/<PROJECT_ID>/seattle-repo/seattle-energy:v1', '.']
images:
  - europe-west9-docker.pkg.dev/<PROJECT_ID>/seattle-repo/seattle-energy:v1
options:
  logging: CLOUD_LOGGING_ONLY
```

```bash
# Construire l'image dans le cloud et la déposer dans Artifact Registry
gcloud builds submit ~/ctx --config ~/cloudbuild.yaml --region europe-west9
```

**C. Déployer sur Cloud Run**

```bash
gcloud run deploy seattle-energy-service \
  --image europe-west9-docker.pkg.dev/<PROJECT_ID>/seattle-repo/seattle-energy:v1 \
  --region europe-west9 \
  --port 3000 --memory 2Gi --cpu 1 --max-instances 1 \
  --allow-unauthenticated
```

La commande affiche le **Service URL** : Swagger à la racine, formulaire sur `<Service URL>/Accueil_conso/`. Le premier appel peut être lent (démarrage à froid du conteneur).

### 6.3 Tests du service déployé

Dans Swagger (`<Service URL>/`) :
1. Une requête valide : réponse 200 avec `energy_use_kbtu` et `ghg_emissions_tco2e`.
2. Une requête incohérente (par exemple `NumberofFloors: 0`) : réponse **400** avec le champ en erreur.

### 6.4 Nettoyage (à faire dès les tests terminés)

Pour ne pas consommer de crédits inutilement :

```bash
gcloud run services delete seattle-energy-service --region europe-west9
gcloud artifacts repositories delete seattle-repo --location=europe-west9
gcloud storage rm -r gs://<PROJECT_ID>-transfer
```

---

## 7. Difficultés rencontrées et enseignements

| Problème | Cause | Solution |
|---|---|---|
| Build Docker en local qui échoue (délais d'attente sur Docker Hub, `apt-get` exit 100, `pip` bloqué) | Accès réseau du poste restreint (conteneurs sans sortie fiable, port 80 filtré) | Construire l'image **dans le cloud** (Cloud Shell puis Cloud Build) : même Docker, réseau fiable |
| Conteneur qui plante au démarrage sur Cloud Run (« model not found ») | Le contexte de build brut du bento ne contient pas les modèles | Générer le contexte avec `construct_containerfile`, qui copie les modèles dans `models/` |
| `docker push` depuis Cloud Shell refusé (connection refused) | Cause non élucidée (connexion vers le registre depuis Cloud Shell) | Passer par **Cloud Build**, qui construit et pousse depuis l'infrastructure Google |
| `Repository "seattle-repo" not found` | Dépôt d'images supprimé après un précédent nettoyage | Recréer le dépôt avant de relancer le build |
| Conteneur lancé sur un port déjà utilisé dans Cloud Shell | Port 3000 occupé | Mapper un autre port côté hôte (`-p 8080:3000`) |
| Le formulaire n'apparaît pas dans l'image | Fichier renommé mais `bentofile.yaml` pas mis à jour | Le nom dans `include` doit être **identique** à celui lu par `service.py` (casse comprise : Linux distingue majuscules et minuscules) |
| Page du formulaire qui répond 404 ou redirige | Barre oblique finale manquante | Utiliser `/Accueil_conso/` |
| Résultats incohérents ou erreur au chargement | Version de scikit-learn différente entre entraînement et service | **Épingler** les versions dans `bentofile.yaml` d'après le notebook |
| Confusion sur les commandes BentoML | Les messages « Next steps » de `bentoml build` renvoient vers BentoCloud (hors périmètre) | Ignorer BentoCloud : l'image est publiée sur Artifact Registry |

À retenir :
- Épingler les versions de bibliothèques entre entraînement et production.
- Embarquer dans le modèle la liste ordonnée des variables.
- Ne pas supposer que l'utilisateur de l'API connaît le pré-traitement : le service reçoit des données brutes.
- Évaluer sur un test qui garde les cas difficiles, et choisir le modèle sur la validation croisée.

---

## 8. Limites et pistes d'amélioration

- **Précision modeste, surtout sur le CO₂** : erreur relative médiane de 56 %. Les sources d'énergie (gaz, vapeur, électricité) sont exclues pour éviter la fuite de données, alors qu'elles expliquent les émissions. Une piste est de faire demander la source de chauffage dans le formulaire, à condition qu'elle ait été connue à l'entraînement comme information structurelle.
- **Sur-apprentissage encore visible** (énergie : R² train 0,79 pour 0,60 en validation) et **jeu de test réduit** (296 bâtiments, dont 7 très gros) : les métriques varient d'un découpage à l'autre.
- **Pas de données sur l'occupation, les équipements ou les systèmes de chauffage.**
- **Données de 2016 seulement** : un jeu plus récent (relevés 2015 à aujourd'hui) permettrait de réentraîner et de mesurer la dérive.
- **Périmètre de l'API** : les valeurs hors des plages d'entraînement (très petits ou très grands bâtiments, hors Seattle) sont refusées plutôt qu'extrapolées.
- **Pistes techniques** : réentraîner avec les seules variables les plus importantes pour alléger le modèle, ajouter des tests automatiques de l'API, un pipeline CI/CD (GitHub Actions) pour reconstruire et déployer l'image, et un suivi des appels en production.
