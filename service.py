from pathlib import Path  # Path : retrouver le fichier Accueil_conso.html à côté de service.py
from typing import Literal  # Literal : liste fermée de valeurs autorisées

import bentoml  # BentoML : transforme la classe en API web
import numpy as np  # numpy : calcul de la distance GPS
import pandas as pd  # pandas : construit le tableau de features attendu par le modèle
from pydantic import BaseModel, Field, model_validator  # Pydantic : validation des données reçues
from starlette.applications import Starlette  # Starlette : mini application web (déjà installée avec BentoML)
from starlette.responses import HTMLResponse  # HTMLResponse : renvoyer une page web
from starlette.routing import Route  # Route : associer une adresse (URL) à une fonction


# 1. Ce que l'utilisateur envoie : des données BRUTES (les features calculées sont créées plus bas)
class BuildingData(BaseModel):
    YearBuilt: int = Field(ge=1900, le=2016, description="Année de construction (1900-2016)")  # le modèle est entraîné sur des relevés 2016
    NumberofFloors: int = Field(ge=1, le=80, description="Nombre d'étages (1-80)")  # 0 étage est impossible
    NumberofBuildings: int = Field(default=1, ge=1, le=30, description="Nombre de bâtiments (1-30)")  # au moins 1 bâtiment
    PropertyGFATotal: float = Field(ge=10_000, le=2_500_000, description="Surface totale en sq ft, parking inclus")  # plage vue à l'entraînement
    PropertyGFAParking: float = Field(default=0, ge=0, description="Surface de parking en sq ft (0 si aucun)")  # jamais négative
    NumUseTypes: int = Field(ge=1, le=3, description="Nombre d'usages du bâtiment (1 à 3)")  # le jeu de données décrit 3 usages maximum
    Latitude: float = Field(ge=47.48, le=47.78, description="Latitude (dans Seattle)")  # boîte englobante de Seattle
    Longitude: float = Field(ge=-122.45, le=-122.22, description="Longitude (dans Seattle)")  # boîte englobante de Seattle
    BuildingType: Literal["Campus", "NonResidential", "Nonresidential COS", "Nonresidential WA", "SPS-District K-12"] = "NonResidential"  # types vus à l'entraînement
    PrimaryPropertyType: Literal[  # usages principaux vus à l'entraînement (sans le logement, hors périmètre)
        "Distribution Center", "Hospital", "Hotel", "K-12 School", "Laboratory", "Large Office", "Medical Office",
        "Mixed Use Property", "Other", "Refrigerated Warehouse", "Residence Hall", "Restaurant", "Retail Store",
        "Self-Storage Facility", "Senior Care Community", "Small- and Mid-Sized Office",
        "Supermarket / Grocery Store", "University", "Warehouse", "Worship Facility"]
    Neighborhood: Literal[  # quartiers vus à l'entraînement (en MAJUSCULES, comme dans les données nettoyées)
        "BALLARD", "CENTRAL", "DELRIDGE", "DOWNTOWN", "EAST", "GREATER DUWAMISH", "LAKE UNION",
        "MAGNOLIA / QUEEN ANNE", "NORTH", "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST"]

    @model_validator(mode="after")  # règle qui compare plusieurs champs, exécutée une fois les champs validés
    def parking_plus_petit_que_total(self):
        if self.PropertyGFAParking >= self.PropertyGFATotal:  # le parking fait partie de la surface totale
            raise ValueError("PropertyGFAParking doit être inférieur à PropertyGFATotal")  # message renvoyé à l'utilisateur
        if self.PropertyGFATotal / self.NumberofFloors < 500:  # moins de 500 sq ft par étage : erreur de saisie
            raise ValueError("Surface par étage inférieure à 500 sq ft : valeurs incohérentes")  # message renvoyé à l'utilisateur
        return self


# 2. Feature engineering : MÊMES formules que dans le notebook
def haversine_km(lat1, lon1, lat2, lon2):  # distance à vol d'oiseau entre deux points GPS
    r = 6371  # rayon de la Terre en km
    p1, p2 = np.radians(lat1), np.radians(lat2)  # latitudes en radians
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2  # formule de haversine
    return 2 * r * np.arcsin(np.sqrt(a))  # distance en km


def construire_features(b: BuildingData, colonnes: list) -> pd.DataFrame:
    tranche = pd.cut([b.YearBuilt], bins=[1800, 1920, 1945, 1970, 1990, 2005, 2016],  # tranche d'époque (mêmes bornes que le notebook)
                     labels=["<1920", "1920-45", "1945-70", "1970-90", "1990-2005", "2005+"])[0]
    ligne = {
        "Age_batiments": 2016 - b.YearBuilt,  # âge au moment du relevé (2016)
        "NumberofFloors": b.NumberofFloors,  # nombre d'étages
        "NumberofBuildings": b.NumberofBuildings,  # nombre de bâtiments
        "PropertyGFATotal": b.PropertyGFATotal,  # surface totale
        "PropertyGFAParking": b.PropertyGFAParking,  # surface de parking
        "GFAPerFloor": b.PropertyGFATotal / b.NumberofFloors,  # surface moyenne par étage
        "ParkingRatio": b.PropertyGFAParking / b.PropertyGFATotal,  # part de parking
        "NumUseTypes": b.NumUseTypes,  # nombre d'usages
        "Latitude": b.Latitude,  # latitude
        "Longitude": b.Longitude,  # longitude
        "DistDowntownKm": haversine_km(b.Latitude, b.Longitude, 47.6062, -122.3321),  # distance au centre-ville
        "BuildingType": b.BuildingType,  # type administratif
        "PrimaryPropertyType": b.PrimaryPropertyType,  # usage principal
        "Neighborhood": b.Neighborhood,  # quartier
        "Tranche_age_batiment": str(tranche),  # tranche d'âge en texte, comme à l'entraînement
    }
    return pd.DataFrame([ligne])[colonnes]  # colonnes dans l'ordre vu à l'entraînement


# 3. La page web (formulaire) : un simple fichier HTML servi à l'adresse /ui
async def page_formulaire(request):  # appelée quand le navigateur ouvre /ui
    html = (Path(__file__).parent / "Accueil_conso.html").read_text(encoding="utf-8")  # lit le fichier ui.html (copié dans l'image par bentofile.yaml)
    return HTMLResponse(html)  # renvoie la page au navigateur


app_ui = Starlette(routes=[Route("/", page_formulaire)])  # mini application : une seule page, à la racine de son chemin


# 4. Le service BentoML
@bentoml.service(name="seattle_energy_service")
@bentoml.asgi_app(app_ui, path="/Accueil_conso")  # branche la page sur le même service : disponible à l'adresse /ui (l'API /predict reste inchangée)
class SeattleEnergyService:
    modele_energie = bentoml.models.BentoModel("seattle_energy_rf:latest")  # modèle d'énergie sauvegardé par le notebook
    modele_co2 = bentoml.models.BentoModel("seattle_co2_rf:latest")  # modèle de CO2 sauvegardé par le notebook

    def __init__(self):
        self.energie = bentoml.sklearn.load_model(self.modele_energie)  # pipeline complet (préparation + Random Forest)
        self.co2 = bentoml.sklearn.load_model(self.modele_co2)  # pipeline complet (préparation + Random Forest)
        self.features = self.modele_energie.custom_objects["features"]  # liste des features dans l'ordre d'entraînement

    @bentoml.api
    def predict(self, building_data: BuildingData) -> dict:
        X = construire_features(building_data, self.features)  # données brutes -> features (le pipeline fait ensuite l'encodage)
        return {
            "energy_use_kbtu": round(float(self.energie.predict(X)[0])),  # énergie estimée (kBtu)
            "ghg_emissions_tco2e": round(float(self.co2.predict(X)[0]), 1),  # émissions estimées (tonnes éq. CO2)
        }
