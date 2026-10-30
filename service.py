from pathlib import Path  # Path : retrouver le fichier Accueil_conso.html à côté de service.py

import bentoml  # BentoML : transforme la classe en API web
import numpy as np  # numpy : calcul de la distance GPS
import pandas as pd  # pandas : construit le tableau de features attendu par le modèle
from starlette.applications import Starlette  # Starlette : mini application web (déjà installée avec BentoML)
from starlette.responses import HTMLResponse  # HTMLResponse : renvoyer une page web
from starlette.routing import Route  # Route : associer une adresse (URL) à une fonction

from schemas import BuildingData  # validation des données envoyées (voir schemas.py)


#1) Feature engineering : MÊMES formules que dans le notebook
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


#2) La page web (formulaire) : un simple fichier HTML servi à l'adresse /Accueil_conso/
async def page_formulaire(request):  # appelée quand le navigateur ouvre /Accueil_conso/
    html = (Path(__file__).parent / "Accueil_conso.html").read_text(encoding="utf-8")  # lit Accueil_conso.html (copié dans l'image par bentofile.yaml)
    return HTMLResponse(html)  # renvoie la page au navigateur


app_ui = Starlette(routes=[Route("/", page_formulaire)])  # mini application : une seule page, à la racine de son chemin


# 3. Le service BentoML
@bentoml.service(name="seattle_energy_service")
@bentoml.asgi_app(app_ui, path="/Accueil_conso")  # branche la page sur le même service : disponible à l'adresse /Accueil_conso/
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
