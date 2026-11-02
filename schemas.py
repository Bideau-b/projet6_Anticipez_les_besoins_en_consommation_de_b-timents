"""Validation des données envoyées à l'API (Pydantic).

Ce fichier ne contient que la logique de validation : service.py l'importe.
"""
from typing import Literal  # Literal : liste fermée de valeurs autorisées

from pydantic import BaseModel, Field, model_validator  # Pydantic : validation des données reçues


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
