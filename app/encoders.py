# app/encoders.py
PRODUCT_TYPE_MAP = {
    "Cold Medicine": 0,
    "Vitamins": 1,
    "Painkiller": 2,
    "Allergy": 3
}

TARGET_AGE_MAP = {
    "Children": 0,
    "Adults": 1,
    "Seniors": 2
}

CLIMATE_MAP = {
    "Cold": 0,
    "Moderate": 1,
    "Warm": 2,
    "Temperate": 2
}

SEASON_MAP = {
    "Winter": 0,
    "Spring": 1,
    "Summer": 2,
    "Autumn": 3
}

URBAN_MAP = {
    "Rural": 0,
    "Semi-urban": 1,
    "Urban": 2
}

FEATURE_NAMES = [
    "average_income", "average_age", "population_density", "elderly_ratio",
    "climate", "season", "product_type", "target_age_group", "average_price", "urbanization"
]
