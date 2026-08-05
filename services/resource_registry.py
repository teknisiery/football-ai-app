"""
Pendaftaran sumber daya (resource) untuk Football AI V2.
Menyimpan informasi semua file database dan model yang digunakan aplikasi.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Resource:
    id: str
    type: str
    serializer: str
    default_filename: str
    version: int = 1
    description: str = ""

class ResourceRegistry:
    HISTORY = Resource("history", "dataframe", "csv", "history_ou.csv")
    DATASET = Resource("dataset", "dataframe", "csv", "dataset_ou.csv")
    DATASET_WITH_GOAL = Resource("dataset_with_goal", "dataframe", "csv", "dataset_withgoal.csv")
    PENDING = Resource("pending", "dataframe", "csv", "pending_predictions.csv")
    LEAGUE_STATS = Resource("league_statistics", "dataframe", "csv", "league_statistics.csv")
    LEAGUE_PROFILE = Resource("league_profile", "dataframe", "csv", "profil_league.csv")
    LEAGUE_PROFILE_HISTORY = Resource("league_profile_history", "dataframe", "csv", "league_profile_history.csv")
    THRESHOLD = Resource("threshold", "dict", "json", "ev_threshold.json")
    MODEL = Resource("model", "pickle", "pickle", "model.pkl")
    FEATURE_COLUMNS = Resource("feature_columns", "dict", "json", "feature_columns.json")
    LEAGUE_THRESHOLD = Resource("league_threshold", "dataframe", "csv", "league_threshold.csv")
    CS_ODDS_HISTORY = Resource("cs_odds_history", "dataframe", "csv", "correct_score_odds_history.csv")

OPTIONAL_RESOURCES = {
    ResourceRegistry.HISTORY.id, ResourceRegistry.DATASET.id,
    ResourceRegistry.DATASET_WITH_GOAL.id, ResourceRegistry.PENDING.id,
    ResourceRegistry.LEAGUE_STATS.id, ResourceRegistry.THRESHOLD.id,
    ResourceRegistry.FEATURE_COLUMNS.id, ResourceRegistry.LEAGUE_THRESHOLD.id,
    ResourceRegistry.LEAGUE_PROFILE.id, ResourceRegistry.LEAGUE_PROFILE_HISTORY.id,
    ResourceRegistry.CS_ODDS_HISTORY.id,
}
