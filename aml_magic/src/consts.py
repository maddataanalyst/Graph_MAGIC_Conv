from pathlib import Path


DATA_DIR = "data"
MAIN_PARAMS_FILE = "params"
STAGE_PARAMS_DIR = "stage_params"
PARAM_STAGES = "stages"

STAGE_PREPARE_DATA = "prepare_data"
STAGE_GNN_TRAINING = "gnn_training"
STAGE_GB_TRAINING = "gb_training"

RESULTS_DIR = Path(DATA_DIR, "results")
SCORES = "scores"
PREV_PAPER_RESULTS_PATH = Path(RESULTS_DIR, "prev_paper")
STUDY_RESULTS_PATH = Path(RESULTS_DIR, "study")
STUDY_COMPARISON_PATH = Path(RESULTS_DIR, "comparison")
