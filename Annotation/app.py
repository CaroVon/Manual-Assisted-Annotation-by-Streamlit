import sys
import os

project_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("当前Python路径:", sys.executable)

import pandas as pd
from main import df_training
from scorer.manual import interactive_manual_scoring_gui
df_streamlit = df_training
interactive_manual_scoring_gui(df_streamlit, save_path="train/data/manual_scores.csv")

# streamlit run app.py