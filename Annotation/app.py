import sys
print("当前Python路径:", sys.executable)
import pandas as pd
from main import df_training

from scorer.manual import interactive_manual_scoring_gui
df_streamlit = df_training
interactive_manual_scoring_gui(df_streamlit, save_path="Annotation/data/output/manual_scores.csv")

# streamlit run app.py