import sys
sys.path.append('.')

from src.data.parse_annotations import parse_distraction, get_label_stats
import pandas as pd

# Parse les deux sessions
df_s1 = parse_distraction(
    json_path='data/raw/gA_3_s1_2019-03-08T10_27_38_01_00_rgb_ann_distraction.json',
    session_name='s1'
)

df_s4 = parse_distraction(
    json_path='data/raw/gE_28_s4_2019-03-21T10_19_50_01_00_rgb_ann_distraction.json',
    session_name='s4'
)

# Combine les deux
df = pd.concat([df_s1, df_s4], ignore_index=True)

print('=== Aperçu ===')
print(df.head(10).to_string())
print()

get_label_stats(df)

# Sauvegarde
df.to_csv('data/processed/distraction_annotations.csv', index=False)
print('\nSauvegardé : data/processed/distraction_annotations.csv')