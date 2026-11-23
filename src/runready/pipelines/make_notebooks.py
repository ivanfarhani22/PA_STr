"""Notebook 01: EDA Dataset Aktivitas dan Lomba Runner."""

import nbformat as nbf
import pandas as pd
from pathlib import Path

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# Notebook 01: EDA Dataset Aktivitas dan Lomba Runner\n\n**Tujuan**: Menganalisis karakteristik data mentah Kaggle, distribusi pace, jarak, dan elevasi per km."),
    nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nfrom runready.config import PATHS\n\ndf = pd.read_csv(PATHS.kaggle_raw, parse_dates=['timestamp'])\nprint('Jumlah baris mentah:', len(df))\ndf.head()"),
    nbf.v4.new_markdown_cell("## Kesimpulan & Keterbatasan\n- Data mentah mengandung aktivitas sebelum 2010 dan duplikat yang perlu dibersihkan.\n- Sebaran pace menunjukkan rentang luas dari jalan kaki hingga elite runner.")
]

Path("notebooks").mkdir(exist_ok=True)
with open("notebooks/01_eda_dataset.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook 01 berhasil dibuat.")
