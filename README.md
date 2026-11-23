# Proyek Akhir: Sistem Prediksi Kesiapan Pelari Menghadapi Lomba Road dan Trail Running Menggunakan Machine Learning

Repositori ini berisi kode refactor dari proyek akhir Machine Learning untuk memprediksi waktu finis pelari berdasarkan riwayat latihan 4 minggu serta karakteristik lomba, lalu menghitung persentase dan kategori kesiapan terhadap *Cut-Off Time* (COT).

## Struktur Direktori
```
proyek-akhir/
├─ pyproject.toml            # Paket "runready" (install: pip install -e .)
├─ README.md
├─ requirements.txt
├─ src/runready/             # Paket utama Python (Clean Code + Design Patterns)
│  ├─ config.py              # Konfigurasi terpusat & dataclass frozen
│  ├─ data/                  # Repository loader & cleaning pipeline
│  ├─ models/                # Registry model, trainer, & baselines
│  ├─ evaluation/            # Metrik regresi & kalibrasi Brier score
│  ├─ readiness.py           # Kalkulator kesiapan runner & kategori
│  ├─ service.py             # PredictionService (Facade untuk web/API)
│  └─ pipelines/             # Entry-point tipis pemanggil modul
├─ notebooks/                # Jupyter Notebooks (01 s/d 07)
├─ tests/                    # Unit test pytest
├─ reports/                  # Laporan tabel CSV & grafik figures/
├─ data/                     # raw/ dan processed/
├─ models/                   # final_model.pkl & residuals_rel.npy
└─ app/                      # Kosong (disiapkan untuk FastAPI nanti)
```

## Cara Menjalankan Pipeline & Pengujian
1. **Instalasi Paket (Editable Mode):**
   ```bash
   pip install -e .
   ```
2. **Menjalankan Unit Test (pytest):**
   ```bash
   python -m pytest tests/test_runready.py
   ```
3. **Menjalankan Pipeline (dari nol secara idempoten):**
   ```bash
   python -m runready.pipelines.run_clean
   python -m runready.pipelines.run_train
   ```

## Keterbatasan
- **Label Proxy**: Menggunakan aturan heuristik (jarak standar + speedup 8%) karena ketiadaan label lomba langsung pada dataset utama Kaggle.
- **Belum Ada Data Trail Individu**: Model dilatih terutama pada data road running datar; untuk trail running (UTMB), digunakan analisis event-level terpisah dengan flag peringatan `out_of_training_range`.
- **DNF (Did Not Finish)**: Tidak dimodelkan secara eksplisit; diasumsikan pelari berhasil finis sesuai COT.
