# football-ai-app
Football AI V2 Application
.
markdown
# ⚽ Football AI V2

Aplikasi Streamlit untuk analisis pertandingan sepak bola dan rekomendasi taruhan Over/Under, BTTS, dan Correct Score.  
Menggunakan model XGBoost dan simulasi Poisson dengan koreksi Dixon-Coles.

## 🗂️ Struktur Proyek
football-ai-app/
├── app.py # Entry point Streamlit (UI + orkestrasi)
├── config.py # Konstanta & konfigurasi (EXPECTED_FEATURES, league_round_config)
├── utils.py # Fungsi utilitas (parsing odds, kelly criterion, hybrid top3)
├── model.pkl # Model XGBoost terlatih (OU + BTTS)
├── requirements.txt # Dependensi Python (kunci: Streamlit 1.55.0)
├── README.md # Dokumentasi ini
├── test_settlement.py # Unit test untuk settlement engine
├── test_feature_eng.py # Unit test untuk feature engineering
├── test_profit_analyzer.py # Unit test untuk profit analyzer
├── test_storage.py # Unit test untuk storage provider
├── assets/
│ └── style.css # Custom CSS
├── services/
│ ├── settlement.py # SettlementEngine (perhitungan hasil taruhan)
│ ├── feature_eng.py # Feature engineering (add_features)
│ ├── model_evaluator.py # Evaluasi performa model
│ ├── profit_analyzer.py # Analisis profit (total, per bulan, per liga)
│ ├── profit_calculator.py # Kalkulasi profit per pertandingan
│ ├── resource_registry.py # Registri resource (database & model)
│ └── storage.py # Storage provider (lokal & GitHub) + DatabaseManager

text

## 🚀 Cara Menjalankan

1. Clone repositori ini.
2. Install dependensi:
pip install -r requirements.txt

text
3. Pastikan file `model.pkl` ada di root.
4. (Opsional) Atur environment variable `GITHUB_TOKEN` jika ingin mengakses database di repositori `football-ai-db`.
5. Jalankan:
streamlit run app.py

text

## 🧪 Menjalankan Unit Test
pytest test_settlement.py test_feature_eng.py test_profit_analyzer.py test_storage.py -q

text

Atau jalankan satu per satu sesuai kebutuhan.

## ⚙️ Konfigurasi Penting

- **`requirements.txt`**: **Jangan ubah versi Streamlit** (`streamlit==1.55.0`) tanpa pengujian menyeluruh. Versi lebih baru tidak kompatibel dengan dependensi internal.
- **Secrets**: Token GitHub disimpan di Streamlit Secrets (`GITHUB_TOKEN`, `GITHUB_REPOSITORY`, `GITHUB_BRANCH`). File `secrets.toml` tidak termasuk dalam repositori.

## 📦 Sumber Data

Aplikasi membaca data dari dua repositori:
- **`football-ai-app`** (public): kode aplikasi dan model.
- **`football-ai-db`** (private): file CSV/JSON database (history, pending, profil liga, dll.).

## 👤 Kontak

- **Operator**: [nama/alias Anda] — pemilik dan pengelola aplikasi.
- Untuk pertanyaan teknis, hubungi Operator.
Keempat file di atas siap diunggah ke root repositori. Ketiga file test dapat langsung dijalankan dengan pytest, dan README.md memberikan dokumentasi yang jelas untuk pengguna dan pengembang.

