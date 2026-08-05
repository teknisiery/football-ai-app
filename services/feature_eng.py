Chief Architect → Lead Software Engineer

Perintah Perbaikan Final – add_features

---

Masalah

Test test_add_features_minimal masih gagal. Setelah perbaikan sebelumnya, error muncul di baris:

```python
df['xg_diff_home'] = df['home_xg'] - df['home_xga']
df['xg_diff_away'] = df['away_xg'] - df['away_xga']
```

Ketika kolom home_xga atau away_xga tidak tersedia, terjadi KeyError. Dua baris ini luput dari perbaikan sebelumnya.

---

Tugas

Ubah dua baris tersebut menjadi:

```python
df['xg_diff_home'] = df['home_xg'] - df.get('home_xga', pd.Series(0, index=df.index))
df['xg_diff_away'] = df['away_xg'] - df.get('away_xga', pd.Series(0, index=df.index))
```

Tidak ada perubahan lain. Dua baris itu saja.

---

Hasil yang Diharapkan

File services/feature_eng.py lengkap setelah perbaikan ini. Operator akan mengganti file lama, lalu menjalankan ulang pytest. Semua 19 test harus PASS.
