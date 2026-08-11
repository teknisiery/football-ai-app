# services/odds_handler.py
"""Handler untuk upload dan penyimpanan odds (1X2, Correct Score)."""
from typing import Dict, Any, Optional
import pandas as pd

from utils import parse_combined_odds_csv, convert_odds_to_wide
from services.resource_registry import ResourceRegistry


def process_combined_odds(
    file_content: bytes,
    match_uid: Optional[str],
    storage,
) -> Dict[str, Any]:
    """
    Parse file CSV combined 1X2 + Correct Score, lalu simpan ke history CS odds.

    Parameters
    ----------
    file_content : bytes
        Isi file CSV mentah.
    match_uid : str or None
        UID pertandingan yang sedang diprediksi. Jika None, history tidak disimpan.
    storage : StorageProvider
        Penyedia penyimpanan (lokal/GitHub).

    Returns
    -------
    dict
        Hasil parsing dengan struktur:
        {'1x2': dict | None, 'cs': dict | None, 'open_1x2': dict | None, 'errors': list, ...}
    """
    combined = parse_combined_odds_csv(file_content)
    if not combined:
        return combined

    # Simpan ke CS odds history hanya jika ada data Correct Score dan match_uid valid
    if combined.get('cs') and match_uid:
        _save_cs_odds_history(match_uid, file_content, combined, storage)

    return combined


def _save_cs_odds_history(
    match_uid: str,
    file_content: bytes,
    combined: dict,
    storage,
) -> None:
    """Simpan atau perbarui baris di correct_score_odds_history.csv."""
    # Siapkan baris data dari konten CSV asli (format wide)
    text = file_content.decode('utf-8-sig')
    idx = text.find('Type,Score,Odds')
    if idx != -1:
        cs_only = text[idx:].encode('utf-8')
    else:
        cs_only = file_content

    row_data = convert_odds_to_wide(match_uid, cs_only)

    # Tambahkan informasi 1X2 dan movement jika tersedia
    if combined.get('open_1x2'):
        for k, v in combined['open_1x2'].items():
            row_data[f'open_1x2_{k}'] = v
    if combined.get('1x2'):
        for k, v in combined['1x2'].items():
            row_data[f'current_1x2_{k}'] = v
    if combined.get('open_1x2') and combined.get('1x2'):
        for k in ['home', 'draw', 'away']:
            open_odd = combined['open_1x2'].get(k)
            current_odd = combined['1x2'].get(k)
            if open_odd is not None and current_odd is not None:
                row_data[f'movement_1x2_{k}'] = round(current_odd - open_odd, 2)

    # Muat history yang sudah ada
    if storage.exists(ResourceRegistry.CS_ODDS_HISTORY):
        cs_hist = storage.load_dataframe(ResourceRegistry.CS_ODDS_HISTORY)
    else:
        cs_hist = pd.DataFrame()

    # Update atau tambahkan baris
    if 'match_uid' in cs_hist.columns and match_uid in cs_hist['match_uid'].values:
        idx_cs = cs_hist[cs_hist['match_uid'] == match_uid].index[0]
        for col in row_data:
            if col.startswith('open_1x2_') or col.startswith('current_1x2_') or col.startswith('movement_1x2_'):
                cs_hist.at[idx_cs, col] = row_data[col]
    else:
        cs_hist = pd.concat([cs_hist, pd.DataFrame([row_data])], ignore_index=True)

    storage.save_dataframe(ResourceRegistry.CS_ODDS_HISTORY, cs_hist)
