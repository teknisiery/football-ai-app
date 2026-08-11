# services/pending_service.py
"""Proses settlement pertandingan dari pending predictions."""
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from services.settlement import SettlementEngine
from services.resource_registry import ResourceRegistry
from services.storage import StorageProvider, DatabaseManager
from services.league_profile import update_league_profile
from utils import normalize_kickoff, reorder_columns


class PendingService:
    @staticmethod
    def process_settlement(
        row_dict: Dict[str, Any],
        ht_home_goals: int,
        ht_away_goals: int,
        ft_home_goals: int,
        ft_away_goals: int,
        storage: StorageProvider,
        session=None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Proses settlement untuk satu baris pending.

        Returns:
            Tuple[bool, str, dict | None]:
                (berhasil, pesan, data history lengkap jika berhasil, None jika gagal)
        """
        db = DatabaseManager(storage)
        
        # Siapkan record lengkap
        full_record = dict(row_dict)
        full_record['kickoff_time'] = normalize_kickoff(full_record.get('kickoff_time'))
        full_record['home_ht_goals'] = ht_home_goals
        full_record['away_ht_goals'] = ht_away_goals
        full_record['home_goals'] = ft_home_goals
        full_record['away_goals'] = ft_away_goals
        full_record['totalgol_ft'] = ft_home_goals + ft_away_goals
        full_record['totalgol_ht'] = ht_home_goals + ht_away_goals
        full_record['settlement_time'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        settlement = SettlementEngine.evaluate(full_record, ft_home_goals, ft_away_goals)
        full_record.update(settlement)

        # Cek duplikasi di history
        existing_hist = db.load_history()
        if not existing_hist.empty and 'match_uid' in existing_hist.columns:
            if full_record['match_uid'] in existing_hist['match_uid'].values:
                return False, "Pertandingan sudah ada di History.", None

        # Simpan ke history
        hist_record_df = pd.DataFrame([full_record])
        hist_record_df = reorder_columns(hist_record_df)
        hist = db.load_history()
        hist = pd.concat([hist, hist_record_df], ignore_index=True)
        db.save_history(hist)

        # Simpan ke dataset dengan goal
        dataset_record_df = hist_record_df.copy()
        if 'league_name' in dataset_record_df.columns:
            dataset_record_df = dataset_record_df.drop(columns=['league_name'])
        
        dataset_wg = db.load_dataset_with_goal()
        if not dataset_wg.empty and 'match_uid' in dataset_wg.columns:
            if full_record['match_uid'] not in dataset_wg['match_uid'].values:
                dataset_wg = pd.concat([dataset_wg, dataset_record_df], ignore_index=True)
                db.save_dataset_with_goal(dataset_wg)
        else:
            db.save_dataset_with_goal(dataset_record_df)

        # Hapus dari pending
        pend = db.load_pending()
        mask = pend['match_uid'] == full_record['match_uid']
        pend = pend[~mask]
        db.save_pending(pend)

        # Perbarui profil liga
        league_code = int(row_dict.get('league_code', 0))
        update_league_profile(storage, league_code, session)

        return True, "Skor disimpan dan dipindahkan ke History.", full_record
