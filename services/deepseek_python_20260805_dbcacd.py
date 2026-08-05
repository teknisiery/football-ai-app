"""
Analisis profit dan ringkasan performa untuk Football AI V2.
Menyediakan perhitungan profit total, per bulan, per liga, serta caching ringkasan.
"""
import pandas as pd
import numpy as np
import json
import streamlit as st
from datetime import timedelta
from pathlib import Path
from typing import Tuple

from config import BASE_DIR
from utils import get_valid_time
from services.settlement import SettlementEngine

# ----------------------------------------------------------------------
# Duplikasi minimal ProfitCalculator (sementara, seperti di model_evaluator)
# ----------------------------------------------------------------------
class ProfitCalculator:
    @staticmethod
    def calculate(row: dict, stake: float = 100000.0) -> Tuple[float, str]:
        home_goals = int(row.get('home_goals', 0) or 0)
        away_goals = int(row.get('away_goals', 0) or 0)
        res = SettlementEngine.evaluate(row, home_goals, away_goals)
        return res['profit'], res['result']
# ----------------------------------------------------------------------

PROFIT_SUMMARY_FILE = BASE_DIR / "profit_summary.json"


@st.cache_data(ttl=300)
def compute_detailed_profits(history_df: pd.DataFrame):
    if history_df.empty:
        return 0.0, pd.DataFrame(), pd.DataFrame(), {
            "total_bets": 0, "full_win": 0, "half_win": 0, "push": 0,
            "half_lose": 0, "full_lose": 0
        }, {}

    df = history_df.copy()
    pc = ProfitCalculator()
    profits = []
    statuses = []
    for _, row in df.iterrows():
        profit, status = pc.calculate(row.to_dict(), 100000)
        profits.append(profit)
        statuses.append(status)
    df['profit'] = profits
    df['result'] = statuses

    bet_df = df[df['result'] != 'NO BET'].copy()
    total_profit = bet_df['profit'].sum() if not bet_df.empty else 0.0

    if not bet_df.empty:
        status_counts = bet_df['result'].value_counts()
        summary = {
            'total_bets': len(bet_df),
            'full_win': status_counts.get('FULL WIN', 0),
            'half_win': status_counts.get('HALF WIN', 0),
            'push': status_counts.get('PUSH', 0),
            'half_lose': status_counts.get('HALF LOSE', 0),
            'full_lose': status_counts.get('FULL LOSE', 0)
        }
        profit_by_status = bet_df.groupby('result')['profit'].sum().to_dict()
    else:
        summary = {
            'total_bets': 0, 'full_win': 0, 'half_win': 0,
            'push': 0, 'half_lose': 0, 'full_lose': 0
        }
        profit_by_status = {}

    bet_df['_valid_time'] = bet_df.apply(lambda row: get_valid_time(row), axis=1)
    invalid = bet_df['_valid_time'].isna().sum()
    if invalid > 0:
        st.warning(f"⚠️ {invalid} baris memiliki timestamp tidak valid dan diabaikan dalam grafik bulanan.")
    bet_df = bet_df.dropna(subset=['_valid_time'])

    if not bet_df.empty:
        df_bulan = bet_df.copy()
        df_bulan['bulan'] = df_bulan['_valid_time'].dt.strftime('%Y-%m')
        df_bulan['bulan'] = df_bulan['bulan'].fillna('Tanpa Tanggal')
        monthly_groups = df_bulan.groupby('bulan')
        monthly_data = {}
        for bulan, group in monthly_groups:
            monthly_profit = group['profit'].sum()
            monthly_summary = {
                'total_bets': len(group),
                'full_win': group['result'].tolist().count('FULL WIN'),
                'half_win': group['result'].tolist().count('HALF WIN'),
                'push': group['result'].tolist().count('PUSH'),
                'half_lose': group['result'].tolist().count('HALF LOSE'),
                'full_lose': group['result'].tolist().count('FULL LOSE')
            }
            monthly_data[bulan] = {
                'profit': monthly_profit,
                'summary': monthly_summary,
                'df': group,
                'time_col': '_valid_time'
            }
    else:
        monthly_data = {}

    return total_profit, bet_df, summary, monthly_data, profit_by_status


def save_profit_summary(league_data: dict):
    slim = {}
    for league, data in league_data.items():
        slim[league] = {
            'profit': data['profit'],
            'summary': data['summary']
        }
    with open(PROFIT_SUMMARY_FILE, 'w') as f:
        json.dump(slim, f, indent=2)


def load_profit_summary():
    if PROFIT_SUMMARY_FILE.exists():
        try:
            with open(PROFIT_SUMMARY_FILE) as f:
                return json.load(f)
        except:
            pass
    return None


def compute_profits_by_league(history_df: pd.DataFrame):
    if history_df.empty:
        return {}

    df = history_df.copy()
    pc = ProfitCalculator()
    profits = []
    statuses = []
    for _, row in df.iterrows():
        profit, status = pc.calculate(row.to_dict(), 100000)
        profits.append(profit)
        statuses.append(status)
    df['profit'] = profits
    df['result'] = statuses

    bet_df = df[df['result'] != 'NO BET'].copy()
    if bet_df.empty:
        return {}

    if 'league_name' not in bet_df.columns:
        bet_df['league_name'] = 'Unknown'
    else:
        bet_df['league_name'] = bet_df['league_name'].str.title().str.strip()

    bet_df['_valid_time'] = bet_df.apply(lambda row: get_valid_time(row), axis=1)
    bet_df = bet_df.dropna(subset=['_valid_time'])

    league_groups = bet_df.groupby('league_name')
    league_data = {}
    for league, group in league_groups:
        profit = group['profit'].sum()
        summary = {
            'total_bets': len(group),
            'full_win': group['result'].tolist().count('FULL WIN'),
            'half_win': group['result'].tolist().count('HALF WIN'),
            'push': group['result'].tolist().count('PUSH'),
            'half_lose': group['result'].tolist().count('HALF LOSE'),
            'full_lose': group['result'].tolist().count('FULL LOSE')
        }
        league_data[league] = {
            'profit': profit,
            'summary': summary,
            'df': group,
            'time_col': '_valid_time'
        }

    save_profit_summary(league_data)
    return league_data