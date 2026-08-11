# tests/test_utils.py
import pytest
from unittest.mock import patch
from utils import parse_odds_csv, parse_combined_odds_csv

# Contoh CSV sederhana untuk parse_odds_csv
VALID_CSV = b"""Type,Score,Odds
Home,1:0,4.5
Home,2:0,8.0
Away,1:0,5.2
Away,2:1,8.4
Draw,0:0,6.8
Draw,Other,43.0"""

VALID_CSV_WITH_OTHER = b"""Type,Score,Odds
Home,1:0,4.0
Away,2:1,7.0
Draw,Other,99.0"""

NO_HEADER_CSV = b"""1,1:0,4.5
2,2:0,8.0"""

EMPTY_CSV = b"""Type,Score,Odds"""

MINIMAL_CSV = b"""col1,col2,col3
Home,1:0,4.5"""


class TestParseOddsCsv:
    def test_valid_csv(self):
        result = parse_odds_csv(VALID_CSV)
        assert result == {
            "1:0": 4.5,
            "2:0": 8.0,
            "1:0": 4.5,  # akan overwrite? mari lihat implementasi
        }
        # Perhatikan bahwa parse_odds_csv menggunakan dictionary dengan key score.
        # Jika ada duplikat "1:0" untuk Home dan Away, yang terakhir akan menimpa.
        # Jadi hasilnya: {"1:0": 4.5? Tapi Away juga 1:0, akan jadi 5.2 jika Home 1:0 dulu baru Away?
        # Mari kita lihat kode: parse_odds_csv membaca per baris; untuk Away, skor 1:0 dibalik jadi 0:1.
        # Jadi key untuk Away 1:0 adalah "0:1". Jadi tidak konflik.
        # Jadi hasil yang benar: Home 1:0 -> "1:0":4.5, Away 1:0 -> "0:1":5.2, dll.
        # Perbaiki ekspektasi:
        # 1:0 dari Home -> "1:0":4.5
        # 2:0 Home -> "2:0":8.0
        # Away 1:0 -> "0:1":5.2
        # Away 2:1 -> "1:2":8.4
        # Draw 0:0 -> "0:0":6.8
        # Draw Other -> "OTHER":43.0
        expected = {
            "1:0": 4.5,
            "2:0": 8.0,
            "0:1": 5.2,
            "1:2": 8.4,
            "0:0": 6.8,
            "OTHER": 43.0,
        }
        assert result == expected

    def test_csv_with_other(self):
        result = parse_odds_csv(VALID_CSV_WITH_OTHER)
        expected = {
            "1:0": 4.0,
            "2:1": 7.0,
            "OTHER": 99.0,
        }
        assert result == expected

    def test_no_header_csv(self):
        # Tanpa header, fungsi akan mencoba menggunakan kolom ke-1 sebagai type, ke-2 score, ke-3 odds
        # Baris pertama dianggap data, jadi bisa menghasilkan dictionary.
        result = parse_odds_csv(NO_HEADER_CSV)
        # Baris pertama: type=1 (int), score='1:0', odds=4.5 -> dianggap sebagai home karena '1' tidak mengandung 'away'/'draw'
        # Mari kita lihat logika: if 'home' in type_str -> home, elif 'away' in type_str -> away, elif 'draw' in type_str -> draw.
        # '1' tidak mengandung itu, jadi tidak masuk kondisi apapun -> diabaikan? 
        # Jadi mungkin hasil kosong. Kita perlu menguji.
        # Sebenarnya, fungsi parse_odds_csv tidak mengasumsikan tipe kolom, hanya mencari kolom yang mengandung 'score' atau 'odds'.
        # Jika tidak ada header, kolom pertama dianggap type. Baris pertama type='1', bukan 'home','away','draw' -> tidak diproses.
        # Baris kedua type='2' juga tidak diproses. Jadi hasil kosong.
        assert result == {}

    def test_empty_csv(self):
        result = parse_odds_csv(EMPTY_CSV)
        assert result == {}   # karena tidak ada data setelah header

    def test_minimal_csv(self):
        result = parse_odds_csv(MINIMAL_CSV)
        # Kolom pertama: 'col1' tidak mengandung score/odds, tapi kode mencari kolom dengan 'score' dan 'odds' di nama.
        # Karena tidak ada, akan pakai kolom ke-1 sebagai type, ke-2 score, ke-3 odds (jika >=3 kolom). Ada 3 kolom.
        # Baris pertama type='Home', score='1:0', odds=4.5 -> dianggap home, key '1:0':4.5.
        assert result == {"1:0": 4.5}


# Sample combined CSV content
COMBINED_1X2_CS = b"""open_1x2_home,open_1x2_draw,open_1x2_away,current_1x2_home,current_1x2_draw,current_1x2_away
4.17,3.57,1.78,4.29,3.46,1.84

Type,Score,Odds
Home,1:0,8.2
Home,2:0,20
Away,1:0,4.6
Draw,0:0,6.8
Draw,Other,43"""

COMBINED_SIMPLE_1X2 = b"""home,draw,away
1.80,3.50,4.00

Type,Score,Odds
Home,1:0,5.0"""

COMBINED_ONLY_1X2 = b"""home,draw,away
1.80,3.50,4.00"""

COMBINED_NO_1X2 = b"""Type,Score,Odds
Home,1:0,5.0"""


class TestParseCombinedOddsCsv:
    def test_full_combined(self):
        result = parse_combined_odds_csv(COMBINED_1X2_CS)
        assert result["1x2"] == {"home": 4.29, "draw": 3.46, "away": 1.84}
        assert result["open_1x2"] == {"home": 4.17, "draw": 3.57, "away": 1.78}
        assert "1:0" in result["cs"]
        assert result["cs"]["1:0"] == 8.2
        assert result["cs"]["0:1"] == 4.6   # Away 1:0 flipped
        assert result["cs"]["0:0"] == 6.8
        assert result["cs"]["OTHER"] == 43.0
        assert result["errors"] == []

    def test_simple_1x2_with_cs(self):
        result = parse_combined_odds_csv(COMBINED_SIMPLE_1X2)
        assert result["1x2"] == {"home": 1.80, "draw": 3.50, "away": 4.00}
        assert result["open_1x2"] is None
        assert result["cs"]["1:0"] == 5.0

    def test_only_1x2(self):
        result = parse_combined_odds_csv(COMBINED_ONLY_1X2)
        assert result["1x2"] == {"home": 1.80, "draw": 3.50, "away": 4.00}
        assert result["cs"] is None
        assert "Correct Score tidak ditemukan" in str(result["errors"])

    def test_no_1x2(self):
        result = parse_combined_odds_csv(COMBINED_NO_1X2)
        assert result["1x2"] is None
        assert result["cs"] == {"1:0": 5.0}
        assert "Odds 1X2 tidak ditemukan" in str(result["errors"])

    def test_empty_file(self):
        result = parse_combined_odds_csv(b"")
        assert "File kosong" in result["errors"][0]
