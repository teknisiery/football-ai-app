# run_shadow_evaluation.py
"""
Skrip offline untuk menjalankan shadow_evaluator terhadap history_ou.csv
dan mencetak laporan perbandingan Production vs Shadow P*.
Tidak terintegrasi dengan app.py atau UI.
"""
import json
import pandas as pd
from services.shadow_evaluator import evaluate_shadow_vs_production
from services.storage import LocalStorageProvider
from services.resource_registry import ResourceRegistry

def main():
    # Load history_ou.csv
    storage = LocalStorageProvider()
    history_df = storage.load_dataframe(ResourceRegistry.HISTORY)
    
    if history_df.empty:
        print("❌ history_ou.csv kosong atau tidak ditemukan.")
        return
    
    # Jalankan evaluasi
    result = evaluate_shadow_vs_production(history_df)
    
    # Cetak laporan
    print("=" * 60)
    print("SHADOW vs PRODUCTION EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\n📊 Sample Size:")
    print(f"  Total settled matches: {result['total_matches']}")
    print(f"  Valid Draw (paired):  {result['valid_draw_matches']}")
    print(f"  Valid 1X2 (paired):   {result['valid_1x2_matches']}")
    print(f"  Valid OU (paired):    {result['valid_ou_matches']}")
    print(f"  Valid BTTS (paired):  {result['valid_btts_matches']}")
    print(f"  Missing shadow:       {result['missing_shadow_count']}")
    print(f"  Missing production:   {result['missing_production_prob_count']}")
    print(f"  Invalid prob sum:     {result.get('invalid_probability_count', 0)}")
    
    print(f"\n🎯 DRAW:")
    print(f"  Baseline prob: {result['baseline_draw_prob']:.4f}")
    print(f"  Baseline Brier: {result['baseline_draw_brier']:.6f}")
    print(f"  Baseline LogLoss: {result['baseline_draw_logloss']:.6f}")
    print(f"  Production Brier: {result['prod_draw_brier']:.6f}")
    print(f"  Production LogLoss: {result['prod_draw_logloss']:.6f}")
    print(f"  Shadow Brier: {result['shadow_draw_brier']:.6f}")
    print(f"  Shadow LogLoss: {result['shadow_draw_logloss']:.6f}")
    
    print(f"\n⚽ 1X2:")
    print(f"  Production LogLoss: {result['prod_1x2_logloss']:.6f}")
    print(f"  Production Brier: {result['prod_1x2_brier']:.6f}")
    print(f"  Production Accuracy: {result['prod_1x2_accuracy']:.4f}")
    print(f"  Shadow LogLoss: {result['shadow_1x2_logloss']:.6f}")
    print(f"  Shadow Brier: {result['shadow_1x2_brier']:.6f}")
    print(f"  Shadow Accuracy: {result['shadow_1x2_accuracy']:.4f}")
    
    print(f"\n📈 OVER/UNDER:")
    print(f"  Production Brier: {result['prod_ou_brier']:.6f}")
    print(f"  Shadow Brier: {result['shadow_ou_brier']:.6f}")
    
    print(f"\n🤝 BTTS:")
    print(f"  Production Brier: {result['prod_btts_brier']:.6f}")
    print(f"  Shadow Brier: {result['shadow_btts_brier']:.6f}")
    
    print(f"\n📐 DRAW CALIBRATION (Production):")
    for bucket, data in result['prod_draw_calibration'].items():
        if data['count'] > 0:
            print(f"  {bucket}: pred={data['predicted']:.4f}, actual={data['actual']:.4f}, n={data['count']}")
    
    print(f"\n📐 DRAW CALIBRATION (Shadow):")
    for bucket, data in result['shadow_draw_calibration'].items():
        if data['count'] > 0:
            print(f"  {bucket}: pred={data['predicted']:.4f}, actual={data['actual']:.4f}, n={data['count']}")
    
    print(f"\n🥅 GOAL DIFFERENCE CALIBRATION (Shadow):")
    gd = result['shadow_goal_diff_calibration']
    for bucket in ['-3', '-2', '-1', '0', '+1', '+2', '+3']:
        data = gd[bucket]
        pred_str = f"{data['predicted']:.4f}" if data['predicted'] is not None else "N/A"
        actual_str = f"{data['actual']:.4f}" if data['actual'] is not None else "N/A"
        print(f"  GD {bucket}: pred={pred_str}, actual={actual_str}, n={data['count']}")
    
    # Simpan ke JSON
    with open("shadow_evaluation_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n✅ Report disimpan ke shadow_evaluation_result.json")

if __name__ == "__main__":
    main()
