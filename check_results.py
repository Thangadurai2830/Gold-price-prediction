import json

with open('results/robust_training_results_20250830_181132.json', 'r') as f:
    results = json.load(f)

print('Available models:')
for k, v in results.get('model_results', {}).items():
    print(f'- {k}: R2={v.get("test_r2", "N/A")}')

print(f'\nBest model: {results.get("best_model", "N/A")}')
print(f'Feature count: {results.get("feature_count", "N/A")}')