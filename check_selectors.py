import pickle
import os

selectors = ['feature_selector.pkl', 'selector.pkl', 'feature_selector_20250824_130408.pkl', 'robust_selector_20250824_133007.pkl', 'robust_selector_20250830_181132.pkl']

print('Checking selectors:')
for s in selectors:
    if os.path.exists(f'models/{s}'):
        try:
            selector = pickle.load(open(f'models/{s}', 'rb'))
            if hasattr(selector, 'k'):
                print(f'{s}: {selector.k} features')
            else:
                print(f'{s}: unknown features (no k attribute)')
        except Exception as e:
            print(f'{s}: error loading - {e}')
    else:
        print(f'{s}: file not found')