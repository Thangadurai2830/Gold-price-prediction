import joblib
import numpy as np

try:
    scaler = joblib.load('models/scaler.joblib')
    print('Scaler type:', type(scaler))
    print('Scaler fitted:', hasattr(scaler, 'scale_') and scaler.scale_ is not None)
    
    # Test with 50 features
    test_data = np.random.rand(1, 50)
    try:
        result = scaler.transform(test_data)
        print('Transform successful with shape:', result.shape)
    except Exception as e:
        print('Transform failed:', e)
        
except Exception as e:
    print('Failed to load scaler:', e)