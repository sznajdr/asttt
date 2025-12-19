
# =============================================================================
# HELPER CODE FOR LOADING MODELS
# =============================================================================
# Copy this to your inference notebook/script

import joblib
import numpy as np

class CalibratedModel:
    """Wrapper that combines raw model + isotonic calibrator"""
    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator
    
    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(raw)
        return np.column_stack([1 - calibrated, calibrated])


def load_models(save_dir='model_artifacts'):
    """Load goal and assist models from saved bundles"""
    
    # Load bundles
    goal_bundle = joblib.load(f'{save_dir}/goal_model_bundle.pkl')
    assist_bundle = joblib.load(f'{save_dir}/assist_model_bundle.pkl')
    
    # Create calibrated model wrappers
    goal_model = CalibratedModel(goal_bundle['model'], goal_bundle['calibrator'])
    assist_model = CalibratedModel(assist_bundle['model'], assist_bundle['calibrator'])
    
    # Also return features for reference
    goal_features = goal_bundle['features']
    assist_features = assist_bundle['features']
    
    return goal_model, assist_model, goal_features, assist_features


# Usage:
# goal_model, assist_model, goal_features, assist_features = load_models()
# prob = goal_model.predict_proba(X)[0, 1]
