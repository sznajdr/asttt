
import joblib
import numpy as np

class CalibratedModel:
    """Wrapper: raw model + isotonic calibrator + optional beta shrinkage"""
    def __init__(self, model, calibrator, shrink_strength=0, base_rate=0):
        self.model = model
        self.calibrator = calibrator
        self.shrink_strength = shrink_strength
        self.base_rate = base_rate

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(raw)
        if self.shrink_strength > 0:
            calibrated = calibrated * (1 - self.shrink_strength) + self.base_rate * self.shrink_strength
        return np.column_stack([1 - calibrated, calibrated])


def load_models(save_dir='model_artifacts'):
    goal_bundle = joblib.load(f'{save_dir}/goal_model_bundle.pkl')
    assist_bundle = joblib.load(f'{save_dir}/assist_model_bundle.pkl')

    goal_model = CalibratedModel(goal_bundle['model'], goal_bundle['calibrator'])
    assist_model = CalibratedModel(
        assist_bundle['model'],
        assist_bundle['calibrator'],
        shrink_strength=assist_bundle.get('shrink_strength', 0),
        base_rate=assist_bundle.get('base_rate', 0)
    )

    return goal_model, assist_model, goal_bundle['features'], assist_bundle['features']
