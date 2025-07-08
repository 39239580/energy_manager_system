from core.data_process_utils.process_fn import sk_load


class DeepModelEvaluator(object):
    def __init__(self, scaler=None, scaler_path=None):
        self.scaler = scaler
        self.scaler_path = scaler_path
        if self.scaler is None:
            if self.scaler_path is not None:
                self._load_scaler(self.scaler_path)
            else:
                raise ValueError("Either scaler or scaler_path should be provided.")

    def _load_scaler(self, scaler_path):
        self.scaler = sk_load(scaler_path)

    def inverse_transform(self, y_pred):
        return self.scaler.inverse_transform(y_pred)

