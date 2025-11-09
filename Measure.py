import time, torch
from codecarbon import EmissionsTracker

class TimerEnergy:
    """
    Context to measure synchronized wall time and process-level gCO2e (proxy).
    """
    def __init__(self, measure_power_secs: float = 1.0):
        self.measure_power_secs = measure_power_secs
        self._t0 = None
        self._t1 = None
        self.tracker = None
        self.emissions_gco2 = 0.0

    def __enter__(self):
        self.tracker = EmissionsTracker(
            measure_power_secs=self.measure_power_secs,
            tracking_mode="process",
            gpu_ids="0",
        )
        self.tracker.start()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t1 = time.perf_counter()
        try:
            self.emissions_gco2 = float(self.tracker.stop() or 0.0)
        except Exception:
            self.emissions_gco2 = 0.0

    @property
    def seconds(self):
        if self._t0 is None or self._t1 is None:
            return 0.0
        return self._t1 - self._t0
