import json
import numpy as np
import pandas as pd
from mbi import Dataset, Domain


class Mechanism:
    """
    Generic mechanism base class for categorical/discretized tabular data.
    This Adult-compatible version removes Colorado-specific transformations.
    """

    def __init__(self, dataset, specs=None, domain_path=None):
        self.dataset = dataset
        self.specs = {}

        if specs is not None and str(specs).strip() not in ("", "None"):
            with open(specs, "r", encoding="utf-8") as f:
                self.specs = json.load(f)

        df = pd.read_csv(dataset).copy()
        self.column_order = list(df.columns)

        # Optional: use a pre-defined discrete domain
        if domain_path is not None and str(domain_path).strip() not in ("", "None"):
            with open(domain_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.domain = Domain.fromdict({k: int(v) for k, v in cfg.items()})

            # If domain_path is used, we assume data is already integer-coded.
            self.value_maps = {}
            self.inverse_maps = {}
        else:
            # Build per-column category mapping from dataset values
            self.value_maps = {}
            self.inverse_maps = {}
            domain_cfg = {}

            for col in self.column_order:
                vals = pd.Series(df[col]).astype(str).fillna("NA").tolist()
                uniq = sorted(set(vals))
                mp = {v: i for i, v in enumerate(uniq)}
                inv = {i: v for v, i in mp.items()}

                self.value_maps[col] = mp
                self.inverse_maps[col] = inv
                domain_cfg[col] = len(uniq)

            self.domain = Domain.fromdict(domain_cfg)

    def setup(self):
        pass

    def load_data(self, path=None):
        if path is None:
            path = self.dataset

        df = pd.read_csv(path).copy()

        # If mappings exist, encode from raw strings to integers.
        if self.value_maps:
            for col, mp in self.value_maps.items():
                s = pd.Series(df[col]).astype(str).fillna("NA")
                # Unknown values fallback to 0 (rare; mostly for train/test mismatch)
                df[col] = s.map(lambda x: mp.get(x, 0)).astype(int)
        else:
            # Pre-discretized mode
            for col in self.domain.attrs:
                df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

        return Dataset(df[list(self.domain.attrs)], self.domain)

    def measure(self):
        pass

    def postprocess(self):
        pass

    def transform_domain(self):
        """
        Decode integer-coded synthetic data back to original string values
        when inverse mapping is available.
        """
        df = self.synthetic.df.copy()

        if self.inverse_maps:
            for col, inv in self.inverse_maps.items():
                if col in df.columns:
                    df[col] = df[col].map(
                        lambda x: inv.get(int(x), inv.get(0, "NA")))

        for col in self.column_order:
            if col not in df.columns:
                df[col] = 0

        self.synthetic = df[self.column_order]

    def run(self, epsilon, delta=1e-9, save=None):
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.save = save

        self.setup()
        self.measure()
        self.postprocess()
        self.transform_domain()

        if save is not None:
            self.synthetic.to_csv(save, index=False)
        return self.synthetic
