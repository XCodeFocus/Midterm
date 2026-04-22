from mechanism import Mechanism
import mbi
from mbi import Domain, Dataset
from mbi.marginal_loss import LinearMeasurement
import matrix
import argparse
import numpy as np
import jax.numpy as jnp
from scipy import sparse, optimize
from functools import reduce
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def gaussian_rdp(order, sigma):
    return order / (2.0 * sigma * sigma)


def rdp_to_epsilon(rdp, orders, delta):
    eps = rdp + np.log(1.0 / delta) / (orders - 1.0)
    return float(np.min(eps))

def transform_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        size = support.sum()
        newdom[col] = int(size)
        if size < support.size:
            newdom[col] += 1
        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size
        df[col] = df[col].map(mapping).astype(int)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


def compressed_query(support, extra, scale=1.0):
    support_idx = jnp.asarray(np.where(support)[0], dtype=int)
    extra_idx = jnp.asarray(np.where(~support)[0], dtype=int)

    def query(factor):
        values = factor.datavector(flatten=False)
        if extra_idx.size == 0:
            return values[support_idx] / scale
        tail = values[extra_idx].sum() / scale
        return jnp.concatenate([values[support_idx], jnp.expand_dims(tail, 0)])

    return query

def reverse_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        mx = support.sum()
        newdom[col] = int(support.size)
        idx, extra = np.where(support)[0], np.where(~support)[0]
        mask = df[col] == mx
        if extra.size == 0:
            pass
        else:
            df.loc[mask, col] = np.random.choice(extra, mask.sum())
        df.loc[~mask, col] = idx[df.loc[~mask, col]]
        df[col] = df[col].astype(int)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)

def moments_calibration(round1, round2, eps, delta):
    orders = np.arange(2, 128)
    
    def obj(sigma):
        rdp1 = gaussian_rdp(orders, sigma / round1)
        rdp2 = gaussian_rdp(orders, sigma / round2)
        rdp = rdp1 + rdp2
        return rdp_to_epsilon(rdp, orders, delta) - eps + 1e-8
    
    sigma_min, sigma_max = 0.1, 1000
    if obj(sigma_min) > 0: return sigma_min
    if obj(sigma_max) < 0: return sigma_max

    res = optimize.root_scalar(obj, bracket=[sigma_min, sigma_max])
    return res.root

class Match3(Mechanism):

    def __init__(self, dataset, specs=None, iters=1000, weight3=1.0, warmup=False, domain_path=None):
        if domain_path is None:
            domain_path = str(PROJECT_ROOT / 'data' / 'adult-domain.json')
        Mechanism.__init__(self, dataset, specs, domain_path=domain_path)
        self.iters = iters
        self.weight3 = weight3
        self.warmup = warmup
        self.elimination_order = None

    def setup(self):
        self.round1 = list(self.domain.attrs)
        candidate_cliques = [
            ('age', 'education-num'),
            ('marital-status', 'race'),
            ('sex', 'hours-per-week'),
            ('hours-per-week', 'income>50K'),
            ('native-country', 'marital-status', 'occupation'),
            ('education-num', 'occupation'),
            ('sex', 'occupation'),
            ('age', 'income>50K'),
        ]
        attrs = set(self.domain.attrs)
        self.round2 = [cl for cl in candidate_cliques if set(cl) <= attrs]

    def measure(self):
        data = self.load_data()
        # round1 and round2 measurements will be weighted to have L2 sensitivity 1
        sigma = moments_calibration(1.0, 1.0, self.epsilon, self.delta)
        print('NOISE LEVEL:', sigma)

        weights = np.ones(len(self.round1), dtype=float)
        if "income>50K" in self.round1:
            weights[self.round1.index("income>50K")] *= 1.5
        weights /= np.linalg.norm(weights) # now has L2 norm = 1

        supports = {}
  
        self.measurements = []
        for col, wgt in zip(self.round1, weights):
            ##########################
            ### Noise-addition step ##
            ##########################
            proj = (col,)
            hist = data.project(proj).datavector()
            noise = sigma*np.random.randn(hist.size)
            y = wgt*hist + noise
          
            #####################
            ## Post-processing ##
            #####################
            if self.domain.size(col) <= 16:
                sup = np.ones(y.size, dtype=bool)
            else:
                sup = y >= 3 * sigma

            supports[col] = sup
            print(col, self.domain.size(col), sup.sum())

            if sup.sum() == y.size:
                y2 = y
                query = lambda factor: factor.datavector(flatten=False)
            else:
                y2 = np.append(y[sup], y[~sup].sum())
                scale = np.sqrt(y.size - y2.size + 1.0)
                y2[-1] /= scale
                query = compressed_query(sup, ~sup, scale=scale)

            self.measurements.append(
                LinearMeasurement(y2 / wgt, clique=proj, stddev=1.0 / wgt, query=query)
            )

        self.supports = supports 
        # perform round 2 measurments over compressed domain
        data = transform_data(data, supports)
        self.domain = data.domain

        self.round2 = [cl for cl in self.round2 if self.domain.size(cl) < 1e6]
        weights = np.ones(len(self.round2), dtype=float)
        if ('native-country', 'marital-status', 'occupation') in self.round2:
            weights[self.round2.index(('native-country', 'marital-status', 'occupation'))] *= self.weight3
        weights /= np.linalg.norm(weights) # now has L2 norm = 1
   
        for proj, wgt in zip(self.round2, weights):
            #########################
            ## Noise-addition step ##
            #########################
            hist = data.project(proj).datavector()
            noise = sigma*np.random.randn(hist.size)
            y = wgt*hist + noise
            self.measurements.append(
                LinearMeasurement(y / wgt, clique=proj, stddev=1.0 / wgt)
            )

    def postprocess(self):
        callback = mbi.callbacks.default(self.measurements, frequency=50)
        self.model = mbi.estimation.mirror_descent(
            self.domain,
            self.measurements,
            iters=self.iters,
            callback_fn=callback,
        )
        self.synthetic = self.model.synthetic_data()
        self.synthetic = reverse_data(self.synthetic, self.supports)

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = str(PROJECT_ROOT / 'data' / 'adult.csv')
    params['specs'] = None
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
    params['save'] = str(PROJECT_ROOT / 'outputs' / 'adult_dp_rmckenna.csv')
    
    return params

if __name__ == '__main__':

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', help='path to dataset csv file')
    parser.add_argument('--specs', help='path to specs json file', default=None)
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')
    parser.add_argument('--save', help='path to save synthetic data to')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    if args.epsilon <= 0.3:
        iters = 7500
        weight3 = 8.0
    elif args.epsilon >= 4.0:
        iters = 10000
        weight3 = 4.0
    else:
        iters = 7500
        weight3 = 6.0

    mech = Match3(args.dataset, args.specs, iters=iters, weight3=weight3, warmup=True, domain_path=str(PROJECT_ROOT / 'data' / 'adult-domain.json'))

    mech.run(args.epsilon, args.delta, args.save)
