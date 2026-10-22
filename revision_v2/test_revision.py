"""Regression checks for the scientific failure modes named in the review."""
from pathlib import Path
import json, tempfile, unittest
import numpy as np
import pandas as pd
import torch
from engine import Data, Network, simplex_mae, DEVICE
from aggregate import holm, paired_inference
from prepare import feature_bank, graph_vectors
from multihorizon import forecast_covariate_update


class RevisionChecks(unittest.TestCase):
    def test_feature_selection_ignores_future_and_removes_duplicates(self):
        data = object.__new__(Data)
        nonlinear = np.tile([-2., -1., 1., 2.], 20)
        noise = np.random.default_rng(42).normal(size=80)
        data.x = np.column_stack([nonlinear, nonlinear, np.ones(80), noise]).astype('float32')
        data.y = nonlinear ** 2
        data.origins = np.arange(80); data.train = np.arange(40)
        data.customers = np.arange(80) % 4
        data.names = ['nonlinear', 'duplicate', 'constant', 'noise']
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            before = data.rank(first)
            data.x[40:] *= 1000; data.y[40:] += 9999
            after = data.rank(second)
        self.assertEqual(before, after)
        self.assertIn('duplicate', before['removed'])
        self.assertIn('constant', before['removed'])
        self.assertGreater(before['relevance']['nonlinear'], 0)

    def test_recursive_context_updates_from_predictions(self):
        baskets = pd.DataFrame({'sid': [1]*6, 'customer': [1]*6, 'basket': range(6),
            'order': range(6), 't': [0, 1, 3, 6, 10, 15], 'amount': [2., 3., 4., 5., 6., 7.],
            'quantity': np.ones(6), 'item_count': np.ones(6), 'dow': [5, 6, 1, 4, 1, 6], 'month': np.ones(6)})
        _, x, names, *_ = feature_bank(baskets, baskets, 'OnlineRetailII')
        data = object.__new__(Data)
        data.x = x; data.names = names; data.mean = np.zeros(len(names)); data.sd = np.ones(len(names))
        data.samples = pd.DataFrame({'t': [15.]}); data.manifest = {'calendar_boundaries': ['2000-01-01']}
        gaps = np.full((1, 20), np.nan); gaps[0, -5:] = [1, 2, 3, 4, 5]
        amounts = np.full((1, 20), np.nan); amounts[0, -6:] = [2, 3, 4, 5, 6, 7]
        raw, gaps, amounts, transformed = forecast_covariate_update(data,
            {'columns': list(range(len(names))), 'K': 0}, np.array([0]), x[-1:].copy(),
            gaps, amounts, np.array([2.]), np.array([2.]))
        self.assertAlmostEqual(raw[0, names.index('gap_mean')], 17/6, places=5)
        self.assertEqual(raw[0, names.index('last_gap')], 2)
        self.assertEqual(raw[0, names.index('gap_lag2')], 5)
        self.assertEqual(raw[0, names.index('events_seen')], 7)
        self.assertEqual(raw[0, names.index('last_amount')], 7)
        self.assertTrue(np.isfinite(transformed).all())

    def test_exact_simplex_solution(self):
        p = np.array([[0., 10.], [10., 0.]])
        w = simplex_mae(p, np.array([2., 8.]))
        np.testing.assert_allclose(w, [.8, .2], atol=1e-7)
        self.assertTrue((w >= 0).all())

    def test_likelihood_zero_day_and_incomplete_gradients(self):
        for kind in ['deepar', 'weibull']:
            model = Network(5, {'kind': kind}, 8., 1., 1.).to(DEVICE)
            raw = torch.zeros((4, 2 if kind == 'deepar' else 1), device=DEVICE, requires_grad=True)
            loss = model.objective(raw, torch.tensor([0., 1., 5., 10.], device=DEVICE),
                torch.zeros(4, device=DEVICE), torch.tensor([True, True, False, False], device=DEVICE))
            self.assertTrue(torch.isfinite(loss)); loss.backward()
            self.assertTrue(torch.isfinite(raw.grad).all())
            if kind == 'weibull':
                self.assertTrue(torch.isfinite(model.shape_raw.grad))

    def test_padding_and_real_zero_gap_remain_distinct(self):
        data = object.__new__(Data)
        data.x = np.array([[10., 2.], [20., 4.], [30., 6.]], dtype='float32')
        data.mean = np.zeros(2); data.sd = np.ones(2)
        data.origins = np.array([2]); data.starts = np.zeros(3, dtype=int)
        data.graph_row = np.arange(3); data.gaps = np.array([0., 0., 5.])
        data.graph = lambda maximum: np.ones((3, 8), dtype='float32')
        cfg = {'columns': [0, 1], 'L': 5, 'K': 0, 'd': 8, 'P': 600, 'seed': 42}
        x, g, valid, anchor = data.batch(np.array([0]), cfg)
        np.testing.assert_array_equal(valid, [[False, False, True, True, True]])
        np.testing.assert_array_equal(x[0, :2], 0)
        np.testing.assert_array_equal(g[0, :2], 0)
        np.testing.assert_array_equal(x[0, :, -1], [0, 0, 1, 1, 1])
        # The real zero-day second purchase contributes to the median anchor;
        # the undefined first-event gap and padded rows do not.
        np.testing.assert_allclose(anchor, [2.5])

    def test_holm_adjustment(self):
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])

    def test_customer_bootstrap_identical_predictions(self):
        y = np.array([1., 2., 4., 8.]); p = np.array([2., 2., 3., 6.])
        result = paired_inference(y, {'selected': p, 'same': p.copy()},
                                 np.array([1, 1, 2, 3]), bootstrap_count=12, permutation_count=19)[0]
        for metric in ['MAE', 'RMSE', 'MedAE']:
            self.assertEqual(result[f'{metric}_reference_minus_selected'], 0)
            self.assertEqual(result[f'{metric}_low'], 0)
            self.assertEqual(result[f'{metric}_high'], 0)
        self.assertEqual(result['MAE_p_raw'], 1)

    def test_future_products_cannot_enter_fitted_graph(self):
        baskets = pd.DataFrame({'basket': range(7), 'phase': [0, 0, 0, 1, 2, 3, 4]})
        edges = pd.DataFrame({'basket': [0, 0, 1, 1, 2, 2, 3, 4, 5, 6],
                              'product': ['a', 'b', 'b', 'c', 'a', 'c', 'future', 'a', 'b', 'c']})
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            graph_vectors(baskets, edges, out)
            for maximum in [100, 300, 600, 1000]:
                products = json.loads((out / f'graph_P{maximum}_products.json').read_text())
                self.assertNotIn('future', products)
                vectors = np.load(out / f'graph_P{maximum}.npy')
                np.testing.assert_allclose(vectors[3], 0)
                embedding = np.load(out / f'graph_P{maximum}_embedding.npy')
                for column in embedding.T:
                    if np.any(column):
                        self.assertGreaterEqual(column[np.argmax(np.abs(column))], 0)


if __name__ == '__main__':
    unittest.main()
