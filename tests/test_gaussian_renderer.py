"""Run after building the local CUDA extension: python -m unittest discover -s tests."""
import importlib.util
from pathlib import Path
import unittest
import warnings

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA PyTorch required')
class GaussianRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = Path(__file__).resolve().parents[1] / 'models/gusd/localagg_prob_fast/local_aggregate_prob_fast'
        spec = importlib.util.spec_from_file_location('renderer_under_test', package / '__init__.py',
                                                     submodule_search_locations=[str(package)])
        import sys
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.aggregator = module.LocalAggregator

    def test_runtime_tiles_and_gradients(self):
        torch.manual_seed(37)
        grid = (9, 11, 7)
        indices = torch.cartesian_prod(*(torch.arange(size, device='cuda') for size in grid))
        points = (indices.float() + 0.5) * 0.1
        means = torch.tensor([[0.31, 0.42, 0.23], [0.54, 0.61, 0.36]], device='cuda')
        scales = torch.tensor([[0.12, 0.18, 0.14], [0.16, 0.11, 0.19]], device='cuda')
        opas = torch.tensor([0.6, 0.4], device='cuda')
        semantics = torch.randn(2, 17, device='cuda').softmax(-1)
        covariance = torch.diag_embed(scales.square().reciprocal())
        upstream = (torch.randn(len(points), 17, device='cuda') * 0.01,
                    torch.randn(len(points), device='cuda') * 0.01,
                    torch.randn(len(points), device='cuda') * 0.01)
        outputs, gradients = [], []
        for tile in (2, 4, 8):
            leaves = [tensor.clone().requires_grad_() for tensor in (means, opas, semantics, covariance)]
            model = self.aggregator(3, *grid, [0, 0, 0], 0.1, tile_size=tile).cuda()
            out = model(points[None], leaves[0][None], leaves[1][None], leaves[2][None],
                        scales[None], leaves[3][None])
            sum((prediction * weight).sum() for prediction, weight in zip(out, upstream)).backward()
            outputs.append(out)
            gradients.append([leaf.grad for leaf in leaves])
        for result in outputs[1:]:
            self.assertTrue(all(torch.equal(actual, expected) for actual, expected in zip(result, outputs[0])))
        for grads in gradients[1:]:
            for actual, expected in zip(grads, gradients[0]):
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.allclose(actual, expected, rtol=2e-4, atol=2e-6))

    def test_legacy_options_and_empty_gaussians(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = self.aggregator(3, 9, 11, 7, [0, 0, 0], 0.1, chunk_num=(2, 2, 2), chunk_margin=5).cuda()
            self.assertEqual(model.tile_size, 4)
            self.assertEqual(len(caught), 1)
        self.assertEqual(list(model.state_dict()), ['pc_min'])
        points = torch.ones(1, 3, 3, device='cuda') * 0.05
        means = torch.empty(1, 0, 3, device='cuda', requires_grad=True)
        opas = torch.empty(1, 0, device='cuda', requires_grad=True)
        semantics = torch.empty(1, 0, 17, device='cuda', requires_grad=True)
        cov = torch.empty(1, 0, 3, 3, device='cuda', requires_grad=True)
        scales = torch.empty(1, 0, 3, device='cuda')
        out = model(points, means, opas, semantics, scales, cov)
        sum(value.sum() for value in out).backward()
        for leaf in (means, opas, semantics, cov):
            self.assertIsNotNone(leaf.grad)
        self.assertTrue(torch.equal(out[1], torch.zeros_like(out[1])))


if __name__ == '__main__':
    unittest.main()