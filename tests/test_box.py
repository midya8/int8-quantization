"""Independent Cartesian-corner checks for supplied-box linear decisions."""
import copy
from fractions import Fraction as F
import itertools
import math
import random
import unittest

from quantlab.box import box_audit
from quantlab.packed import encode, decode
from quantlab.quantize import quantize_per_channel, quantize_tensor


def first_max(values):
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def evaluate(matrix, point):
    return [sum((a*b for a, b in zip(row, point)), F(0)) for row in matrix]


class BoxTests(unittest.TestCase):
    def check(self, weights, packed, lower, upper, selected):
        report = box_audit(weights, packed, lower, upper, selected)
        source = [[F(float(x)) for x in row] for row in weights]
        restored = [[F(q)*F(packed['scales'][0 if packed['mode']=='per-tensor' else i])
                     for q in row] for i, row in enumerate(packed['q'])]
        corners = list(itertools.product(*[(F(float(a)), F(float(b))) for a, b in zip(lower, upper)]))
        for key, matrix in [('source', source), ('packed', restored)]:
            all_scores = [evaluate(matrix, corner) for corner in corners]
            expected = all(first_max(scores)==selected for scores in all_scores)
            self.assertEqual(report[key]['selected_wins_everywhere'], expected)
            for pair in report[key]['pairs']:
                rival = pair['rival']
                margins = [scores[selected]-scores[rival] for scores in all_scores]
                self.assertEqual(F(pair['minimum_margin']), min(margins))
                corner = list(map(F, pair['minimizing_corner']))
                self.assertIn(tuple(corner), corners)
                scores = evaluate(matrix, corner)
                self.assertEqual(scores[selected]-scores[rival], min(margins))
                self.assertEqual(list(map(F, pair['corner_scores'])), scores)
                self.assertEqual(pair['corner_prediction'], first_max(scores))
                self.assertEqual(sum(map(F, pair['margin_contributions'])), min(margins))
                if not pair['condition_holds']:
                    self.assertNotEqual(first_max(scores), selected)
        self.assertEqual(report['both_maps_keep_selected_everywhere'],
                         report['source']['selected_wins_everywhere'] and report['packed']['selected_wins_everywhere'])
        return report

    def test_generated_all_corners_classes_modes_and_scale_precisions(self):
        rng = random.Random(630)
        count = 0
        for case in range(80):
            rows, cols = 2 + case % 3, 1 + case % 5
            weights = [[rng.randint(-100, 100)/16 for _ in range(cols)] for _ in range(rows)]
            lower = [rng.randint(-16, 16)/8 for _ in range(cols)]
            upper = [v + rng.randint(0, 16)/8 for v in lower]
            for quantize in (quantize_tensor, quantize_per_channel):
                for bits in (32, 64):
                    packed = decode(encode(quantize(weights), bits))['packed']
                    for selected in range(rows):
                        self.check(weights, packed, lower, upper, selected)
                        count += 1
        self.assertEqual(count, 956)

    def test_first_index_ties_and_point_boxes(self):
        weights = [[1., -1.], [1., -1.], [-1., 1.]]
        packed = {'mode': 'per-tensor', 'q': [[1, -1], [1, -1], [-1, 1]], 'scales': [1.]}
        a = self.check(weights, packed, [1., 0.], [2., 0.], 0)
        b = self.check(weights, packed, [1., 0.], [2., 0.], 1)
        self.assertTrue(a['both_maps_keep_selected_everywhere'])
        self.assertFalse(b['both_maps_keep_selected_everywhere'])
        self.assertEqual(b['source']['pairs'][0]['minimum_margin'], '0/1')
        for selected in range(3):
            self.check(weights, packed, [-0., 0.], [0., 0.], selected)

    def test_quantization_can_break_source_box_and_rival_need_not_win(self):
        weights = [[0., 1.], [0.49, 1.], [-1., -1.]]
        packed = decode(encode(quantize_tensor(weights), 32))['packed']
        # Use a direct coarse artifact to isolate the changed decision.
        coarse = {'mode': 'per-tensor', 'q': [[0, 1], [0, 1], [-1, -1]], 'scales': [1.]}
        r = self.check(weights, coarse, [1., 1.], [2., 2.], 1)
        self.assertTrue(r['source']['selected_wins_everywhere'])
        self.assertFalse(r['packed']['selected_wins_everywhere'])
        weights = [[0.], [1.], [3.]]
        p = {'mode': 'per-channel', 'q': [[0], [1], [3]], 'scales': [1., 1., 1.]}
        r = self.check(weights, p, [1.], [2.], 0)
        pair = r['source']['pairs'][0]
        self.assertEqual(pair['rival'], 1)
        self.assertEqual(pair['corner_prediction'], 2)
        self.check([[0., 1.], [0.49, 1.], [-1., -1.]], packed, [1., 1.], [2., 2.], 1)

    def test_extreme_finite_subnormal_scales_and_nonmutation(self):
        p = {'mode':'per-channel','q':[[127, -127], [-127, 127]],'scales':[math.ulp(0.), 1e6]}
        w = [[math.ulp(0.), -math.ulp(0.)], [-1e6, 1e6]]
        inputs = (w, p, [-1e6, -1e6], [1e6, 1e6])
        before = copy.deepcopy(inputs)
        for selected in (0, 1):
            self.check(*inputs, selected)
        self.assertEqual(inputs, before)

    def test_rejects_bad_boxes_classes_and_artifacts(self):
        w = [[1.], [-1.]]
        p = {'mode':'per-tensor','q':[[1], [-1]],'scales':[1.]}
        for low, high, selected in [([1.], [0.], 0), ([0.], [], 0), ([False], [1.], 0),
                ([0.], [float('nan')], 0), ([0.], [1e7], 0), ([0.], [1.], True),
                ([0.], [1.], -1), ([0.], [1.], 2), ((0.,), [1.], 0)]:
            with self.assertRaises(ValueError):
                box_audit(w, p, low, high, selected)
        for broken in [dict(p, extra=1), dict(p, q=[[1, 0], [-1, 0]]),
                       dict(p, scales=[0.]), dict(p, q=[[-128], [1]])]:
            with self.assertRaises(ValueError):
                box_audit(w, broken, [0.], [1.], 0)


if __name__ == '__main__':
    unittest.main()
