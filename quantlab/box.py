"""Exact selected-class stability on a supplied closed input box.

Coefficients are the represented source numbers and stored QLB1 scales. This
checks real linear maps, not floating-point accumulation or feature extraction.
"""
from fractions import Fraction as F

from .inspector import finite, ratio, validate_input, winner
from .packed import validate_quantized


def box_audit(weights, packed, lower, upper, selected):
    """Return every attained worst pairwise margin for source and packed maps.

    For a fixed selected class, these conditions are necessary and sufficient
    on this box, including first-index ties. A failing rival's corner need not
    make that rival the winner, but always makes the selected class lose.
    """
    if type(lower) is not list or type(upper) is not list:
        raise ValueError('box endpoints must be lists')
    source, lo = validate_input({'weights': weights, 'x': lower})
    if len(upper) != len(lo):
        raise ValueError('box endpoint dimensions differ')
    hi = [finite(v) for v in upper]
    if any(a > b for a, b in zip(lo, hi)):
        raise ValueError('lower exceeds upper')
    rows, cols = validate_quantized(packed)
    if rows != len(source) or cols != len(lo):
        raise ValueError('source and packed shapes differ')
    if type(selected) is not int or not 0 <= selected < rows:
        raise ValueError('selected class outside row range')
    originals = [[F(v) for v in row] for row in source]
    restored = [[q * F(packed['scales'][0 if packed['mode'] == 'per-tensor' else i])
                 for q in row] for i, row in enumerate(packed['q'])]
    low, high = list(map(F, lo)), list(map(F, hi))

    def audit(matrix):
        pairs = []
        for rival in range(rows):
            if rival == selected:
                continue
            difference = [a-b for a, b in zip(matrix[selected], matrix[rival])]
            corner = [a if coefficient >= 0 else b
                      for coefficient, a, b in zip(difference, low, high)]
            terms = [a*b for a, b in zip(difference, corner)]
            minimum = sum(terms, F(0))
            strict = rival < selected
            safe = minimum > 0 if strict else minimum >= 0
            scores = [sum((a*b for a, b in zip(row, corner)), F(0)) for row in matrix]
            pairs.append({'rival': rival, 'strict_margin_required': strict,
                          'difference_coefficients': [ratio(v) for v in difference],
                          'minimizing_corner': [ratio(v) for v in corner],
                          'margin_contributions': [ratio(v) for v in terms],
                          'minimum_margin': ratio(minimum), 'condition_holds': safe,
                          'corner_scores': [ratio(v) for v in scores],
                          'corner_prediction': winner(scores)})
        return {'selected_wins_everywhere': all(p['condition_holds'] for p in pairs),
                'pairs': pairs}

    a, b = audit(originals), audit(restored)
    return {'schema': 'quantlab-box/1', 'selected': selected,
            'lower': [ratio(v) for v in low], 'upper': [ratio(v) for v in high],
            'source': a, 'packed': b,
            'both_maps_keep_selected_everywhere': a['selected_wins_everywhere'] and b['selected_wins_everywhere'],
            'scope': 'exact represented-real source and packed-weight linear maps on the supplied closed box; not floating-point accumulation, activation quantization, learned uncertainty, unseen accuracy or a hardware benchmark'}
