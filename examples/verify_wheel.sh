#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
python -m venv "$TMP/venv"
"$TMP/venv/bin/pip" --disable-pip-version-check install --no-deps "$ROOT"
cd "$TMP"
"$TMP/venv/bin/python" - <<'PY'
import importlib
import importlib.resources as resources
import json
from pathlib import Path
from quantlab.cli import _train_fixture
from quantlab.model import featurize
from quantlab.analysis import imbalanced_matrix
from quantlab.inspector import inspect
from quantlab.packed import decode
for name in ('quantlab','quantlab.quantize','quantlab.model','quantlab.analysis','quantlab.cli','quantlab.packed','quantlab.inspector'):
    importlib.import_module(name)
fixture=json.loads(resources.files('quantlab').joinpath('data/fixture.json').read_text())
witnesses=[w for mode in ('per-tensor','per-channel') for bits in (32,64) for w in json.loads(resources.files('quantlab').joinpath(f'data/witnesses-{mode}-{bits}.json').read_text())]
clf,train,test=_train_fixture(0,64,5)
assert clf.weights==fixture['weights']
assert [dict(s,x=featurize(s['words'],64)) for s in train]==fixture['train']
assert [dict(s,x=featurize(s['words'],64)) for s in test]==fixture['test']
assert imbalanced_matrix()==fixture['study']
for w in witnesses:
    r=inspect(w['value'],w['mode'],w['scale_bits'])
    assert r==w['report']
    assert decode(bytes.fromhex(r['payload_hex']))['rows']==len(w['value']['weights'])
Path('input.json').write_text(json.dumps({'weights':clf.weights,'x':fixture['test'][0]['x']}))
print('Installed outside checkout: 7 modules; complete fixture retraining; 52 full reports')
PY
"$TMP/venv/bin/quantlab" evaluate --max-accuracy-drop 0 --min-ratio 3.5
"$TMP/venv/bin/quantlab" analyze --min-snr-gain 10
"$TMP/venv/bin/quantlab" json > original.json
"$TMP/venv/bin/quantlab" inspect --input input.json > report.json
"$TMP/venv/bin/quantlab" pack --input input.json --output weights.qlb
"$TMP/venv/bin/quantlab" unpack --input weights.qlb > decoded.json

# New module and reports are verified with the installed interpreter outside checkout.
"$TMP/venv/bin/python" "$ROOT/scripts/verify_integer.py"
"$TMP/venv/bin/python" - <<'PY2'
import json,importlib.resources as r
from pathlib import Path
c=json.loads(r.files('quantlab').joinpath('data/integer-0.json').read_text())[0]
Path('integer-input.json').write_text(json.dumps(c['input']))
PY2
"$TMP/venv/bin/quantlab" integer --input integer-input.json > integer-console.json
"$TMP/venv/bin/python" -m quantlab.cli integer --input integer-input.json > integer-module.json
cmp integer-console.json integer-module.json

"$TMP/venv/bin/python" - <<'PYBOX'
from fractions import Fraction as F
import importlib
import importlib.resources as resources
import json
from quantlab.box import box_audit
from quantlab.packed import decode
m=importlib.import_module('quantlab.box')
assert 'site-packages' in m.__file__
count=0
for mode in ('per-tensor','per-channel'):
    for bits in (32,64):
        witnesses=json.loads(resources.files('quantlab').joinpath(f'data/witnesses-{mode}-{bits}.json').read_text())
        for witness in witnesses:
            weights=witness['value']['weights']; x=witness['value']['x']
            packed=decode(bytes.fromhex(witness['report']['payload_hex']))['packed']
            scores=[sum((F(float(a))*F(float(b)) for a,b in zip(row,x)),F(0)) for row in weights]
            selected=max(range(len(scores)),key=lambda i:scores[i])
            for radius in (0., .01, .25):
                lower=[max(-1e6,float(v)-radius) for v in x]
                upper=[min(1e6,float(v)+radius) for v in x]
                report=box_audit(weights,packed,lower,upper,selected)
                for key in ('source','packed'):
                    for pair in report[key]['pairs']:
                        values=list(map(F,pair['corner_scores']))
                        assert F(pair['minimum_margin'])==values[selected]-values[pair['rival']]
                        assert pair['corner_prediction']==max(range(len(values)),key=lambda i:values[i])
                        if not pair['condition_holds']:assert pair['corner_prediction']!=selected
                if radius==0:assert report['source']['selected_wins_everywhere']
                count+=1
assert count==156
print('Installed box module and 156 complete archived source/packed box reports verified outside checkout')
PYBOX
