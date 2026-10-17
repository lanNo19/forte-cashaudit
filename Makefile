PY ?= python3
VIDEOS ?= ../data/open/videos
OUT ?= runs/latest
SHIFTED ?= ../data/shifted
URL ?= http://127.0.0.1:8000
KEY ?= dev
PREDS ?= runs/latest

.PHONY: install check test licences serve-stub serve-oracle serve-pipeline eval-local eval-api load-test shift-set calibrate

install:
	$(PY) -m pip install -r requirements.txt

check: licences test

licences:
	$(PY) tools/check_licences.py

test:
	$(PY) -m pytest -q tests

serve-stub:
	API_KEYS=dev ANALYZER=stub $(PY) -m uvicorn cashaudit.api:app --host 0.0.0.0 --port 8000

serve-oracle:
	API_KEYS=dev ANALYZER=oracle $(PY) -m uvicorn cashaudit.api:app --host 0.0.0.0 --port 8000

serve-pipeline:
	API_KEYS=$${API_KEYS:-dev} ANALYZER=pipeline $(PY) -m uvicorn cashaudit.api:app --host 0.0.0.0 --port 8000

eval-local:
	$(PY) tools/eval.py --local pipeline --videos $(VIDEOS) --out $(OUT)

eval-api:
	$(PY) tools/eval.py --api $(URL) --key $(KEY) --videos $(VIDEOS) --out $(OUT) --parallel 5

# the jury's worst case: the largest clip (~29 MB, as in FAQ 25) uploaded 5 times at once
load-test:
	@big=$$(ls -S $(VIDEOS)/*.mov | head -1); d=$$(mktemp -d); echo "load test: 5 x $$big"; \
	for i in 1 2 3 4 5; do ln -s "$$(realpath "$$big")" $$d/worst-$$i.mov; done; \
	$(PY) tools/eval.py --api $(URL) --key $(KEY) --files $$d/*.mov --out $(OUT)-load --parallel 5; rm -rf $$d

shift-set:
	$(PY) tools/make_shift_set.py --videos $(VIDEOS) --out $(SHIFTED)

calibrate:
	$(PY) tools/calibrate.py --preds $(PREDS)
