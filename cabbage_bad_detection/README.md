# cabbage_bad_detection

Обучил YOLOv8, класс `bad` — кочаны с дырами на листьях.

## Установка

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```
python scripts/prepare_data.py
python scripts/train.py --epochs 100 --imgsz 800 --batch 8 --device mps --name full_run
python scripts/predict_test.py --source ../test_photo --out results
```

`prepare_data.py` собирает списки в `data/`, датасет сам не меняет.

## Модель

`models/full_run_best.pt` — готовая, но в гит не залита (6 МБ, проще самому обучить).

Чтобы получить:
```
python scripts/train.py --epochs 100 --imgsz 800 --batch 8 --device mps --name full_run
```

На valid: precision 0.88, recall 0.92, mAP50 0.94.

На test_photo с высоты дрона находит нормально. Если снимать лист крупным планом в руке — почти ничего, таких фото в обучении не было.

Метрики прогона на test_photo — в `results/summary.json` (сами картинки в гит не заливал).
