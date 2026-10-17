"""Global EMA Validation selection, independent of native stage restart decisions."""
import csv
import json
import math
from pathlib import Path


def read_selection(directory):
    path = Path(directory) / 'selection.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'ap': -1.0, 'epoch': -1}


def is_better(ap, selection):
    # Epochs are visited in order: exact AP ties retain the earlier checkpoint.
    return math.isfinite(ap) and ap > selection['ap']


def record_validation(directory, epoch, metrics, selection):
    directory = Path(directory)
    path = directory / 'epoch_validation.csv'
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(['epoch', 'ema_validation_ap', 'ap50', 'ap75', 'ap_small', 'ap_medium', 'ap_large'])
        writer.writerow([epoch, *metrics[:6]])
    (directory / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    info_path = directory / 'run_info.json'
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding='utf-8'))
        info.update(best_validation_epoch=selection['epoch'], best_validation_ap=selection['ap'])
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
