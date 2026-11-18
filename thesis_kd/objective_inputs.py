"""Study02 reference identities from the original M seed-42 Train/Val records."""
from copy import deepcopy
import hashlib
import json

from .factories import sha256_file


# Both archived m_base and m_kd resolved to this native recipe. Relocation of
# the dataset does not change the recipe; the annotations are checked separately.
NATIVE_SHA256 = '81c2bda65be447862e6ff4fee0353282cf89f018c3732ffe97ab55e05477c852'
ANNOTATION_SHA256 = {
    'train': '20690a67f4e4103c098f93c3a282f0f8287d3c51354ebb0f863518841a6fc656',
    'val': '4c4af5676ab7d5ccfc505f10094082bd819502ad06683b4cb9a701500516f440',
}
TEACHER_ID = '006e70cb31d5900040a8263727228bc0c1c731f7baccc1fb0f6754d01213050f/ema.module'
TEACHER_CONFIG_SHA256 = 'f2f28d996f2765b617f0a56c400614b75a95ec7d736bd445926e3f18e99b3c5d'


def native_fingerprint(native):
    recipe = deepcopy(native)
    for split in ANNOTATION_SHA256:
        dataset = recipe[split + '_dataloader']['dataset']
        del dataset['img_folder'], dataset['ann_file']
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def verify_annotation(split, path):
    digest = sha256_file(path)
    if digest != ANNOTATION_SHA256[split]:
        raise ValueError(f'Study02 {split} annotation differs from study01: {path} ({digest})')
    return digest


def verify_study02_inputs(cfg, identity, *, resume=False):
    if identity['id'] != TEACHER_ID or identity['config_digest'] != TEACHER_CONFIG_SHA256:
        raise ValueError('Study02 requires the study01 teacher EMA and training configuration')
    native = cfg['native']
    recipe = native_fingerprint(native)
    if recipe != NATIVE_SHA256:
        raise ValueError(f'Study02 native training recipe differs from study01: {recipe}')
    annotations = {split: verify_annotation(split, native[split + '_dataloader']['dataset']['ann_file'])
                   for split in ANNOTATION_SHA256}
    record = {'native_recipe_sha256': recipe, 'annotation_sha256': annotations}
    if not resume:
        record['student_init_sha256'] = {
            key: sha256_file(cfg['resources']['THESIS_KD_M_' + key])
            for key in ('BACKBONE_WEIGHTS', 'DETECTOR_CHECKPOINT')
        }
    return record
