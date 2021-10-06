# from LightingModule.VAE import VAE
from LightingModule.BaseClassifier import BaseClassifier
from LightingModule.NoiseInjection import NoiseInjection
from LightingModule.NoiseInj_resnet import NoiseInj_resnet
# from LightingModule.CVAE_mixup import CVAE
from LightingModule.NoiseDist import NoiseDist
from LightingModule.NoiseGeneration import NoiseGeneration
from LightingModule.CutMix import CutMix
from LightingModule.Style import Style


model_dict = {
    # 'vae': VAE,
    'BaseClassifier': BaseClassifier,
    'NoiseInjection': NoiseInjection,
    "NoiseInj_resnet": NoiseInj_resnet,
    'NoiseDist': NoiseDist,
    # "CVAE_mixup": CVAE,
    'NoiseGeneration': NoiseGeneration,
    'CutMix': CutMix,
    'Style': Style,
}