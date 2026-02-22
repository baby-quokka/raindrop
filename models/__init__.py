from .Restormer import *
from .ConvIR import ConvIR
from .NAFNet import NAFNet, NAFNet_width32, NAFNet_width48, NAFNet_width64
from .Uformer import Uformer
from .IDT import IDT
from .PWFNet import PWFNet
from .SwinIR import SwinIR
from .MambaIR import MambaIR, MambaIR_Tiny, MambaIR_Small, MambaIR_Base
from .Restormer import Restormer, Restormer_Small, Restormer_Base, Restormer_Base_MultiScale

# Diffusion 모델들 (별도 학습 스크립트 train_diffusion.py 사용)
from .DiffIR import DiffIR, DiffIR_Small, DiffIR_Base
# from .SimpleDiffusion import SimpleDiffusion

# STRRNet - NTIRE 2025 1st Place (Miracle Team)
from .STRRNet import STRRNet, STRRNet_Base, STRRNet_Small, STRRNet_NoSemantic, SemanticClassifier


# 편의를 위한 ConvIR 버전별 래퍼
def ConvIR_base():
    return ConvIR(version="base")


def ConvIR_large():
    return ConvIR(version="large")
