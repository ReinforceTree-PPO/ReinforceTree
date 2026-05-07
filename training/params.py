from model.Dynamic_network.DualPrompt import DualPrompt
from model.Regular.Tree_LoRA import Tree_LoRA
from model.Regular.HideLoRA import HideLoRA
try:
    from model.Regular.LwF import LwF
except ImportError:
    LwF = None
from model.Regular.EWC import EWC
try:
    from model.Regular.GEM import GEM
except ImportError:
    GEM = None
from model.Regular.OGD import OGD
from model.Replay.MbPAplusplus import MbPAplusplus
from model.Replay.LFPT5 import LFPT5
from model.Regular.O_LoRA import O_LoRA
from model.base_model import CL_Base_Model
from model.lora import lora

Method2Class = {"EWC"      : EWC,
                "OGD"      : OGD,
                "DualPrompt": DualPrompt,
                "MbPA++"   : MbPAplusplus,
                "LFPT5"    : LFPT5,
                "O_LoRA"   : O_LoRA,
                "Hide_LoRA" : HideLoRA,
                "Tree_LoRA": Tree_LoRA,
                "base"     : CL_Base_Model,
                "lora"     : lora}

# Add optional methods if dependencies are available
if GEM is not None:
    Method2Class["GEM"] = GEM
if LwF is not None:
    Method2Class["LwF"] = LwF

AllDatasetName = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

OLoRADatasetStandardName = ["dbpedia", "amazon", "yahoo", "agnews"]
