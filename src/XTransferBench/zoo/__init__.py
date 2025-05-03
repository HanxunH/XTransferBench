import copy
import json
import importlib.resources
from XTransferBench import attacker
from collections import OrderedDict

attacker_collections = {
    'linf_non_targeted': OrderedDict([]),
    'l2_non_targeted': OrderedDict([]),
    'unrestricted_non_targeted': OrderedDict([]),
    'linf_targeted': OrderedDict([]),
    'l2_targeted': OrderedDict([]),
}

with importlib.resources.open_text("XTransferBench.zoo.collections", "linf_non_targeted.json") as file:
    data = json.load(file)
    for key in data:
        attacker_name = key
        attacker_config = data[key]
        attacker_collections['linf_non_targeted'][attacker_name] = (attacker.xtransfer.XTransferLinf, attacker_config)

with importlib.resources.open_text("XTransferBench.zoo.collections", "linf_targeted.json") as file:
    data = json.load(file)
    for key in data:
        attacker_name = key
        attacker_config = data[key]
        attacker_collections['linf_targeted'][attacker_name] = (attacker.xtransfer.XTransferLinf, attacker_config)

with importlib.resources.open_text("XTransferBench.zoo.collections", "l2_non_targeted.json") as file:
    data = json.load(file)
    for key in data:
        attacker_name = key
        attacker_config = data[key]
        attacker_collections['l2_non_targeted'][attacker_name] = (attacker.xtransfer.XTransferL2Attack, attacker_config)

with importlib.resources.open_text("XTransferBench.zoo.collections", "l2_targeted.json") as file:
    data = json.load(file)
    for key in data:
        attacker_name = key
        attacker_config = data[key]
        attacker_collections['l2_targeted'][attacker_name] = (attacker.xtransfer.XTransferL2Attack, attacker_config)

with importlib.resources.open_text("XTransferBench.zoo.collections", "advclip.json") as file:
    data = json.load(file)
    for key in data:
        attacker_name = key
        attacker_config = data[key]
        attacker_collections['unrestricted_non_targeted'][attacker_name] = (attacker.advclip.AdvCLIP, attacker_config)

def list_threat_model():
    return attacker_collections.keys()

def list_attacker(threat_model="linf_non_targeted"):
    return list(attacker_collections[threat_model].keys())

def load_attacker(threat_model, method_name):
    method = attacker_collections[threat_model][method_name][0]
    configs = attacker_collections[threat_model][method_name][1]
    attacker = method(**configs)
    if "checkpoint_path" in configs:
        checkpoint_path = configs["checkpoint_path"]
        if checkpoint_path.startswith("hf-hub:"):
            attacker = attacker.from_pretrained(checkpoint_path.replace("hf-hub:", ""))
        else:
            attacker.load()
    return attacker