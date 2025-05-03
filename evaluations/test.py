import XTransferBench
import XTransferBench.zoo
from pprint import pprint

for threat_model in XTransferBench.zoo.list_threat_model():
    pprint(XTransferBench.zoo.list_attacker(threat_model))
    for attacker in XTransferBench.zoo.list_attacker(threat_model):
        XTransferBench.zoo.load_attacker(threat_model, attacker)