from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
import time

flows = defaultdict(list)

def packet_callback(packet):
    if IP in packet:
        src = packet[IP].src
        dst = packet[IP].dst

        proto = "TCP" if TCP in packet else "UDP" if UDP in packet else "OTHER"

        key = (src, dst, proto)

        flows[key].append({
            "time": time.time(),
            "length": len(packet)
        })

def start_sniffing():
    sniff(prn=packet_callback, store=0)