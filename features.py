import pandas as pd

def extract_features(flow):
    times = [p["time"] for p in flow]
    lengths = [p["length"] for p in flow]

    duration = max(times) - min(times) if len(times) > 1 else 0

    return {
        "Flow Duration": duration,
        "Tot Fwd Pkts": len(flow),
        "TotLen Fwd Pkts": sum(lengths)
    }