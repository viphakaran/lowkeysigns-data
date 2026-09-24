import json

MANIFEST_PATH = "data/manifest.json"

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"{'Class':<15} | {'Attempted':<10} | {'Succeeded':<10} | {'Failed':<8} | {'Status':<12}")
print("-" * 65)

at_risk = []
emergency_stats = None

for gloss, stats in manifest.items():
    attempted = len(stats["attempted"])
    succeeded = len(stats["succeeded"])
    failed = len(stats["failed"])
    
    status = "OK"
    if succeeded < 5:
        status = "AT-RISK (<5)"
        at_risk.append((gloss, succeeded))
        
    if gloss == "emergency":
        emergency_stats = (attempted, succeeded, failed, status)
        
    print(f"{gloss:<15} | {attempted:<10} | {succeeded:<10} | {failed:<8} | {status:<12}")

print("-" * 65)

if emergency_stats:
    att, succ, fail, st = emergency_stats
    print(f"\n[EXPLICIT NOTE] 'emergency': attempted={att}, succeeded={succ}, failed={fail}, status={st}")

if at_risk:
    print(f"\n[WARNING] Classes with fewer than 5 succeeded samples: {[c[0] for c in at_risk]}")
else:
    print("\n[SUCCESS] All classes have at least 5 succeeded samples!")
