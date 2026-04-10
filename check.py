import os, re

def check_file(filepath):
    print(f"\n🔍 {filepath}")
    if not os.path.exists(filepath):
        print("❌ FILE MISSING")
        return False
    with open(filepath, 'r') as f:
        content = f.read()
    
    issues = []
    if re.search(r'0\.001|0\.999', content):
        issues.append("⚠️  0.001/0.999 clamp (becomes 0.00/1.00 with :.2f)")
    if re.search(r'\.3f', content):
        issues.append("⚠️  .3f formatting")
    if re.search(r'reward=\s*\{[^}]*(?<!rewards=)', content):
        issues.append("⚠️  reward= singular (needs rewards=)")
    if re.search(r'print\(f"\[END\] score=\{.*?\}"', content):
        issues.append("⚠️  extra [END] score= line")
    if 'flush=True' not in content:
        issues.append("⚠️  missing flush=True")
    if 'finally:' not in content:
        issues.append("⚠️  missing finally: block")
    
    if issues:
        print("❌ ISSUES:", *issues)
        return False
    print("✅ CLEAN")
    return True

ok1 = check_file('inference.py')
ok2 = check_file('inference_wdif.py')
print(f"\n{'🟢 READY TO PUSH' if ok1 and ok2 else '🔴 FIX ISSUES FIRST'}")
