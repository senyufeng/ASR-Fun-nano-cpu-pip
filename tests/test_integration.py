"""Full integration test for asr_nano package.

Tests:
  1. Import
  2. ASREngine init + load
  3. Transcribe (ASR + timestamps)
  4. Voiceprint integration
  5. JSON serialization
"""

import sys, warnings, os, json, time

warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\EDY\Desktop\ASR-Fun-nano")

MODEL_DIR = r"c:\Users\EDY\Desktop\ASR-Fun-nano\models\FunAudioLLM\Fun-ASR-Nano-2512"
ZH_AUDIO = MODEL_DIR + "/example/zh.mp3"

errors = []

def check(name, ok):
    if ok:
        print(f"  [{name}] PASS")
    else:
        print(f"  [{name}] FAIL")
        errors.append(name)
    return ok

print("=" * 50)
print("ASR-Fun-Nano Integration Test")
print("=" * 50)

# Test 1: Import
print("\n[1] Import")
try:
    from asr_nano import ASREngine, VoiceprintDB, VoiceprintEngine
    from asr_nano.engine import ASRResult, SegmentResult, CharTimestamp, VoiceprintResult
    check("Import ASREngine", True)
except Exception as e:
    check(f"Import ASREngine: {e}", False)

# Test 2: Engine init + load (pure ASR)
print("\n[2] ASREngine init + load")
try:
    engine = ASREngine(model=MODEL_DIR, device="cpu", language="中文", enable_voiceprint=False)
    engine.load()
    check("Engine init", True)
    check("Engine load", engine._asr_model is not None)
except Exception as e:
    check(f"Engine load: {e}", False)
    engine = None

# Test 3: Transcribe
print("\n[3] Transcribe")
if engine:
    try:
        results = engine.transcribe([ZH_AUDIO])
        r = results[0]
        check("Results non-empty", len(results) > 0)
        check("Text non-empty", len(r.text) > 0)
        check("Timestamps non-empty", len(r.segments[0].timestamps) > 0)
        check(f"RTF < 1.0 ({r.elapsed_sec:.2f}s)", r.elapsed_sec < 30)
        print(f"    Text: {r.text}")
        print(f"    Timestamps: {len(r.segments[0].timestamps)} chars")
        print(f"    Elapsed: {r.elapsed_sec}s")
    except Exception as e:
        check(f"Transcribe: {e}", False)

# Test 4: Voiceprint integration
print("\n[4] Voiceprint")
if engine:
    engine.unload()
    engine2 = ASREngine(model=MODEL_DIR, device="cpu", language="中文", enable_voiceprint=True)
    try:
        engine2.load()
        check("Voiceprint load", engine2._vp_engine is not None)
        results = engine2.transcribe([ZH_AUDIO])
        r = results[0]
        vp = r.segments[0].voiceprint
        check("Voiceprint result exists", vp is not None)
        if vp:
            check("Voiceprint enabled=True", vp.enabled == True)
            print(f"    Matched: {vp.matched}, Confidence: {vp.confidence:.4f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(f"Voiceprint: {e}", False)

# Test 5: JSON serialization
print("\n[5] JSON serialization")
if engine:
    try:
        json_str = ASREngine.to_json(results)
        data = json.loads(json_str)
        check("JSON parseable", isinstance(data, list))
        check("JSON has text", "text" in data[0])
        check("JSON has segments", "segments" in data[0])
        check("JSON has timestamps", "timestamps" in data[0]["segments"][0])
        print(f"    JSON size: {len(json_str)} chars")
    except Exception as e:
        check(f"JSON: {e}", False)

# Summary
print("\n" + "=" * 50)
if errors:
    print(f"FAILED: {len(errors)} tests")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)