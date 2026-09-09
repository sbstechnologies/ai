import os
import sys

# Base directory setup to ensure sub-packages and assets load seamlessly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_AI_DIR = os.path.join(BASE_DIR, "face_ai")

if FACE_AI_DIR not in sys.path:
    sys.path.insert(0, FACE_AI_DIR)

# Delegate execution to face_ai/app.py
target_app = os.path.join(FACE_AI_DIR, "app.py")
with open(target_app, "r", encoding="utf-8") as f:
    code = compile(f.read(), target_app, "exec")
    exec(code, globals())
