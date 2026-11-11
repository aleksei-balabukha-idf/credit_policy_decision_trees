# -*- mode: python ; coding: utf-8 -*-
# Build: pyinstaller app.spec
a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=['pandas', 'numpy'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='DecisionTree',
    debug=False,
    strip=False,
    upx=True,
    console=False,          # windowed app
)
