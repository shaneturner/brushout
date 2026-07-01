# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('models/*.onnx', 'models'),
    ],
    hiddenimports=[
        'onnxruntime.capi._pybind_state',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Unused PySide6 submodules — saves ~400 MB from the bundle
        'PySide6.Qt3DAnimation', 'PySide6.Qt3DCore', 'PySide6.Qt3DExtras',
        'PySide6.Qt3DInput', 'PySide6.Qt3DLogic', 'PySide6.Qt3DRender',
        'PySide6.QtBluetooth', 'PySide6.QtCharts', 'PySide6.QtDataVisualization',
        'PySide6.QtDesigner', 'PySide6.QtHelp', 'PySide6.QtLocation',
        'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets', 'PySide6.QtNfc',
        'PySide6.QtOpenGL', 'PySide6.QtOpenGLWidgets', 'PySide6.QtPositioning',
        'PySide6.QtPrintSupport', 'PySide6.QtQml', 'PySide6.QtQuick',
        'PySide6.QtQuick3D', 'PySide6.QtQuickControls2', 'PySide6.QtQuickWidgets',
        'PySide6.QtRemoteObjects', 'PySide6.QtSensors', 'PySide6.QtSerialBus',
        'PySide6.QtSerialPort', 'PySide6.QtSql', 'PySide6.QtSvg',
        'PySide6.QtSvgWidgets', 'PySide6.QtTest', 'PySide6.QtUiTools',
        'PySide6.QtWebChannel', 'PySide6.QtWebEngine', 'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets', 'PySide6.QtWebSockets', 'PySide6.QtXml',
        # Not needed at runtime: models are pre-bundled so onnxsim/onnx are never called
        'onnx', 'onnxsim',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Brushout',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='brushout.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['*.onnx'],
    name='Brushout',
)
