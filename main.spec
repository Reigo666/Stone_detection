# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['main.py'],
    pathex=['D:\\jpf\\mine_terminal'],
    binaries=[],
    datas=[('D:\\jpf\\mine_terminal\\config.yml', 'config'),
    ('D:\\jpf\\mine_terminal\\client\\ui.ui', 'client'),
    ('D:\\jpf\\mine_terminal\\client\\segment\\model_config\\model\\cascade_mask_rcnn_v2c_bt.pth', 'model_config')],
    hiddenimports=['mmcv','mmcv._ext', 'pyqt5'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
