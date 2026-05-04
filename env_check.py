import sys
print('PYTHON:', sys.executable)

packages = ['numpy', 'scipy', 'sklearn', 'cv2', 'tensorflow', 'tensorflow_intel']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'no-version')
        print(pkg, version, getattr(module, '__file__', 'no-file'))
    except Exception as exc:
        print(pkg, 'ERROR', repr(exc))
