rm -rf build dist BiWAKO.egg-info
python setup.py bdist_wheel
twine upload dist/*
