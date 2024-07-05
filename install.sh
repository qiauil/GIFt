pip uninstall GIFt
rm -rf build dist foxutils.egg-info
python3 setup.py sdist bdist_wheel
cd dist
pip install GIFt-*.whl
