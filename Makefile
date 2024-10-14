################################################################################
# Makefile
################################################################################

.PHONY: build testpypi pypi install-test check count coverage clean help

#-------------------------------------------------------------------------------
# Define package names and target source files
#-------------------------------------------------------------------------------

SOFTWARE = rfflearn
PY_FILES = $(shell find rfflearn | grep \.py$$ | sort)

help:
	@echo "Usage:"
	@echo "    make <command>"
	@echo ""
	@echo "Build commands:"
	@echo "    build         Build package"
	@echo "    testpypi      Upload package to TestPyPi"
	@echo "    pypi          Upload package to PyPi"
	@echo "    install-test  Install from TestPyPi"
	@echo ""
	@echo "Test and code check commands:"
	@echo "    check         Check the code quality"
	@echo "    count         Count the lines of code"
	@echo "    coverage      Measure code coverage"
	@echo ""
	@echo "Other commands:"
	@echo "    clean         Cleanup cache files"
	@echo "    help          Show this message"

#-------------------------------------------------------------------------------
# Build commands
#-------------------------------------------------------------------------------

build:             
	python3 -m build

testpypi:          
	twine upload --repository pypitest dist/*

pypi:
	twine upload --repository pypi dist/*

install-test:
	python3 -m pip install --index-url https://test.pypi.org/simple/ $(SOFTWARE)

#-------------------------------------------------------------------------------
# Test commands
#-------------------------------------------------------------------------------

check:
	pyflakes $(PY_FILES)
	pylint $(PY_FILES) --disable C0103,R0913,R0917,R0801

count:
	cloc --by-file $(PY_FILES)

coverage:
	rm -rf .coverage
	coverage erase
	@echo "Run tests on CPU..."
	coverage run --source rfflearn -a tests/cca_for_artificial_data.py
	coverage run --source rfflearn -a tests/feature_importances_for_california_housing.py
	coverage run --source rfflearn -a tests/gpc_for_mnist.py cpu --rtype rff
	coverage run --source rfflearn -a tests/gpc_for_mnist.py cpu --rtype orf
	coverage run --source rfflearn -a tests/gpc_for_mnist.py cpu --rtype qrf
	coverage run --source rfflearn -a tests/gpr_sparse_data.py cpu --rtype rff
	coverage run --source rfflearn -a tests/gpr_sparse_data.py cpu --rtype orf
	coverage run --source rfflearn -a tests/gpr_sparse_data.py cpu --rtype qrf
	coverage run --source rfflearn -a tests/least_square_regression.py --rtype rff
	coverage run --source rfflearn -a tests/least_square_regression.py --rtype orf
	coverage run --source rfflearn -a tests/least_square_regression.py --rtype qrf
	coverage run --source rfflearn -a tests/optuna_for_california_housing.py
	coverage run --source rfflearn -a tests/pca_for_swissroll.py cpu --rtype rff
	coverage run --source rfflearn -a tests/pca_for_swissroll.py cpu --rtype orf
	coverage run --source rfflearn -a tests/pca_for_swissroll.py cpu --rtype qrf
	coverage run --source rfflearn -a tests/svc_for_mnist.py cpu --rtype rff --skip_tune
	coverage run --source rfflearn -a tests/svc_for_mnist.py cpu --rtype orf --skip_tune
	coverage run --source rfflearn -a tests/svc_for_mnist.py cpu --rtype qrf --skip_tune
	coverage run --source rfflearn -a tests/svr_sparse_data.py cpu --rtype rff
	coverage run --source rfflearn -a tests/svr_sparse_data.py cpu --rtype orf
	coverage run --source rfflearn -a tests/svr_sparse_data.py cpu --rtype qrf
	@echo "Run tests on GPU..."
	coverage run --source rfflearn -a tests/gpc_for_mnist.py gpu --rtype rff
	coverage run --source rfflearn -a tests/gpc_for_mnist.py gpu --rtype orf
	coverage run --source rfflearn -a tests/gpc_for_mnist.py gpu --rtype qrf
	coverage run --source rfflearn -a tests/gpr_sparse_data.py gpu --rtype rff
	coverage run --source rfflearn -a tests/gpr_sparse_data.py gpu --rtype orf
	coverage run --source rfflearn -a tests/gpr_sparse_data.py gpu --rtype qrf
	coverage run --source rfflearn -a tests/pca_for_swissroll.py gpu --rtype rff
	coverage run --source rfflearn -a tests/pca_for_swissroll.py gpu --rtype orf
	coverage run --source rfflearn -a tests/pca_for_swissroll.py gpu --rtype qrf
	coverage run --source rfflearn -a tests/svc_for_mnist.py gpu --rtype rff --skip_tune
	coverage run --source rfflearn -a tests/svc_for_mnist.py gpu --rtype orf --skip_tune
	coverage run --source rfflearn -a tests/svc_for_mnist.py gpu --rtype qrf --skip_tune
	coverage run --source rfflearn -a tests/svc_train_cpu_predict_gpu.py
	coverage html

#-------------------------------------------------------------------------------
# Other commands
#-------------------------------------------------------------------------------

clean:
	rm -rf dist .coverage htmlcov scikit_learn_data `find -type d | grep __pycache__`

# vim: noexpandtab tabstop=4 shiftwidth=4
