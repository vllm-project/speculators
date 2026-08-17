CHECKDIRS := src setup.py

# clean package
clean:
	rm -fr dist src/*.egg-info;
	find $(CHECKDIRS) | grep -E "(__pycache__|\.pyc|\.pyo|version\.py|version\.txt)" | xargs rm -fr;
