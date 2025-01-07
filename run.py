import pybinsim
import logging

pybinsim.logger.setLevel(logging.INFO)    # defaults to INFO
# Use logging.WARNING for printing warnings only
with pybinsim.BinSim('settings.cfg') as binsim:
    binsim.run_server()
