import logging

logger = logging.getLogger(__name__)

from .abstract_simulate import AbstractSimulator

class ConcreteSimulator(AbstractSimulator):
    def __init__(self):
        return