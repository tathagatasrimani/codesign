import sympy as sp
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src import preprocess
import logging
logger = logging.getLogger(__name__)

def debug_buf_access_time_pow_exprs():
    """
    Reads Buf_access_time.txt as a sympy expression and calls find_pow_exprs_to_constrain.
    """
    expr_file = os.path.join(os.path.dirname(__file__), '../src/cacti/symbolic_expressions/Buf_access_time.txt')
    with open(expr_file, 'r') as f:
        expr_str = f.read().strip()
    expr = sp.sympify(expr_str)
    preprocessor = preprocess.Preprocessor()
    preprocessor.find_pow_exprs_to_constrain(expr, debug=False)
    logger.info(f"Power expressions to constrain: {preprocessor.pow_exprs_s}")


if __name__ == '__main__':
    if os.path.exists('debug/debug_preprocessor.log'):
        os.remove('debug/debug_preprocessor.log')
    logging.basicConfig(filename='debug/debug_preprocessor.log', level=logging.INFO)
    debug_buf_access_time_pow_exprs()
