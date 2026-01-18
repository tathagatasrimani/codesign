"""
Optimize a Pareto surface model using CVXPY geometric programming.

This script loads a parametric Pareto surface model from JSON and optimizes
for energy-delay product (EDP).
"""

import json
import argparse
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def optimize_pareto_surface(model_json_file, objective='edp', additional_constraints=None):
    """
    Optimize a parametric Pareto surface model.

    Args:
        model_json_file: Path to JSON file containing the Pareto surface model
        objective: Optimization objective. Options:
                  - 'edp': Energy-Delay Product (delay * (Edynamic + Pstatic*delay))
                  - 'delay': Minimize delay
                  - 'energy': Minimize total energy (Edynamic + Pstatic*delay)
                  - 'area': Minimize area
        additional_constraints: Optional dict with additional constraints like
                               {'delay_max': 1e-3, 'area_max': 3e-16}

    Returns:
        Dictionary with optimal values and design point
    """
    import cvxpy as cp

    # Load model
    logger.info(f"Loading Pareto surface model from: {model_json_file}")
    with open(model_json_file, 'r') as f:
        model = json.load(f)

    # Verify it's a parametric model
    if model.get('type') != 'parametric':
        raise ValueError(f"Expected parametric model, got: {model.get('type')}")

    param_names = model['param_names']
    output_metrics = model['output_metrics']
    n_params = model['n_params']

    logger.info(f"Model has {n_params} parameters: {param_names}")
    logger.info(f"Output metrics: {output_metrics}")

    # Create CVXPY variables for abstract parameters
    params = {name: cp.Variable(pos=True, name=name) for name in param_names}

    # Create CVXPY variables for output metrics
    outputs = {name: cp.Variable(pos=True, name=name) for name in output_metrics}

    # Build constraints from the model
    constraints = []

    # Add parameter bounds (critical to prevent unbounded solutions!)
    if 'param_bounds' in model:
        for param_name, bounds in model['param_bounds'].items():
            constraints.append(params[param_name] >= bounds['min'])
            constraints.append(params[param_name] <= bounds['max'])
            logger.info(f"Parameter bounds: {bounds['min']:.3f} <= {param_name} <= {bounds['max']:.3f}")
    else:
        logger.warning("No parameter bounds found in model! Problem may be unbounded.")
        logger.warning("Regenerate the model with the latest version of sweep_tech_codesign.py")

    for constraint in model['constraints']:
        constraint_spec = model['constraints'][constraint]
        output = constraint_spec['output']
        
        # Check if this is a posynomial model (new format) or monomial model (old format)
        if constraint_spec.get('type') == 'posynomial' and 'terms' in constraint_spec:
            # Posynomial format: sum of terms
            # Build sum of monomial terms: sum(c_k * prod(param^exp_k))
            posynomial_sum = None
            for term in constraint_spec['terms']:
                c = term['coefficient']
                exponents = term.get('exponents', {})
                
                # Build monomial: c * prod(param^exp)
                if len(exponents) == 0:
                    # Constant term
                    monomial = c
                else:
                    monomial = c
                    for param, exp in exponents.items():
                        if param in params:
                            monomial = monomial * cp.power(params[param], exp)
                
                # Add to sum
                if posynomial_sum is None:
                    posynomial_sum = monomial
                else:
                    posynomial_sum = posynomial_sum + monomial
            
            constraints.append(outputs[output] >= posynomial_sum)
        elif 'coefficient' in constraint_spec and 'exponents' in constraint_spec:
            # Old monomial format: single term
            c = constraint_spec['coefficient']
            exponents = constraint_spec['exponents']

            # Build the posynomial: output >= c * prod(param^exp)
            posynomial_terms = [cp.power(params[p], exponents[p]) for p in param_names if p in exponents]

            # Create the constraint
            constraint_expr = c
            for term in posynomial_terms:
                constraint_expr = constraint_expr * term

            constraints.append(outputs[output] >= constraint_expr)
        else:
            raise ValueError(f"Unrecognized constraint format for {output}: {list(constraint_spec.keys())}")

        logger.info(f"Added constraint: {constraint_spec.get('formula', 'N/A')}")

    # Add additional user constraints
    if additional_constraints:
        for metric, max_value in additional_constraints.items():
            if metric in outputs:
                constraints.append(outputs[metric] <= max_value)
                logger.info(f"Added constraint: {metric} <= {max_value}")
    objective = 'edp'
    delay = outputs['R_avg_inv'] * outputs['C_gate']* 1e9 
    Edynamic = outputs['V_dd']**2 * outputs['C_gate']
    Pstatic = outputs['V_dd'] * outputs['Ioff']
    # Define objective function
    if objective == 'edp':
        # Energy-Delay Product: delay * (Edynamic + Pstatic * delay)
        # This is equivalent to: delay * Edynamic + delay^2 * Pstatic
        delay = outputs['R_avg_inv'] * outputs['C_gate']* 1e9 
        Edynamic = outputs['V_dd']**2 * outputs['C_gate']
        Pstatic = outputs['V_dd'] * outputs['Ioff']
        obj_expr = delay * Edynamic + cp.power(delay, 2) * Pstatic
        logger.info("Objective: Minimize Energy-Delay Product (EDP)")
    elif objective == 'delay':
        obj_expr = outputs['R_avg_inv'] * outputs['C_gate'] * 1e9 # ns
        logger.info("Objective: Minimize delay")
    elif objective == 'energy':
        obj_expr = outputs['Edynamic'] + outputs['Pstatic'] * outputs['delay']
        logger.info("Objective: Minimize total energy")
    elif objective == 'area':
        obj_expr = outputs['area']
        logger.info("Objective: Minimize area")
    else:
        raise ValueError(f"Unknown objective: {objective}")

    # Create and solve the problem
    objective_cp = cp.Minimize(obj_expr)
    problem = cp.Problem(objective_cp, constraints)

    logger.info("Solving with CVXPY DGP (Disciplined Geometric Programming)...")
    problem.solve(gp=True, verbose=True)

    # Check if solved successfully
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        logger.error(f"Optimization failed with status: {problem.status}")
        return None

    logger.info(f"Optimization status: {problem.status}")
    logger.info(f"Optimal objective value: {problem.value:.6e}")

    # Extract results
    results = {
        'status': problem.status,
        'objective_value': float(problem.value),
        'objective_type': objective,
        'parameters': {name: float(params[name].value) for name in param_names},
        'output_metrics': {name: float(outputs[name].value) for name in output_metrics}
    }



    # Print results
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*60)
    logger.info(f"\nAbstract Parameters (knobs):")
    for name, value in results['parameters'].items():
        logger.info(f"  {name} = {value:.6f}")

    logger.info(f"\nOutput Metrics:")
    for name, value in results['output_metrics'].items():
        logger.info(f"  {name} = {value:.6e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize a Pareto surface model using CVXPY"
    )
    parser.add_argument(
        "model_json",
        type=str,
        help="Path to Pareto surface model JSON file"
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="edp",
        choices=["edp", "delay", "energy", "area"],
        help="Optimization objective (default: edp)"
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        help="Maximum delay constraint"
    )
    parser.add_argument(
        "--area-max",
        type=float,
        help="Maximum area constraint"
    )

    args = parser.parse_args()

    # Build additional constraints
    additional_constraints = {}
    if args.delay_max:
        additional_constraints['delay'] = args.delay_max
    if args.area_max:
        additional_constraints['area'] = args.area_max

    # Optimize
    results = optimize_pareto_surface(
        args.model_json,
        objective=args.objective,
        additional_constraints=additional_constraints if additional_constraints else None
    )

    if results:
        logger.info("\nOptimization completed successfully!")
        logger.info("\nTo look up actual design parameters (L, W, V_dd, etc.),")
        logger.info("you need to use the fit_results from fit_pareto_surface_parametric():")
        logger.info("  design, dist = fit_results['find_nearest_design'](a0=a0_value, a1=a1_value)")
    else:
        logger.error("Optimization failed!")
