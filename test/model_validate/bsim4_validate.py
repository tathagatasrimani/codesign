import argparse
from src import codesign
import copy
import sympy as sp
import logging
import datetime
import os

logger = logging.getLogger(__name__)

def validate_bsim4_model(args):
    if args.tech_node is not None:
        tech_nodes = [args.tech_node]
    else:
        tech_nodes = [
            "default",
            #"tech-1982",
            "tech-1984",
            #"tech-1986",
            "tech-1990",
            #"tech-1992",
            #"tech-1994",
            "tech-1996",
            #"tech-1998",
            #"tech-2000",
            "tech-2002",
            #"tech-2004",
            "tech-2006",
            #"tech-2008",
            #"tech-2010",
        ]
    log_save_dir = os.path.join(os.path.dirname(__file__), "logs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_save_dir, "bsim4_validate.log"))
    logger.info(args.tech_node)
    I_d_ref = {}
    I_d_bsim4 = {}
    u_n_ref = {}
    u_n_bsim4 = {}
    delay_ref = {}
    delay_bsim4 = {}
    V_th_eff_bsim4 = {}
    #Vgsteff_bsim4 = {}
    Gate_tunneling_ref = {}
    Gate_tunneling_bsim4 = {}
    I_GIDL_ref = {}
    I_GIDL_bsim4 = {}
    delta_vt_dibl_bsim4 = {}
    delta_vt_sce_bsim4 = {}
    delta_vt_nw_bsim4 = {}
    R_wire_bsim4 = {}
    C_wire_bsim4 = {}
    I_sub_bsim4 = {}
    for tech_node in tech_nodes:
        args.tech_node = tech_node
        codesign_model = codesign.Codesign(args)
        I_d_bsim4[tech_node] = (codesign_model.hw.circuit_model.tech_model.I_d_n/(codesign_model.hw.circuit_model.tech_model.base_params.W*1e6)).xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        u_n_bsim4[tech_node] = codesign_model.hw.circuit_model.tech_model.u_n_eff.xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        delay_bsim4[tech_node] = codesign_model.hw.circuit_model.tech_model.delay.xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        V_th_eff_bsim4[tech_node] = codesign_model.hw.circuit_model.tech_model.V_th_eff.xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        #Vgsteff_bsim4[tech_node] = codesign_model.hw.circuit_model.tech_model.Vgsteff.xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        Gate_tunneling_bsim4[tech_node] = (codesign_model.hw.circuit_model.tech_model.I_tunnel/(codesign_model.hw.circuit_model.tech_model.base_params.W*1e6)).xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        I_GIDL_bsim4[tech_node] = (codesign_model.hw.circuit_model.tech_model.I_GIDL/(codesign_model.hw.circuit_model.tech_model.base_params.W*1e6)).xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        delta_vt_dibl_bsim4[tech_node] = codesign_model.hw.circuit_model.tech_model.dVth_DIBL.xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        delta_vt_sce_bsim4[tech_node] = codesign_model.hw.circuit_model.tech_model.dVth_SCE.xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        delta_vt_nw_bsim4[tech_node] = codesign_model.hw.circuit_model.tech_model.dVth_nw_1.xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        R_wire_bsim4[tech_node] = codesign_model.hw.circuit_model.tech_model.R_wire.xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        C_wire_bsim4[tech_node] = codesign_model.hw.circuit_model.tech_model.C_wire.xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        I_sub_bsim4[tech_node] = codesign_model.hw.circuit_model.tech_model.I_sub_n.xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        new_args = copy.deepcopy(args)
        new_args.model_cfg = "default"
        codesign_ref_model = codesign.Codesign(new_args)
        I_d_ref[tech_node] = (codesign_ref_model.hw.circuit_model.tech_model.I_d_nmos/(codesign_ref_model.hw.circuit_model.tech_model.base_params.W*1e6)).xreplace(codesign_ref_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        if type(codesign_ref_model.hw.circuit_model.tech_model.u_n_eff) == sp.Symbol:
            u_n_ref[tech_node] = codesign_ref_model.hw.circuit_model.tech_model.u_n_eff.xreplace(codesign_ref_model.hw.circuit_model.tech_model.base_params.tech_values)
        else:
            u_n_ref[tech_node] = codesign_ref_model.hw.circuit_model.tech_model.u_n_eff
        delay_ref[tech_node] = codesign_ref_model.hw.circuit_model.tech_model.delay.xreplace(codesign_ref_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        Gate_tunneling_ref[tech_node] = (codesign_ref_model.hw.circuit_model.tech_model.I_tunnel/(codesign_ref_model.hw.circuit_model.tech_model.base_params.W*1e6)).xreplace(codesign_ref_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        I_GIDL_ref[tech_node] = (codesign_ref_model.hw.circuit_model.tech_model.I_GIDL/(codesign_ref_model.hw.circuit_model.tech_model.base_params.W*1e6)).xreplace(codesign_ref_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
        logger.info("")
        logger.info(f"I_d_ref[{tech_node}] per um: {I_d_ref[tech_node]}")
        logger.info(f"I_d_bsim4[{tech_node}] per um: {I_d_bsim4[tech_node]}")
        logger.info(f"u_n_ref[{tech_node}]: {u_n_ref[tech_node]}")
        logger.info(f"u_n_bsim4[{tech_node}]: {u_n_bsim4[tech_node]}")
        logger.info(f"delay_ref[{tech_node}]: {delay_ref[tech_node]}")
        logger.info(f"delay_bsim4[{tech_node}]: {delay_bsim4[tech_node]}")
        logger.info(f"V_th_eff_bsim4[{tech_node}]: {V_th_eff_bsim4[tech_node]}")
        #logger.info(f"Vgsteff_bsim4[{tech_node}]: {Vgsteff_bsim4[tech_node]}")
        logger.info(f"Gate_tunneling_ref[{tech_node}]: {Gate_tunneling_ref[tech_node]}")
        logger.info(f"Gate_tunneling_bsim4[{tech_node}] per um: {Gate_tunneling_bsim4[tech_node]}")
        logger.info(f"delta_vt_dibl_bsim4[{tech_node}]: {delta_vt_dibl_bsim4[tech_node]}")
        logger.info(f"delta_vt_sce_bsim4[{tech_node}]: {delta_vt_sce_bsim4[tech_node]}")
        logger.info(f"delta_vt_nw_bsim4[{tech_node}]: {delta_vt_nw_bsim4[tech_node]}")
        logger.info(f"I_GIDL_ref[{tech_node}] per um: {I_GIDL_ref[tech_node]}")
        logger.info(f"I_GIDL_bsim4[{tech_node}] per um: {I_GIDL_bsim4[tech_node]}")
        logger.info(f"R_wire_bsim4[{tech_node}]: {R_wire_bsim4[tech_node]}")
        logger.info(f"C_wire_bsim4[{tech_node}]: {C_wire_bsim4[tech_node]}")
        logger.info(f"I_sub_bsim4[{tech_node}]: {I_sub_bsim4[tech_node]}")
        logger.info("")
    logger.info(f"I_d_ref per um: {I_d_ref}")
    logger.info(f"I_d_bsim4 per um: {I_d_bsim4}")
    logger.info(f"u_n_ref: {u_n_ref}")
    logger.info(f"u_n_bsim4: {u_n_bsim4}")
    logger.info(f"delay_ref: {delay_ref}")
    logger.info(f"delay_bsim4: {delay_bsim4}")
    logger.info(f"V_th_eff_bsim4: {V_th_eff_bsim4}")
    #logger.info(f"Vgsteff_bsim4: {Vgsteff_bsim4}")
    logger.info(f"Gate_tunneling_ref: {Gate_tunneling_ref}")
    logger.info(f"Gate_tunneling_bsim4 per um: {Gate_tunneling_bsim4}")
    logger.info(f"delta_vt_dibl_bsim4: {delta_vt_dibl_bsim4}")
    logger.info(f"delta_vt_sce_bsim4: {delta_vt_sce_bsim4}")
    logger.info(f"delta_vt_nw_bsim4: {delta_vt_nw_bsim4}")
    logger.info(f"I_GIDL_ref per um: {I_GIDL_ref}")
    logger.info(f"I_GIDL_bsim4 per um: {I_GIDL_bsim4}")
    logger.info(f"R_wire_bsim4: {R_wire_bsim4}")
    logger.info(f"C_wire_bsim4: {C_wire_bsim4}")
    logger.info(f"I_sub_bsim4: {I_sub_bsim4}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Codesign",
        description="Runs a two-step loop to optimize architecture and technology for a given application.",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        help="Name of benchmark to run"
    )
    parser.add_argument(
        "-f",
        "--savedir",
        type=str,
        default="logs",
        help="Path to the save new architecture file",
    )
    parser.add_argument(
        "--parasitics",
        type=str,
        choices=["detailed", "estimation", "none"],
        default="estimation",
        help="determines what type of parasitic calculations are done for wires",
    )

    parser.add_argument(
        "--openroad_testfile",
        type=str,
        default="openroad_interface/tcl/codesign_top.tcl",
        help="what tcl file will be executed for openroad",
    )
    parser.add_argument(
        "-N",
        "--num_iters",
        type=int,
        default=10,
        help="Number of Codesign iterations to run",
    )
    parser.add_argument(
        "-a",
        "--area",
        type=float,
        default=1000000,
        help="Area constraint in um2",
    )
    parser.add_argument(
        "--no_memory",
        type=bool,
        default=False,
        help="disable memory modeling",
    )
    parser.add_argument('--debug_no_cacti', type=bool, default=False, 
                        help='disable cacti in the first iteration to decrease runtime when debugging')
    parser.add_argument("-c", "--checkpoint", type=bool, default=False, help="save a design checkpoint upon exit")
    parser.add_argument("--logic_node", type=int, default=7, help="logic node size")
    parser.add_argument("--mem_node", type=int, default=32, help="memory node size")
    parser.add_argument("--inverse_pass_improvement", type=float, help="improvement factor for inverse pass")
    parser.add_argument("--tech_node", "-T", type=str, help="technology node to use as starting point")
    parser.add_argument("--obj", type=str, default="edp", help="objective function")
    parser.add_argument("--model_cfg", type=str, default="bsim4_limited", help="symbolic model configuration")

    args = parser.parse_args()
    validate_bsim4_model(args)