import numpy as np
from gep_config_parser import parse_config
import json
import os
import time
import pickle
import optuna
import argparse

from primal_dual import PrimalDualTrainer
from create_gep_dataset import create_gep_ed_dataset
from devices import KNOWN_DEVICES, resolve_device_name
from paths import (
    add_path_args,
    ensure_dir,
    resolve_roots,
    under_data_root,
    under_repo,
    under_root,
)


CONFIG_FILE_NAME = "configs/config.toml"

'''
ARGS_FILE_NAME is the DEFAULT config; override it with -c/--config, e.g.
    python main.py -c configs/config-5node.json
Options:
- "configs/config.json": Default config for experiments. (3 Node)
- "configs/config-4node.json": Config for 4-node experiments.
- "configs/config-5node.json": Config for 5-node experiments.
- "configs/config-6node.json": Config for 6-node experiments.
'''
ARGS_FILE_NAME = "configs/config.json"


def parse_cli_args():
    parser = argparse.ArgumentParser()

    #! --home-path / --data-root / --output-root
    add_path_args(parser)

    parser.add_argument(
        "-c", "--config",
        dest="config",
        default=ARGS_FILE_NAME,
        help=f"Path to the run config JSON. Relative paths resolve against the "
             f"repository, not the working directory. Default: {ARGS_FILE_NAME}",
    )
    parser.add_argument(
        "--toml-config", "--toml_config",
        dest="toml_config",
        default=CONFIG_FILE_NAME,
        help=f"Path to the input-data TOML config. Default: {CONFIG_FILE_NAME}",
    )

    parser.add_argument(
        "--device",
        dest="device",
        choices=list(KNOWN_DEVICES),
        default=None,
        help="Override the config's device. 'auto' picks CUDA when present and "
             "CPU otherwise; MPS is never auto-selected because it forces "
             "float32. Unavailable devices raise instead of falling back.",
    )

    #! W&B overrides, so a job script does not have to edit the JSON configs.
    #! All default to None, which leaves the config file's values in force.
    parser.add_argument(
        "--wandb-mode", "--wandb_mode",
        dest="wandb_mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="Use 'offline' on compute nodes without outbound network, then "
             "'wandb sync <output-root>/wandb/offline-run-*' from a login node.",
    )
    parser.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default=None)
    parser.add_argument("--wandb-entity", "--wandb_entity", dest="wandb_entity", default=None)
    parser.add_argument(
        "--wandb-group", "--wandb_group",
        dest="wandb_group",
        default=None,
        help="Cluster related runs on one chart, e.g. the SLURM job id.",
    )
    parser.add_argument(
        "--no-wandb",
        dest="no_wandb",
        action="store_true",
        help="Disable W&B logging regardless of the config file.",
    )

    return parser.parse_args()


def apply_cli_overrides(args, cli_args):
    """CLI beats the config file; an unset flag leaves the config value alone."""
    if cli_args.no_wandb:
        args["use_wandb"] = False
    for key in ("device", "wandb_mode", "wandb_project", "wandb_entity", "wandb_group"):
        value = getattr(cli_args, key, None)
        if value:
            args[key] = value

    #! Resolve "auto" once, here, and write the concrete name back. Every
    #! downstream resolve_device() reads this same dict, so one resolution
    #! propagates; and args.json then records the device actually used rather
    #! than the word "auto", which is what Benders later reads back.
    args["device"] = resolve_device_name(args.get("device"))


def get_data_root(args, cli_data_root=None):
    """Deprecated: kept so older callers keep working. Use paths.resolve_roots."""
    return resolve_roots(args, argparse.Namespace(data_root=cli_data_root))["data_root"]


def build_ed_data_save_path(ED_args, nodes_count, nodes_str, gens_str, lines_str):
    """
    Build save path for ED dataset file.

    Cases:
    1. generate_capacity_sobol=True and gen_data_constraint=True
       -> use CapSobol_GenConst in filename
    2. gen_data_constraint=True
       -> use existing GenConst naming
    3. otherwise
       -> use default ED naming
    """

    if ED_args.get("generate_capacity_sobol", False) and ED_args.get("gen_data_constraint", False):

        node_constraint = ED_args.get("gen_data_node_constraint", False)
        node_tag = "_NodeConst" if node_constraint else ""
        label_tag = "_Label" if ED_args.get("precompute_heuristic_lambda_labels", False) else ""
        gen_data_renew_availability = ED_args.get("gen_data_renew_availability", 100)

        print(f"Cap Use Hard Cap setting is {ED_args.get('capacity_sobol_use_hard_cap', False)}")

        if ED_args.get("capacity_sobol_use_hard_cap", False):
            cap_sobel_tag = "_CapSobol_GenConst"
        else:
            cap_sobel_tag = "_CapSobolNoHardCap_GenConst"

        if gen_data_renew_availability == 100:
            data_save_path = (
                f"data/ED_data/Constraint/{nodes_count}Loc/"
                f"ED_N{nodes_str}_G{gens_str}_{lines_str}"
                f"_c{int(ED_args['benders_compact'])}"
                f"_s{int(ED_args['scale_problem'])}"
                f"_p{int(ED_args['perturb_operating_costs'])}"
                f"{cap_sobel_tag}"
                f"_ui_constraint{ED_args['max_investment']}"
                f"_smp{ED_args['2n_synthetic_samples']}"
                f"_renewMaxInv{ED_args.get('gen_data_constraint_ub_renewable_max_inv', True)}"
                f"{node_tag}{label_tag}.pkl"
            )
        else:
            data_save_path = (
                f"data/ED_data/Constraint/{nodes_count}Loc/"
                f"ED_N{nodes_str}_G{gens_str}_{lines_str}"
                f"_c{int(ED_args['benders_compact'])}"
                f"_s{int(ED_args['scale_problem'])}"
                f"_p{int(ED_args['perturb_operating_costs'])}"
                f"{cap_sobel_tag}"
                f"_ui_constraint{ED_args['max_investment']}"
                f"_smp{ED_args['2n_synthetic_samples']}"
                f"_renewPerc{gen_data_renew_availability}"
                f"{node_tag}{label_tag}.pkl"
            )

    elif ED_args.get("gen_data_constraint", False):
        lb_setting_name = ED_args.get("gen_data_constraint_lb", False)
        renewable_ub_max_inv = ED_args.get("gen_data_constraint_ub_renewable_max_inv", True)
        gen_data_renew_availability = ED_args.get("gen_data_renew_availability", 100)
        node_constraint = ED_args.get("gen_data_node_constraint", False)
        node_tag = "_NodeConst" if node_constraint else ""
        label_tag = "_Label" if ED_args.get("precompute_heuristic_lambda_labels", False) else ""

        if gen_data_renew_availability == 100:
            data_save_path = (
                f"data/ED_data/Constraint/{nodes_count}Loc/"
                f"ED_N{nodes_str}_G{gens_str}_{lines_str}"
                f"_c{int(ED_args['benders_compact'])}"
                f"_s{int(ED_args['scale_problem'])}"
                f"_p{int(ED_args['perturb_operating_costs'])}"
                f"_ui_constraint{ED_args['max_investment']}"
                f"_smp{ED_args['2n_synthetic_samples']}"
                f"_GenConst_lb{lb_setting_name}"
                f"_renewMaxInv{renewable_ub_max_inv}"
                f"{node_tag}{label_tag}.pkl"
            )
        else:
            data_save_path = (
                f"data/ED_data/Constraint/{nodes_count}Loc/"
                f"ED_N{nodes_str}_G{gens_str}_{lines_str}"
                f"_c{int(ED_args['benders_compact'])}"
                f"_s{int(ED_args['scale_problem'])}"
                f"_p{int(ED_args['perturb_operating_costs'])}"
                f"_ui_constraint{ED_args['max_investment']}"
                f"_smp{ED_args['2n_synthetic_samples']}"
                f"_GenConst_lb{lb_setting_name}"
                f"_renewPerc{gen_data_renew_availability}"
                f"{node_tag}{label_tag}.pkl"
            )

    else:
        if ED_args.get("precompute_heuristic_lambda_labels", False):
            label_tag = "_Label"
        else:
            label_tag = ""

        data_save_path = (
            f"data/ED_data/"
            f"ED_N{nodes_str}_G{gens_str}_{lines_str}"
            f"_c{int(ED_args['benders_compact'])}"
            f"_s{int(ED_args['scale_problem'])}"
            f"_p{int(ED_args['perturb_operating_costs'])}"
            f"_smp{ED_args['2n_synthetic_samples']}{label_tag}.pkl"
        )

    return data_save_path


if __name__ == "__main__":
    cli_args = parse_cli_args()

    # Load the arguments.
    #! Repo-anchored, so the config is found whatever the working directory is
    #! and its resolution never depends on the roots resolved just below.
    run_config_file = under_repo(cli_args.config)
    with open(run_config_file, "r") as file:
        args = json.load(file)

    roots = resolve_roots(args, cli_args)
    args.update(roots)
    data_root = roots["data_root"]
    output_root = roots["output_root"]

    apply_cli_overrides(args, cli_args)

    print(f"Run config:  {run_config_file}")
    print(f"Dataset root: {data_root}")
    print(f"Output root:  {output_root}")
    print(f"Device:       {args['device']}")

    QP_args = args["QP_args"]

    assert args["problem_type"] in ["ED", "GEP", "QP"], (
        "Problem type must be either 'ED', 'GEP', or 'QP'"
    )

    run_name = (
        f"learn_primal:{args['learn_primal']}"
        f"_train:{args['train']}"
        f"_rho:{args['rho']}"
        f"_rhomax:{args['rho_max']}"
        f"_alpha:{args['alpha']}"
        f"_L:{args['alpha']}"
    )

    #! Rooting save_dir roots every output: args.json, the repeat and Optuna
    #! subdirectories, the checkpoints and CSVs written by PrimalDualTrainer,
    #! and the TensorBoard event files, all of which join onto it.
    save_dir = under_root(
        os.path.join(
            "outputs",
            "PDL",
            args["problem_type"],
            run_name + "-" + str(time.time()).replace(".", "-"),
        ),
        output_root,
    )

    ensure_dir(save_dir)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(args, f, indent=4)

    if args["problem_type"] == "QP":
        from create_QP_dataset import (
            create_QP_dataset,
            create_nonconvex_QP_dataset,
            create_varying_G_dataset,
            create_varying_Q_dataset,
        )
        if QP_args["random_hyperparams"]:
            tau = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9], size=QP_args["repeats"])
            rho = np.random.choice([0.1, 0.5, 1, 10], size=QP_args["repeats"])
            rho_max = np.random.choice([1000, 5000, 10000, 50000], size=QP_args["repeats"])
            alpha = np.random.choice([1, 1.5, 2, 5, 10], size=QP_args["repeats"])
        else:
            rho = args["rho"]
            rho_max = args["rho_max"]
            alpha = args["alpha"]

        for QP_type in QP_args["type"]:
            relative_data_save_path = (
                f"data/QP_data/QP_type:{QP_type}"
                f"_var:{QP_args['var']}"
                f"_ineq:{QP_args['ineq']}"
                f"_eq:{QP_args['eq']}"
                f"_num_samples:{QP_args['num_samples']}.pkl"
            )

            data_save_path = under_data_root(relative_data_save_path, data_root)
            curr_type_save_dir = os.path.join(save_dir, QP_type)

            # Create dataset if it doesn't exist
            if not os.path.exists(data_save_path):
                directory = os.path.dirname(data_save_path)
                os.makedirs(directory, exist_ok=True)

                if QP_type == "simple":
                    create_QP_dataset(
                        QP_args["var"],
                        QP_args["ineq"],
                        QP_args["eq"],
                        QP_args["num_samples"],
                        data_save_path,
                    )
                elif QP_type == "row":
                    create_varying_G_dataset(
                        QP_args["var"],
                        QP_args["ineq"],
                        QP_args["eq"],
                        QP_args["num_samples"],
                        "row",
                        data_save_path,
                    )
                elif QP_type == "column":
                    create_varying_G_dataset(
                        QP_args["var"],
                        QP_args["ineq"],
                        QP_args["eq"],
                        QP_args["num_samples"],
                        "column",
                        data_save_path,
                    )
                elif QP_type == "random":
                    create_varying_G_dataset(
                        QP_args["var"],
                        QP_args["ineq"],
                        QP_args["eq"],
                        QP_args["num_samples"],
                        "random",
                        data_save_path,
                    )
                elif QP_type == "obj":
                    create_varying_Q_dataset(
                        QP_args["var"],
                        QP_args["ineq"],
                        QP_args["eq"],
                        QP_args["num_samples"],
                        data_save_path,
                    )
                elif QP_type == "nonconvex":
                    create_nonconvex_QP_dataset(
                        QP_args["var"],
                        QP_args["ineq"],
                        QP_args["eq"],
                        QP_args["num_samples"],
                        data_save_path,
                    )
                else:
                    raise ValueError(f"QP type {QP_type} not supported")

            # Load data
            with open(data_save_path, "rb") as file:
                data = pickle.load(file)

            for repeat in range(QP_args["repeats"]):
                if QP_args["random_hyperparams"]:
                    curr_repeat_save_dir = os.path.join(
                        curr_type_save_dir,
                        (
                            f"tau_{tau[repeat]}"
                            f"_rho_{rho[repeat]}"
                            f"_rhomax_{rho_max[repeat]}"
                            f"_alpha_{alpha[repeat]}"
                            f"_repeat:{repeat}"
                        ),
                    )
                    args["tau"] = tau[repeat]
                    args["rho"] = rho[repeat]
                    args["rho_max"] = rho_max[repeat]
                    args["alpha"] = alpha[repeat]
                else:
                    curr_repeat_save_dir = os.path.join(curr_type_save_dir, f"repeat:{repeat}")

                if not os.path.exists(curr_repeat_save_dir):
                    os.makedirs(curr_repeat_save_dir)

                args["repeat"] = repeat
                if not args.get("wandb_group"):
                    args["wandb_group"] = os.path.basename(save_dir)

                trainer = PrimalDualTrainer(data, args, curr_repeat_save_dir)
                primal_net, dual_net = trainer.train_PDL()

    else:
        ED_args = args["ED_args"]

        # Reads the input data using config.toml's experiment.inputs.data path.
        input_data = parse_config(under_repo(cli_args.toml_config))

        # Take first experiment, we don't change the inputs here.
        gep_ed_data = input_data["experiment"]["experiments"][0]

        print("--------_GEP ED Dataset Info--------")
        print(gep_ed_data)

        if ED_args["use_direct_data"]:
            data_save_path = under_data_root(ED_args["direct_data_path"], data_root)
            print(f"[direct] loading ED data from: {data_save_path}")
            assert os.path.exists(data_save_path), f"direct_data_path not found: {data_save_path}"
            with open(data_save_path, "rb") as file:
                data = pickle.load(file)
        else:
            if args["problem_type"] == "ED":
                # TODO:
                # Not all configs are correctly parsed here.
                # E.g. when first running BEL and GER with both coal generators,
                # is the same as with both gas generators.

                # For nodes, just use first letters:
                # ['BEL', 'GER', 'NED'] → 'B-G-N'
                nodes_str = "-".join([n[0] for n in ED_args["N"]])
                nodes_count = len(ED_args["N"])

                # For generators, count per node:
                # [['BEL', 'WindOn'], ['BEL', 'Gas'], ...] = 'B3-G2-N2'
                gen_counts = {}
                for g in ED_args["G"]:
                    node = g[0]
                    gen_counts[node] = gen_counts.get(node, 0) + 1

                gens_str = "-".join([f"{node[0]}{count}" for node, count in gen_counts.items()])

                # For lines, just count:
                # [['BEL', 'GER'], ['BEL', 'NED'], ['GER', 'NED']] → 'L3'
                lines_str = f"L{len(ED_args['L'])}"

                data_save_path = build_ed_data_save_path(
                    ED_args=ED_args,
                    nodes_count=nodes_count,
                    nodes_str=nodes_str,
                    gens_str=gens_str,
                    lines_str=lines_str,
                )

            elif args["problem_type"] == "GEP":
                data_save_path = (
                    f"data/GEP_data/"
                    f"N:{ED_args['N']}"
                    f"_G:{ED_args['G']}"
                    f"_L:{ED_args['L']}"
                    f"_scale-prob:{ED_args['scale_problem']}.pkl"
                )

            data_save_path = under_data_root(data_save_path, data_root)

            print(f"Data save path: {data_save_path} and does it exist? {os.path.exists(data_save_path)}")
            print("---------------------")

            if not os.path.exists(data_save_path):
                directory = os.path.dirname(data_save_path)
                os.makedirs(directory, exist_ok=True)

                data = create_gep_ed_dataset(
                    args=args,
                    problem_args=ED_args,
                    inputs=gep_ed_data,
                    problem_type=args["problem_type"],
                    save_path=data_save_path,
                )

            # Load data
            with open(data_save_path, "rb") as file:
                data = pickle.load(file)

        if args["Optuna_args"]["optuna"]:
            # Tune the hyperparameters using Optuna
            optuna_args = args["Optuna_args"]

            # Don't log to tensorboard for Optuna trials, it will be too slow.
            args["log"] = False

            def objective(trial):
                # Suggest hyperparameters with Optuna
                if args["learn_primal"]:
                    args["primal_lr"] = trial.suggest_float(
                        "primal_lr",
                        *optuna_args["primal_lr"],
                    )

                if args["learn_dual"]:
                    args["dual_lr"] = trial.suggest_float(
                        "dual_lr",
                        *optuna_args["dual_lr"],
                    )

                args["hidden_size_factor"] = trial.suggest_int(
                    "hidden_size_factor",
                    *optuna_args["hidden_size_factor"],
                )
                args["n_layers"] = trial.suggest_int(
                    "n_layers",
                    *optuna_args["n_layers"],
                )
                args["decay"] = trial.suggest_float(
                    "decay",
                    *optuna_args["decay"],
                )
                args["batch_size"] = trial.suggest_categorical(
                    "batch_size",
                    optuna_args["batch_size"],
                )

                trial_save_dir = os.path.join(save_dir, f"optuna_trial:{trial.number}")
                os.makedirs(trial_save_dir, exist_ok=True)

                trainer = PrimalDualTrainer(data, args, trial_save_dir)
                primal_net, dual_net, primal_loss, dual_loss = trainer.train_PDL(trial)

                if args["learn_primal"]:
                    return primal_loss
                elif args["learn_dual"]:
                    return dual_loss
                else:
                    raise ValueError("Must learn either primal or dual")

            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2000,
                interval_steps=10,
            )

            study = optuna.create_study(direction="minimize", pruner=pruner)
            study.optimize(objective, n_trials=optuna_args["optuna_trials"])

            df = study.trials_dataframe()
            df.to_csv(os.path.join(save_dir, "optuna_trials.csv"), index=False)

            print("Best trial:", study.best_trial.params)

        else:
            # Use best-found hyperparameters using Optuna
            if args["learn_primal"]:
                best_args = {
                    "primal_lr": 0.0006785456069117277,
                    "hidden_size_factor": 28,
                    "n_layers": 2,
                    "decay": 0.9989743016070536,
                    "batch_size": 2048,
                }

                args["primal_lr"] = best_args["primal_lr"]
                args["hidden_size_factor"] = best_args["hidden_size_factor"]
                args["n_layers"] = best_args["n_layers"]
                args["decay"] = best_args["decay"]
                args["batch_size"] = best_args["batch_size"]

            for repeat in range(ED_args["repeats"]):
                curr_repeat_save_dir = os.path.join(save_dir, f"repeat:{repeat}")
                os.makedirs(curr_repeat_save_dir, exist_ok=True)

                #! Repeats of one launch share a W&B group, so they cluster
                #! into a single mean/range band per chart.
                args["repeat"] = repeat
                if not args.get("wandb_group"):
                    args["wandb_group"] = os.path.basename(save_dir)

                # Run PDL
                trainer = PrimalDualTrainer(data, args, curr_repeat_save_dir)
                primal_net, dual_net, primal_loss, dual_loss, train_time = trainer.train_PDL()