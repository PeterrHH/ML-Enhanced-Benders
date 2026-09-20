import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from paths import ensure_dir

try:
    import wandb
except Exception as exc:  # not installed, or a broken install
    wandb = None
    _WANDB_IMPORT_ERROR = exc
else:
    _WANDB_IMPORT_ERROR = None


DEFAULT_WANDB_PROJECT = "MLBenders"

#! Flags that become W&B tags, but only when they are switched on.
FEATURE_TAGS = {
    "dual_alternate_loss": "dual_alt_loss",
    "dual_classification": "dual_cls",
    "dual_completion": "dual_completion",
    "oracle_supervised_dual": "oracle_dual",
    "use_heuristic_lambda_loss": "heuristic_lambda",
    "entropy_in_loss": "entropy_loss",
    "use_difficulty_weighting": "difficulty_weighting",
    "use_topology_features": "topology_features",
    "dual_gumbel": "gumbel",
    "repair": "repair",
    "opt_targets": "opt_targets",
    "SeperatePredicationHead": "separate_head",
}


def model_tag(args):
    """Short name for the primal/dual net combination that will be built.

    Mirrors the net selection in PrimalDualTrainer.__init__, so the tag always
    matches the classes that are actually instantiated. Other training entry
    points (e.g. flowfirst) add their own tag here.
    """
    problem_type = args["problem_type"]
    if problem_type == "QP":
        return "qp-baseline"
    if problem_type != "ED":
        return problem_type.lower()

    if not args["dual_alternate_loss"]:
        return "baseline"  # PrimalNetEndToEnd + DualNet
    if args["dual_classification"]:
        return "e2e-cls"
    if args["dual_completion"]:
        return "e2e-completion"
    return "e2e-alt"


def problem_size_tag(args):
    if args["problem_type"] == "ED":
        return f"{len(args['ED_args']['N'])}node"
    if args["problem_type"] == "QP":
        return str(args["QP_args"].get("type", "qp"))
    return None


def run_name(args):
    """Short, scannable name, e.g. 'ED-3node-e2e-cls-0920_1432'.

    Repeats get an -r<i> suffix. Everything else lives in tags and
    wandb.config; the name stays editable in the W&B UI.
    """
    parts = [args["problem_type"], problem_size_tag(args), model_tag(args)]
    parts = [p for p in parts if p]
    name = "-".join(parts) + "-" + time.strftime("%m%d_%H%M")
    #! Repeats of one launch can finish inside the same minute, so the index
    #! is what keeps their names distinct.
    if args.get("repeat") is not None:
        name += f"-r{args['repeat']}"
    return name


def run_tags(args):
    tags = [args["problem_type"], problem_size_tag(args), model_tag(args)]
    tags += [tag for key, tag in FEATURE_TAGS.items() if args.get(key, False)]
    return [t for t in tags if t]


def _scalar(value):
    return value.item() if torch.is_tensor(value) else value


class Logger():
    """Fans training metrics out to TensorBoard and/or Weights & Biases.

    Both backends receive the same metric names: the "/" prefixes that group
    scalars into TensorBoard sections are also what W&B groups its panels by.
    """

    def __init__(self, args, data, X, scale_factors, train_indices, valid_indices, save_dir, opt_targets):
        self.args = args
        self.writer = SummaryWriter(log_dir=save_dir) if args.get("use_tensorboard", True) else None
        self.run = self._init_wandb(args, save_dir)
        self.watching = False

        self.opt_targets = opt_targets
        self.data = data
        self.problem_type = args["problem_type"]
        self.is_qp = self.problem_type == "QP"
        self.is_qp_simple = self.problem_type == "QP" and args["QP_args"]["type"] == "simple"
        self.is_qp_not_simple = self.problem_type == "QP" and args["QP_args"]["type"] != "simple"

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        self.X_train = X[train_indices].to(self.DTYPE).to(self.DEVICE)
        self.X_valid = X[valid_indices].to(self.DTYPE).to(self.DEVICE)
        self.scale_factors_train = scale_factors[train_indices]
        self.scale_factors_valid = scale_factors[valid_indices]

        self.train_indices = train_indices
        self.valid_indices = valid_indices

        if self.opt_targets:
            if not self.is_qp:
                self.Y_target_train = data.opt_targets["y_operational"].to(self.DTYPE).to(self.DEVICE)[self.train_indices]
                self.mu_target_train = data.opt_targets["mu_operational"].to(self.DTYPE).to(self.DEVICE)[self.train_indices]  
                self.lamb_target_train = data.opt_targets["lamb_operational"].to(self.DTYPE).to(self.DEVICE)[self.train_indices]
                self.Y_target_valid = data.opt_targets["y_operational"].to(self.DTYPE).to(self.DEVICE)[self.valid_indices]
                self.mu_target_valid = data.opt_targets["mu_operational"].to(self.DTYPE).to(self.DEVICE)[self.valid_indices]  
                self.lamb_target_valid = data.opt_targets["lamb_operational"].to(self.DTYPE).to(self.DEVICE)[self.valid_indices]
            elif self.is_qp_simple:
                self.Y_target_train = data.trainY
                self.mu_target_train = data.train_mu
                self.lamb_target_train = data.train_lamb
                self.Y_target_valid = data.validY
                self.mu_target_valid = data.valid_mu
                self.lamb_target_valid = data.valid_lamb
            elif self.is_qp_not_simple:
                self.Y_target_train = data.trainY
                self.Y_target_valid = data.validY

    #! ------------------------------------------------------------------
    #! Backend plumbing
    #! ------------------------------------------------------------------

    def _init_wandb(self, args, save_dir):
        if not args.get("use_wandb", False):
            return None
        if wandb is None:
            print(
                "[logger] use_wandb is true but the wandb package is unavailable "
                f"({_WANDB_IMPORT_ERROR}). Continuing without W&B."
            )
            return None


        return wandb.init(
            project=args.get("wandb_project") or DEFAULT_WANDB_PROJECT,
            entity=args.get("wandb_entity") or None,
            mode=args.get("wandb_mode") or "online",
            name=run_name(args),
            tags=run_tags(args),
            group=args.get("wandb_group") or None,
            #! save_dir points at the outputs/ folder holding the checkpoints
            #! and args.json for this run.
            #! W&B writes into <dir>/wandb/. Defaulting to "." reproduces the
            #! previous <cwd>/wandb/; on a cluster this follows the output root
            #! into scratch instead of filling the home quota.
            dir=ensure_dir(args.get("output_root") or "."),
            config={**args, "save_dir": os.path.abspath(save_dir)},
            reinit=True,
        )

    def _log(self, metrics, step, to_wandb=True):
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)
        if self.run is not None and to_wandb:
            self.run.log({name: _scalar(value) for name, value in metrics.items()}, step=step)

    def watch(self, primal_net, dual_net):
        """Hand the nets to W&B so it tracks gradients and parameters itself.

        TensorBoard keeps its own per-parameter grad norms in log_train; this is
        the W&B equivalent and logs on its own frequency.
        """
        if self.run is None or self.watching:
            return
        freq = self.args.get("wandb_watch_freq", 100)
        if freq <= 0:
            return
        wandb.watch(primal_net, log="all", log_freq=freq, idx=0)
        wandb.watch(dual_net, log="all", log_freq=freq, idx=1)
        self.watching = True

    def finish(self):
        """Close both backends. Safe to call more than once."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.run is not None:
            if self.watching:
                try:
                    wandb.unwatch()
                except Exception:
                    pass
                self.watching = False
            self.run.finish()
            self.run = None

    #! ------------------------------------------------------------------
    #! Metrics
    #! ------------------------------------------------------------------

    def log_primal_loss(self, loss, obj, lagrange_eq, lagrange_ineq, penalty, step):
        self._log({
            "Train_loss/primal_loss": loss,
            "Train_loss_components/obj": obj,
            "Train_loss_components/primal_lagrange_eq": lagrange_eq,
            "Train_loss_components/primal_lagrange_ineq": lagrange_ineq,
            "Train_loss_components/primal_penalty": penalty,
        }, step)

    def log_dual_loss(self, loss, step, obj, lagrange_eq, lagrange_ineq, penalty):
        self._log({
            "Train_loss/dual_loss": loss,
            "dual_loss_components/obj": obj,
            "dual_loss_components/lagrange_eq": lagrange_eq,
            "dual_loss_components/lagrange_ineq": lagrange_ineq,
            "dual_loss_components/dual_penalty": penalty,
        }, step)

    def log_rho_vk(self, rho, v_k, step):
        self._log({
            "Rho_and_violation/rho": rho,
            "Rho_and_violation/v_k": v_k,
        }, step)

    def log_train(self, data, primal_net, dual_net, rho, step):
        with torch.no_grad():
            primal_net.eval()
            dual_net.eval()
            Y = primal_net(self.X_train, self.scale_factors_train)
            mu, lamb = dual_net(self.X_train)

            obj = data.obj_fn(self.X_train, Y)

            ineq_resid = data.ineq_resid(self.X_train, Y)
            ineq_dist = data.ineq_dist(self.X_train, Y)

            eq_resid = data.eq_resid(self.X_train, Y)

            metrics = {}

            if self.opt_targets:
                obj_target = data.obj_fn(self.X_train, self.Y_target_train)
                if not self.is_qp_not_simple:
                    dual_obj = data.dual_obj_fn(self.X_train, mu, lamb)
                    dual_obj_target = data.dual_obj_fn(self.X_train, self.mu_target_train, self.lamb_target_train)
                    metrics["Train_obj/dual_obj_optimality_gap"] = ((dual_obj_target - dual_obj)/dual_obj_target).mean()
                    metrics["Train_obj/dual_obj"] = dual_obj.mean()
                    metrics["Train_obj/duality_gap"] = ((obj - dual_obj)/obj).mean()

                optimality_gap = (obj - obj_target)/obj_target
                metrics["Train_obj/obj_optimality_gap"] = optimality_gap.mean()

            # Obj funcs
            metrics["Train_obj/obj"] = obj.mean()

            #! Neural network outputs and targets
            # metrics["Train_outputs/Y"] = Y.mean()
            # metrics["Train_outputs/mu"] = mu.mean()
            # metrics["Train_outputs/lamb"] = lamb.mean()
            # if self.opt_targets:
            #     if data.args["benders_compact"]:
            #         Y_diff = (Y[:, data.num_g:] - self.Y_target_train).abs()
            #         lamb_diff = (lamb[:, data.num_g:] - self.lamb_target_train).abs()
            #     else:
            #         Y_diff = (Y - self.Y_target_train).abs()
            #         lamb_diff = (lamb - self.lamb_target_train).abs()
            #     mu_diff = (mu - self.mu_target_train).abs()
            #     metrics["Train_outputs/Y_diff"] = Y_diff.mean()
            #     metrics["Train_outputs/mu_diff"] = mu_diff.mean()
            #     metrics["Train_outputs/lamb_diff"] = lamb_diff.mean()

            #! Constraint violations
            # metrics["Train_constraints/eq_resid"] = eq_resid.mean()
            # metrics["Train_constraints/ineq_resid"] = ineq_resid.mean()
            # metrics["Train_constraints/ineq_mean"] = ineq_dist.mean()
            # metrics["Train_constraints/ineq_max"] = ineq_dist.max()
            # metrics["Train_constraints/eq_mean"] = eq_resid.abs().mean()
            # metrics["Train_constraints/eq_max"] = eq_resid.abs().max()

            if not self.is_qp:
                p_gt, f_lt, md_nt = data.split_dec_vars_from_Y(Y)
                eq_rhs_train, ineq_rhs_train = data.split_X(self.X_train)
                p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = data.split_ineq_constraints(ineq_rhs_train)
                p_gt_ub = p_gt_ub * self.scale_factors_train
                for i in range(p_gt.shape[1]):
                    metrics[f"Train_decvars/generator{i}"] = p_gt[:, i].mean()
                    #! First g of ineq_rhs are lower bounds, second g are upper bounds
                    metrics[f"Train_decvars/generator{i}_frac"] = (p_gt[:, i]/p_gt_ub[:, i]).mean()
                for i in range(f_lt.shape[1]):
                    metrics[f"Train_decvars/lineflow{i}"] = f_lt[:, i].mean()
                for i in range(md_nt.shape[1]):
                    metrics[f"Train_decvars/missed_demand{i}"] = md_nt[:, i].mean()
                for i in range(md_nt.shape[1]):
                    metrics[f"Train_decvars/missed_demand_abs{i}"] = md_nt[:, i].abs().mean()

            if not self.is_qp:
                if self.opt_targets:
                    # Primal variable specific differences

                    p_gt_target, f_lt_target, md_nt_target = data.split_dec_vars_from_Y(self.Y_target_train, log=True)
                    diff_p_gt = p_gt - p_gt_target
                    diff_f_lt = f_lt - f_lt_target
                    diff_md_nt = md_nt - md_nt_target

                    net_flow = data.net_flow(f_lt)
                    net_flow_target = data.net_flow(f_lt_target)
                    diff_net_flow = net_flow - net_flow_target

                    # diff_ui_g = (Y[:, data.ui_g_indices] - Y_target[:, data.ui_g_indices])
                    metrics["Train_var_diffs/diff_p_gt"] = diff_p_gt.mean()
                    metrics["Train_var_diffs/diff_f_lt"] = diff_f_lt.mean()
                    metrics["Train_var_diffs/diff_md_nt"] = diff_md_nt.mean()
                    metrics["Train_var_diffs/diff_net_flow"] = diff_net_flow.mean()
                    # metrics["Train_var_diffs/diff_ui_g"] = diff_ui_g.mean()

                h, b, d, e, i, j = data.split_ineq_constraints(ineq_dist)
                ui_g, c = data.split_eq_constraints(eq_resid)

                metrics["Train_constraint_specific/p_gt_ub"] = b.mean()
                metrics["Train_constraint_specific/node_balance"] = c.abs().mean()
                metrics["Train_constraint_specific/f_lt_lb"] = d.mean()
                metrics["Train_constraint_specific/f_lt_ub"] = e.mean()
                # metrics["Train_constraint_specific/f"] = f.mean()
                # metrics["Train_constraint_specific/g"] = g.mean()
                metrics["Train_constraint_specific/p_gt_lb"] = h.mean()
                metrics["Train_constraint_specific/md_nt_lb"] = i.mean()
                metrics["Train_constraint_specific/md_nt_ub"] = j.mean()

            # if self.opt_targets: 
            #     # Dual variable specific differences
            #     # inequality
            #     mu_h, mu_b, mu_d, mu_e, mu_i, mu_j = data.split_ineq_constraints(mu)
            #     mu_target_h, mu_target_b, mu_target_d, mu_target_e, mu_target_i, mu_target_j = data.split_ineq_constraints(self.mu_target_train)
            #     mu_h_diff = mu_target_h - mu_h
            #     mu_b_diff = mu_target_b - mu_b
            #     mu_d_diff = mu_target_d - mu_d
            #     mu_e_diff = mu_target_e - mu_e
            #     mu_i_diff = mu_target_i - mu_i
            #     mu_j_diff = mu_target_j - mu_j
            #     # # equality
            #     ui_g, lamb_c = data.split_eq_constraints(lamb)
            #     ui_g, lamb_target_c = data.split_eq_constraints(self.lamb_target_train, log=True)
            #     lamb_c_diff = lamb_target_c - lamb_c

            #     metrics["Train_dual_var_diffs/gen_ub"] = mu_b_diff.mean()
            #     metrics["Train_dual_var_diffs/node_balance"] = lamb_c_diff.mean()
            #     metrics["Train_dual_var_diffs/lineflow_lb"] = mu_d_diff.mean()
            #     metrics["Train_dual_var_diffs/lineflow_ub"] = mu_e_diff.mean()
            #     metrics["Train_dual_var_diffs/gen_lb"] = mu_h_diff.mean()
            #     metrics["Train_dual_var_diffs/md_lb"] = mu_i_diff.mean()
            #     metrics["Train_dual_var_diffs/md_ub"] = mu_j_diff.mean()

            # Dual constraints
            if not self.is_qp_not_simple:
                dual_eq_resid = data.dual_eq_resid(mu, lamb)
                dual_ineq_resid = data.dual_ineq_resid(mu, lamb)
                dual_ineq_dist = torch.clamp(dual_ineq_resid, 0)
                metrics["Dual_constraints/eq_resid"] = dual_eq_resid.abs().mean()
                metrics["Dual_constraints/ineq_mean"] = dual_ineq_dist.mean()

            self._log(metrics, step)

            # Log gradients
            # Iterate over all layers and log their gradients.
            #! TensorBoard only: W&B collects these through wandb.watch instead.
            grads = {}
            for name, param in primal_net.named_parameters():
                if param.grad is not None:  # Skip parameters without gradients
                    grads[f"Gradients_primal/{name}"] = param.grad.norm().item()

            for name, param in dual_net.named_parameters():
                if param.grad is not None:  # Skip parameters without gradients
                    grads[f"Gradients_dual/{name}"] = param.grad.norm().item()

            self._log(grads, step, to_wandb=False)

    def log_val(self, data, primal_net, dual_net, step):
        with torch.no_grad():
            primal_net.eval()
            dual_net.eval()
            Y = primal_net(self.X_valid, self.scale_factors_valid)
            mu, lamb = dual_net(self.X_valid)
            obj = data.obj_fn(self.X_valid, Y) # Containes penalization of negative missed demand

            metrics = {}

            if self.opt_targets:
                # Y_target = data.opt_targets["y_operational"][data.valid_indices]
                # mu_target = data.opt_targets["mu_operational"][data.valid_indices]
                # lamb_target = data.opt_targets["lamb_operational"][data.valid_indices]
                obj_target = data.obj_fn(self.X_valid, self.Y_target_valid)
                metrics["Validation/obj_optimality_gap"] = ((obj - obj_target)/obj_target).mean()
                if not self.is_qp_not_simple:
                    dual_obj = data.dual_obj_fn(self.X_valid, mu, lamb)
                    dual_obj_target = data.dual_obj_fn(self.X_valid, self.mu_target_valid, self.lamb_target_valid)

                    metrics["Validation/dual_obj_optimality_gap"] = ((dual_obj_target - dual_obj)/dual_obj_target).mean()
                    metrics["Validation/dual_obj"] = dual_obj.mean()

            ineq_dist = data.ineq_dist(self.X_valid, Y)

            eq_resid = data.eq_resid(self.X_valid, Y)

            # Obj funcs
            metrics["Validation/obj"] = obj.mean()
            # Constraint violations
            metrics["Validation/ineq_mean"] = ineq_dist.mean()
            metrics["Validation/ineq_max"] = ineq_dist.max(dim=1)[0].mean()
            metrics["Validation/eq_mean"] = eq_resid.abs().mean()
            metrics["Validation/eq_max"] = eq_resid.abs().max(dim=1)[0].mean()

            self._log(metrics, step)


#! Kept so existing imports keep working; the logger now covers W&B too.
TensorBoardLogger = Logger
