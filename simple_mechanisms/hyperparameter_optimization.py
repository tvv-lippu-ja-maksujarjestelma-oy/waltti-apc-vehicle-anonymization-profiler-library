import logging, logging.config
import pickle
import sys

from functools import partial

import numpy as np
import jax.numpy as jnp

import optuna, jax, argparse
from optuna.storages import RDBStorage

import sqlalchemy
from sqlalchemy.engine import URL

from infer import adp_penalty, pure_dp_penalty, l2_penalty, distance_penalty, learn_with_sgd

## Set up the data
# following the waltti git
n_seats = 78
n_cats = 6
cat_edges = [5, 40, 50, 65, 72, 78]
categories = np.empty((n_seats+1, n_cats))
j = 0
for i in range(n_seats+1):
    if i > cat_edges[j]:
        j += 1
    categories[i] = np.eye(n_cats)[j]

# compute distances between the categories across seats
distances = np.abs(np.arange(0,n_cats).reshape(-1, 1) - np.arange(0,n_cats).reshape(1, -1))
distance_matrix = categories @ distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--num_iters", type=int, default=int(1e5))
    parser.add_argument("--num_trials", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--study_name", type=str, default="hyperparam-opt-test1", help='Used for identifying Optuna test runs')
    parser.add_argument("--save_results", default=False, action='store_true', help='Save Optuna results to database')
    parser.add_argument("--debug", default=False, action='store_true', help='Use debugging prints')
    args = parser.parse_args()

    logging.basicConfig(filename=f'{args.study_name}.log', encoding='utf-8', level=logging.WARNING)
    logger = logging.getLogger("joint_logger")

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Using debugging output')
        silent = False
    else:
        logger.setLevel(logging.INFO)
        silent = True

    epsilon = args.epsilon
    delta_target = args.delta

    assert epsilon > 0 and 0 <= delta_target <= 1, f'Invalid privacy params: eps={epsilon}, delta={delta_target}!'

    if delta_target == 0:
        logger.info(f'Running hyperparam optimisation using pure DP with eps={epsilon}')
    else:
        logger.info(f'Running hyperparam optimisation using ADP with eps={epsilon}, delta={delta_target}')

    # create a prototype penalties to initialize the task
    distance_penalty_fn = partial(distance_penalty, distances=distance_matrix, n_seats=n_seats)
    if delta_target == 0:
        dp_penalty_fn = partial(pure_dp_penalty, eps=epsilon, n_seats=n_seats)
    else:
        dp_penalty_fn = partial(adp_penalty, eps=epsilon, delta_target=delta_target, n_seats=n_seats)

    def create_penalty_fn(l2_weight, dist_weight, dp_weight):
        def penalty_fn(qs):
            return l2_weight * l2_penalty(qs) + dist_weight * distance_penalty_fn(qs) + dp_weight * dp_penalty_fn(qs)
        return penalty_fn


    # create a prototype penalties to initialize the task
    total_penalty = create_penalty_fn(l2_weight=0.001, dist_weight=0.001, dp_weight=10.)
    task = learn_with_sgd(n_seats, n_cats, categories, total_penalty)
    n_iters = args.num_iters

    def evaluate_results(logits, epsilon_target):
        log_probs = jax.nn.log_softmax(logits.reshape(n_seats+1,-1), axis=1)
        # log-prob of reporting the correct category
        log_p_correct_category = log_probs[np.where(categories)]
        probs = jnp.exp(log_probs)

        # total DP epsilon
        epsilon_total = jnp.max(jax.nn.relu( jnp.abs(log_probs[1:] - log_probs[:-1]) - epsilon_target) )

        # calculate total delta for ADP
        delta_add = jnp.max(jnp.sum(jnp.clip(probs[:-1,:] - jnp.exp(epsilon_target)*probs[1:,:], a_min=0, a_max=None),1))
        delta_remove = jnp.max(jnp.sum(jnp.clip(probs[1:,:] - jnp.exp(epsilon_target)*probs[:-1,:], a_min=0, a_max=None),1))
        delta_total = jnp.max( jnp.array((delta_add,delta_remove)))

        return log_probs, log_p_correct_category, epsilon_total, delta_total

    def objective(trial):
        dp_weight = trial.suggest_float(name='dp weight', low=1e-3, high=1e1)
        l2_weight = trial.suggest_float(name='l2 weight', low=1e-4, high=1e0)
        dist_weight = trial.suggest_float(name='dist. weight', low=1e-5, high=1e-3)
        # replace the penalties in the task
        new_total_penalty = create_penalty_fn(l2_weight=l2_weight, dist_weight=dist_weight, dp_weight=dp_weight)
        task.penalty_fn = new_total_penalty
        # learn the parameters using SGD
        learned_logits = task.train(n_iters, init_seed=args.seed, silent=silent)
        #
        log_probs, logp_correct, epsilon_totals, delta_totals = evaluate_results(learned_logits, epsilon)
        #
        logp_loss = np.linalg.norm(logp_correct)

        dp_params_loss = np.linalg.norm(epsilon_totals-epsilon) + np.linalg.norm(delta_totals - delta_target)

        furthest_cat_logp = log_probs[np.arange(n_seats+1), np.argmax(categories @ distances, axis=1)]
        dist_loss = np.sum(np.exp(furthest_cat_logp) > 1e-3)
        return logp_loss, dp_params_loss, dist_loss, np.linalg.norm(np.exp(furthest_cat_logp))

    
    if args.save_results:
        # save results using postgresql database
        # SET THESE to point to some actual database
        db_url = URL.create(
                "postgresql",
                #username="mynonsuperuser",
                #password="mynonsuperuser",  # plain (unescaped) text
                host="localhost",
                database="myinner_db",)

        storage = RDBStorage(url=str(db_url))
    else:
        # or run test version without saving results
        db_url = None
        storage = None

    study = optuna.create_study(study_name=args.study_name, directions=["minimize", "minimize", "minimize", "minimize"],
                                storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=args.num_trials)



if __name__ == "__main__":
    main()
