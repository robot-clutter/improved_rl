import clt_core as clt
import os
import yaml
import bridging_the_gap as btg

import sys
import argparse


def run(exp, seed, dataset_dir, n_episodes, resume_model, model_to_eval, compare_with):
    with open('params.yml', 'r') as stream:
        params = yaml.safe_load(stream)


    if exp == 'eval_es':
        btg.eval_heuristic(seed=seed + 1, exp_name='es', n_episodes=n_episodes, local=False)

    elif exp == 'eval_les':
        btg.eval_heuristic(seed=seed + 1, exp_name='les', n_episodes=n_episodes, local=True)

    elif exp == 'collect_dataset':
        mdp_rl = btg.PushTargetWhole(params, name='rl')
        btg.collect_offpolicy_env_transitions(dataset_dir=dataset_dir, params=params, mdp=mdp_rl, seed=seed,
                                              episodes=n_episodes)
    elif exp == 'train_rl':
        mdp_rl = btg.PushTargetWhole(params, name='rl')
        btg.preprocess_offpolicy_env_transitions_to_mdp_transitions(dataset_dir=dataset_dir, mdp=mdp_rl, seed=seed)
        trainer = btg.PushTargetOffpolicy(seed=seed, exp_name='rl', dataset_dir=dataset_dir,
                                          mdp=mdp_rl)
        trainer.init_datasets()
        trainer.train()

    elif exp == 'eval_rl':
        mdp_rl = btg.PushTargetWhole(params, name='rl')
        trainer = btg.PushTargetOffpolicy(seed=seed, exp_name='rl', mdp=mdp_rl,
                                          dataset_dir=dataset_dir, check_exist=False)
        if compare_with == 'es':
            trainer.eval_gt([model_to_eval], n_episodes=n_episodes, local=False)
        elif compare_with == 'les':
            trainer.eval_gt([model_to_eval], n_episodes=n_episodes, local=True)
        else:
            clt.error("Choose between es or les for compare_with argument.")

    elif exp == 'train_rl_es':
        mdp_rl_es = btg.PushTargetWholeTowardsEmptySpace(params, local=False, plot=False, name='rl_es')
        btg.preprocess_offpolicy_env_transitions_to_mdp_transitions(dataset_dir=dataset_dir, mdp=mdp_rl_es, seed=seed)
        trainer = btg.PushTargetOffpolicy(seed=seed, exp_name='rl_es', dataset_dir=dataset_dir,
                                          mdp=mdp_rl_es)
        trainer.init_datasets()
        trainer.train()

    elif exp == 'eval_rl_es':
        mdp_rl_es = btg.PushTargetWholeTowardsEmptySpace(params, local=False, plot=False, name='rl_es')
        trainer = btg.PushTargetOffpolicy(seed=seed, exp_name='rl_es', mdp=mdp_rl_es,
                                          dataset_dir=dataset_dir, check_exist=False)
        trainer.eval_gt([model_to_eval], n_episodes=n_episodes, local=False)

    elif exp == 'train_rl_les':
        mdp_rl_les = btg.PushTargetWholeTowardsEmptySpace(params, local=True, plot=False, name='rl_les')
        btg.preprocess_offpolicy_env_transitions_to_mdp_transitions(dataset_dir=dataset_dir, mdp=mdp_rl_les, seed=seed)
        trainer = btg.PushTargetOffpolicy(seed=seed, exp_name='rl_les', dataset_dir=dataset_dir,
                                          mdp=mdp_rl_les)
        trainer.init_datasets()
        trainer.train()

    elif exp == 'eval_rl_les':
        mdp_rl_les = btg.PushTargetWholeTowardsEmptySpace(params, local=True, plot=False, name='rl_les')
        trainer = btg.PushTargetOffpolicy(seed=seed, exp_name='rl_les', mdp=mdp_rl_les,
                                          dataset_dir=dataset_dir, check_exist=False)
        trainer.eval_gt([model_to_eval], n_episodes=n_episodes, local=True)

    else:
        clt.error("Exp " + exp + " is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp', default='eval_empty_space', type=str, help='Name of experiment to run')
    parser.add_argument('--seed', default=0, type=int, help='Seed that will run the experiment')
    parser.add_argument('--dataset_dir', default=os.path.join(os.getenv('ROBOT_CLUTTER_WS'), 'clt_logs/dataset'),
                        type=str, help='Directory of the dataset for the training experiments')
    parser.add_argument('--n_episodes', default=200, type=int,
                        help='Number of episodes to run for')
    parser.add_argument('--resume_model', default='None', type=str, help='Path for the model to resume training')
    parser.add_argument('--model_to_eval', default='epoch_01', help='The model to evaluate')
    parser.add_argument('--compare_with', default='es', help='The heuristic to compare with during evaluation of rl. Possible values: "es", "les".')
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    run(**args)
