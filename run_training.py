""" Run the task-driven learning procedure """

import argparse
import neptune
import os
import ray

from functools import partial
from typing import Dict, List
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.experiment.trial import Trial

from core.training.trainer import Trainer
from tools.cfg_parser import Config
from tools.utils import makelogger


class TrialStatusReporter(CLIReporter):
    """ Class responsible for generating training reports """
    def __init__(self, cfg):
        super(CLIReporter, self).__init__()
        self.statuses = []
        self.cfg = cfg

    def should_report(self, trials, done=False):
        """ Generates the report once any new trial changes its status """
        old_statuses = self.statuses
        self.statuses = [t.status for t in trials]
        return old_statuses != self.statuses

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        progress_str = self._progress_str(trials, done, *sys_info)
        with open(os.path.join(self.cfg.ray_trials_path, "trial_logs.txt"), 'a') as f:
            f.write(progress_str + '\n')
            print(progress_str)


def training(cfg, hyperparam_cfg):
    """ Runs a single training with given configurations """
    os.chdir(cfg.project_root_path)

    # Check if the NEPTUNE_API_TOKEN token is set in environment variables
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    if api_token:
        project = os.getenv("NEPTUNE_PROJECT")
        neptune_run = neptune.init_run(name=cfg.expr_ID, project=project, source_files=['core/*'])
        logger.info("NEPTUNE_API_TOKEN found. Logging experiment to the cloud.")
    else:
        neptune_run = neptune.init_run(mode="offline")  # Runs locally without cloud logging
        logger.warning("NEPTUNE_API_TOKEN not found. Running in offline mode.")
    
    neptune_run['versioning/code_version'].track_files(os.path.join(cfg.project_root_path, 'core'))
    neptune_run["hyperparam_cfg"] = hyperparam_cfg
    neptune_run["cfg"] = cfg
    neptune_run["sys/tags"].add([cfg.expr_ID])

    cfg['core_dir'] = os.path.dirname(os.path.abspath(train.get_context().get_trial_dir()))
    cfg['trial_dir'] = train.get_context().get_trial_dir()

    # save general config file and the trial hyperparam config file
    cfg.write_cfg(os.path.join(cfg.core_dir, 'TR%02d_%s' % (cfg.try_num, "training_cfg.yml")))
    hyperparam_cfg = Config(hyperparam_cfg)
    hyperparam_cfg.write_cfg(os.path.join(train.get_context().get_trial_dir(), 'trial_hyperparam_cfg.yml'))

    # run the training
    trainer = Trainer(cfg=cfg, hyperparam_cfg=hyperparam_cfg, neptune_run=neptune_run)
    trainer.fit()
    neptune_run.stop()


def run_hyperparameter_search(cfg, cpus_per_trial=1, gpus_per_trial=1):
    """ Runs a hyperparameter search using Ray, which starts parallel trainings """
    ray.init(object_store_memory=10**9)
    assert ray.is_initialized() is True

    # define the hyperparameters that have to be explored
    hyperparam_config = {
        "hidden": (128, 256),
        "adaptive": True,
        "attention": True,
        "lr": 0.01,
        "kernel_size": 9,
        "layers": 2,
        "batch_size": 2,
    }

    # define a scheduler for the hyperparameter search
    scheduler = ASHAScheduler(
        time_attr='iter',
        metric="loss",
        mode="min",
        max_t=cfg.n_epochs,
        grace_period=cfg.n_epochs // 10 + 1,
        reduction_factor=2
    )

    # run a single training procedure
    result = tune.run(
        partial(training, cfg),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=hyperparam_config,
        num_samples=cfg.num_trials,
        scheduler=scheduler,
        progress_reporter=TrialStatusReporter(cfg),
        storage_path =cfg.ray_trials_path,
        log_to_file=("trial_stdout.log", "trial_stderr.log")
    )

    # save the results
    best_trial = result.get_best_trial("loss", "min", "all")
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    logger.info(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    ray.shutdown()
    assert ray.is_initialized() is False


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--work-dir', required=True, type=str,
                        help='The path to the working directory where the training will be saved')
    parser.add_argument('--data-path', required=True, type=str,
                        help='The path to the folder that contains preprocessed data')
    parser.add_argument('--expr-ID', default='V01-train', type=str,
                        help='Training ID')
    return parser.parse_args()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args()
    cwd = os.getcwd()
    logger = makelogger()

    # Define the experiment configuration (user config overwrites default)
    user_cfg_path = os.path.join(cwd, 'configs/training_cfg.yml')
    default_config = {
        'dataset_dir': args.data_path,
        'expr_ID': args.expr_ID,
        'work_dir': os.path.join(args.work_dir, args.expr_ID),
        'user_cfg_path': user_cfg_path,
        'project_root_path': cwd,
        'ray_trials_path': os.path.join(args.work_dir, args.expr_ID),
    }
    config = Config(default_config, user_cfg_path)
    config["cls_tasks"] = sum(config['cls_task_names'].values())
    config["reg_tasks"] = sum(config['reg_task_names'].values())

    run_hyperparameter_search(config)
