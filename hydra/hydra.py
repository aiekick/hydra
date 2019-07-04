import argparse
import inspect
import logging
import os
import sys

import pkg_resources
from omegaconf import OmegaConf

from . import utils


def get_args():
    parser = argparse.ArgumentParser(description='Hydra experimentation framework')
    version = pkg_resources.require("hydra")[0].version
    parser.add_argument('--version', action='version', version="hydra {}".format(version))
    parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                     "(use dots for.nested=overrides)")
    parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                     "(use dots for.nested=overrides)")
    parser.add_argument('--verbose', '-v',
                        help='Activate debug logging, otherwise takes a '
                             'comma separated list of loggers ("root" for root logger)',
                        nargs='?',
                        default=None)

    parser.add_argument('--cfg', '-c', action='store_true', help='Show config')
    parser.add_argument('--cfg_type', '-t',
                        help='Config type to show',
                        choices=['job', 'hydra', 'both'],
                        default='job')

    parser.add_argument('--run', '-r', action='store_true', help='Run a job')
    parser.add_argument('--sweep', '-s', action='store_true', help='Perform a sweep')

    return parser.parse_args()


class Hydra:

    def __init__(self,
                 task_name,
                 conf_dir,
                 conf_filename,
                 task_function,
                 verbose):
        utils.setup_globals()
        utils.JobRuntime().set('name', utils.get_valid_filename(task_name))
        self.conf_dir = conf_dir
        self.conf_filename = conf_filename
        self.task_function = task_function
        self.verbose = verbose

    def _create_hydra_cfg(self, overrides):
        defaults_res = utils.create_cfg(cfg_dir='pkg://hydra.default_conf', cfg_filename='hydra.yaml')
        return utils.create_hydra_cfg(cfg_dir=self.conf_dir,
                                      hydra_cfg_defaults=defaults_res['cfg'],
                                      overrides=overrides)

    def run(self, overrides):
        hydra_cfg = self._create_hydra_cfg(overrides)
        return utils.run_job(cfg_dir=self.conf_dir,
                             cfg_filename=self.conf_filename,
                             hydra_cfg=hydra_cfg,
                             task_function=self.task_function,
                             overrides=overrides,
                             verbose=self.verbose,
                             job_dir=hydra_cfg.hydra.run.dir,
                             job_subdir_key=None)

    def sweep(self, overrides):
        hydra_cfg = self._create_hydra_cfg(overrides)
        utils.configure_log(hydra_cfg.hydra.logging, self.verbose)
        if hydra_cfg.hydra.launcher is None:
            raise RuntimeError("Hydra launcher is not configured")
        launcher = utils.instantiate(hydra_cfg.hydra.launcher)
        launcher.setup(cfg_dir=self.conf_dir,
                       cfg_filename=self.conf_filename,
                       hydra_cfg=hydra_cfg,
                       task_function=self.task_function,
                       verbose=self.verbose,
                       overrides=overrides)

        return launcher.launch()

    def get_cfg(self, overrides):
        hydra_cfg = self._create_hydra_cfg(overrides)
        ret = utils.create_cfg(cfg_dir=self.conf_dir,
                               cfg_filename=self.conf_filename,
                               cli_overrides=overrides)
        ret['hydra_cfg'] = hydra_cfg
        return ret


def run_hydra(task_function, config_path):
    stack = inspect.stack()
    calling_file = stack[2][0].f_locals['__file__']

    target_file = os.path.basename(calling_file)
    task_name = os.path.splitext(target_file)[0]
    args = get_args()

    global log
    log = logging.getLogger(__name__)

    if os.path.isabs(config_path):
        raise RuntimeError("Config path should be relative")
    abs_config_path = os.path.realpath(os.path.join(os.path.dirname(calling_file), config_path))
    if not os.path.exists(abs_config_path):
        raise RuntimeError("Config path '{}' does not exist".format(abs_config_path))
    if os.path.isfile(abs_config_path):
        conf_dir = os.path.dirname(abs_config_path)
        conf_filename = os.path.basename(abs_config_path)
    else:
        conf_dir = abs_config_path
        conf_filename = None

    hydra = Hydra(task_name=task_name,
                  conf_dir=conf_dir,
                  conf_filename=conf_filename,
                  task_function=task_function,
                  verbose=args.verbose)

    if args.run + args.cfg + args.sweep > 1:
        raise ValueError("Only one of --run, --sweep and --cfg can be specified")
    if args.run + args.cfg + args.sweep == 0:
        args.run = True

    if args.run:
        command = "run"
    elif args.sweep:
        command = "sweep"
    elif args.cfg:
        command = "cfg"

    if command == 'run':
        hydra.run(overrides=args.overrides)
    elif command == 'sweep':
        hydra.sweep(overrides=args.overrides)
    elif command == 'cfg':
        task_cfg = hydra.get_cfg(overrides=args.overrides)
        job_cfg = task_cfg['cfg']
        if args.cfg_type == 'job':
            cfg = job_cfg
        elif args.cfg_type == 'hydra':
            cfg = task_cfg['hydra_cfg']
        elif args.cfg_type == 'both':
            cfg = OmegaConf.merge(task_cfg['hydra_cfg'], job_cfg)
        else:
            assert False
        # if args.debug:
        #     for file, loaded in task_cfg['checked']:
        #         if loaded:
        #             print("Loaded: {}".format(file))
        #         else:
        #             print("Not found: {}".format(file))

        print(cfg.pretty())
    else:
        print("Command not specified")


def main(config_path="."):
    def main_decorator(task_function):
        def decorated_main():
            try:
                run_hydra(task_function, config_path)
            except KeyboardInterrupt:
                sys.exit(-1)
            except SystemExit:
                pass

        return decorated_main

    return main_decorator
