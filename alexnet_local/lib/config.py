#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for handler command line arguments and YAML config files in python scripts

"""
# Imports
import sys
import argparse
import yaml

def show_args(args)-> None:
  ''' Show command line arguments '''
  if type(args) != dict:
    args = vars(args)
  print("\033[1;37mCommand line parameters:\033[0m")
  for key in args.keys():
    if (args[key] is not None):
      print("\033[32m{}\033[1;32m\"{}\"\033[0m".format(key+': ', args[key]))
  #print("\033[32m░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\033[0m\n")

def show_config(config: dict)-> None:
  ''' Show parameters colected from YAML config file '''
  for key in config.keys():
    if (key == 'this_file'):
      print("\033[1;37mUsing configs from file: \033[1;34m{}\033[0m".format(config[key]))
    elif (config[key] is not None):
      print("\033[32m{}\033[1;32m\"{}\"\033[0m".format(key+': ', config[key]))
  print("\033[32m░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\033[0m\n")


def get_args()-> argparse.Namespace:
  ''' return command line arguments '''
  #parser = argparse.ArgumentParser(prog = 'ProgramName',  description = 'What the program does', epilog = 'Text at the bottom of help')
  parser = argparse.ArgumentParser()

  parser.add_argument('-c', '--config_file',
                      dest='config_file',
                      #default='sssss',
                      #type=argparse.FileType(mode='r'),
                      help='The yaml configuration file', required=True)

  #parser.add_argument('--dataset_file', default=None, required=True, help='The dataset file')

  #parser.add_argument("-o", "--Output", help = "Show Output", required=True)
  #parser.add_argument('--database', default=None, required=False, help='The database file')
  #parser.add_argument('--phase', default=None, required=True, help='train or val')

  args = parser.parse_args()
  return args


def get_confs(filename: str)-> dict:
  ''' return configs from YAML config file '''
  try:
    with open(filename, 'r') as file:
      fdict = yaml.safe_load(file)
  except FileNotFoundError as fnf:
    print(f"Config file: {filename} not found, check path.")
    sys.exit(1)
  return fdict

def parse_args():
  ''' return args from command line and configs YAML file in a single set '''
  #parser = argparse.ArgumentParser(prog = 'ProgramName',  description = 'What the program does', epilog = 'Text at the bottom of help')
  parser = argparse.ArgumentParser()

  parser.add_argument('-c', '--config_file',
                      dest='config_file',
                      # type=argparse.FileType(mode='r'),
                      help='The yaml configuration file',)

  args, unprocessed_args = parser.parse_known_args()

  #parser.add_argument('--data_root', default=None, required=True, help='The data folder')
  #parser.add_argument('--phase', default=None, required=True, help='train or val')
  #parser.add_argument('--database', default=None, required=False, help='The database file')

  if args.config_file:
    try:
      with open(args.config_file, 'r') as f:
        parser.set_defaults(**yaml.load(f, yaml.SafeLoader))
    except FileNotFoundError as fnf:
      print("Config file .yaml not found, check path.")
      sys.exit(1)
  args = parser.parse_args(unprocessed_args)
  show_config(vars(args))
  #visualize_config(args)
  return args

'''
def print_config():
  config_file="configs/base.yaml"
  with open(config_file, 'r') as cf:
    try:
      config = yaml.safe_load(cf)
    except yaml.YAMLError as exc:
      print(exc)
'''

if __name__ == "__main__":
  print("config demo:")
  dic={'param1': 'value1', 'param2': 'value2' }
  show_args(dic)
  cfg={'this_file': 'filename.yaml',  'key1': 'value1', 'key2': 'value2', 'key3': 0.2}
  show_config(cfg)
  sys.exit(0)
