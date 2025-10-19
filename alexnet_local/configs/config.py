# Common imports
import yaml
import argparse
# Especific imports

def show_config(config: dict):
  cfile = config['this_file']
  print("\033[1;37mUsing parameters from file: \033[1;34m{}\033[0m".format(config['this_file']))
  #for key in sorted(config.keys()):
  for key in config.keys():
    if (config[key] is not None) and (key != 'this_file'):
      print("\033[32m{}\033[1;32m\"{}\"\033[0m".format(key+': ', config[key]))
  print("\033[32m░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\033[0m\n")

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--load_config',
                      dest='config_file',
                      # type=argparse.FileType(mode='r'),
                      help='The yaml configuration file')
  args, unprocessed_args = parser.parse_known_args()

  # parser.add_argument('--data_root', default=None, required=True, help='The data folder')
  # parser.add_argument('--phase', default=None, required=True, help='train or val')

  if args.config_file:
    with open(args.config_file, 'r') as f:
      parser.set_defaults(**yaml.load(f, yaml.SafeLoader))

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
  sys.exit(0)
