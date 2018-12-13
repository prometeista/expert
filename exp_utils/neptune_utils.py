import argparse
import os
from deepsense import neptune


def is_neptune_online():
  # I wouldn't be suprised if this would depend on neptune version
  return 'NEPTUNE_ONLINE_CONTEXT' in os.environ


def patch_experiment(experiment, limit_per_minute):
  stock_create_channel = experiment.create_channel

  import time
  from threading import Lock

  class RequestLimiter(object):
    SECONDS_PER_MINUTE = 60

    def __init__(self):
      self._request_times = []
      self._lock = Lock()

    def is_request_allowed(self):
      with self._lock:
        now = time.time()
        self._request_times = [
          t for t in self._request_times if t > now - self.SECONDS_PER_MINUTE
        ]

        if len(self._request_times) < limit_per_minute:
          self._request_times.append(now)
          return True
        else:
          return False

  class Throttler(object):
    def __init__(self):
      self._limiter = RequestLimiter()

    class Sender(object):
      def __init__(self, limiter, stock_send):
        self._limiter = limiter
        self._stock_send = stock_send

      def __call__(self, *args, **kwargs):
        if self._limiter.is_request_allowed():
          return self._stock_send(*args, **kwargs)
        else:
          print('Channel value dropped: {} {}'.format(args, kwargs))

    def replace_send(self, channel):
      if not isinstance(channel.send, self.Sender):
        stock_send = channel.send
        setattr(channel, 'send', self.Sender(self._limiter, stock_send))
      return channel

    def create_channel(self, *args, **kwargs):
      channel = stock_create_channel(*args, **kwargs)
      return self.replace_send(channel)

  throttler = Throttler()

  setattr(experiment, 'create_channel', throttler.create_channel)

  for _, channel in experiment._channels.items():
    throttler.replace_send(channel)


def get_configuration():
  if is_neptune_online():
    # running under neptune
    ctx = neptune.Context()
    ctx.integrate_with_tensorflow()
    patch_experiment(ctx._experiment, limit_per_minute=5)
    # I can't find storage path in Neptune 2 context
    # exp_dir_path = ctx.storage_url - this was used in neptune 1.6
    exp_dir_path = os.getcwd()
  else:
    # local run
    parser = argparse.ArgumentParser(description='Debug run.')
    parser.add_argument('--exp_spec', type=str)
    parser.add_argument("--exp_dir_path", default='/tmp')
    commandline_args = parser.parse_args()
    if commandline_args.exp_spec != None:
      vars = {}
      exec(open(commandline_args.exp_spec).read(), vars)
      spec_func = vars['spec']
      # take first experiment (params configuration)
      experiment = spec_func()[0]
      params = experiment.parameters
    else:
      params = {}
    # create offline context
    ctx = neptune.Context(offline_parameters=params)
    exp_dir_path = commandline_args.exp_dir_path
  return ctx, exp_dir_path
