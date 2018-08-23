from chainer.training import extensions
from io import StringIO
import sys


class PrintStackReport(extensions.PrintReport):

    """Trainer extension to print the accumulated results to slack.

    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.

    Args:
        entries (list of str): List of keys of observations to print.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the trainer, or a LogReport instance to use
            internally.
        out: Stream to print the bar. Standard output is used by default.
    """

    def __init__(self, entries, log_report='LogReport', out=sys.stdout):
        self._out_final = out
        super(PrintStackReport, self).__init__(entries, log_report, StringIO())

    def __call__(self, trainer):
        super(PrintStackReport, self).__call__(trainer)
        text = self._out.getvalue()
        self._out = StringIO()
        self._out_final.write(text)
