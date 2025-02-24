import logging
import wandb


class WandbLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_entries = {"log/debug": [], "log/info": [], "log/warning": [], "log/error": [], "log/critical": []}

    def emit(self, record):
        log_entry = self.format(record)
        self.log_entries[f"log/{record.levelname.lower()}"].append(log_entry)

    def flush_to_wandb(self, epoch, metrics=None, commit=False):
        if metrics is not None:
            self.log_entries.update(metrics)
        wandb.log(self.log_entries, step=epoch, commit=commit)
        self.log_entries = {"log/debug": [], "log/info": [], "log/warning": [], "log/error": [], "log/critical": []}


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""

    format = "%(asctime)s - %(levelname)s -"
    message_format = "%(message)s"

    FORMATS = {
        logging.DEBUG: "\033[36m" + format + "\033[0m " + message_format,  # Cyan
        logging.INFO: "\033[32m" + format + "\033[0m " + message_format,  # Green
        logging.WARNING: "\033[33m" + format + "\033[0m " + message_format,  # Yellow
        logging.ERROR: "\033[31m" + format + "\033[0m " + message_format,  # Red
        logging.CRITICAL: "\033[31;43m" + format + "\033[0m " + message_format,  # Red with white background
    }

    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format)
        return formatter.format(record)


def setup_logging():
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    wh = WandbLogHandler()
    wh.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(wh)

    logger.propagate = False
    logger.wandb_handler = wh

    return logger


class TableLogger:
    def __init__(self, logger, title=None):
        self.logger = logger
        self.title = title
        self.entries = []

    def add_info(self, msg):
        try:
            param, value = msg.split(": ")
            self.entries.append((param.strip(), value.strip()))
        except ValueError:
            self.entries.append((msg, ''))  # Handle messages without ':'

    def flush(self):
        # Create the header and formatting strings
        header_format = "{:30} | {:}"
        divider = '-' * 32 + '+' + '-' * 32

        # Manually print the title and dividers
        if self.title:
            print("\n" + self.title.center(65))  # Center the title in the assumed console width
            print(divider)  # Horizontal line after the title

        # Log the header
        self.logger.info(header_format.format('Parameter', 'Value'))
        self.logger.info(divider)  # Horizontal line after the header

        # Log each entry
        for param, value in self.entries:
            self.logger.info(header_format.format(param, value))

        self.logger.info(divider)  # Horizontal line after the last row
        self.entries = []  # Clear entries after flushing