import logging
from datetime import datetime
from typing import Optional

from rich.console import ConsoleRenderable
from rich.logging import RichHandler
from rich.traceback import Traceback


class NNRichHandler(RichHandler):
    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Optional[Traceback],
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        # Hack to display the logger name instead of the filename in the rich logs
        path = record.name  # str(Path(record.pathname))
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)

        log_renderable = self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=path,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
        )
        return log_renderable
