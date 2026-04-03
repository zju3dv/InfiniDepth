import logging


class Log:
    _logger = logging.getLogger("InfiniDepth")
    if not _logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )

    @staticmethod
    def info(msg: str) -> None:
        Log._logger.info(msg)

    @staticmethod
    def warning(msg: str) -> None:
        Log._logger.warning(msg)

    @staticmethod
    def warn(msg: str) -> None:
        Log.warning(msg)

    @staticmethod
    def error(msg: str) -> None:
        Log._logger.error(msg)
