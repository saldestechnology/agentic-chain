from agentic.runnable.runnable import Runnable
from agentic.utils.config import LogLevel, Settings
import sys
import json

settings = Settings()


def log(*message: object, log_level: LogLevel = "INFO") -> None:
    if not settings.debug:
        return

    caller_frame = sys._getframe(1)
    caller_name = caller_frame.f_code.co_name
    caller_locals = caller_frame.f_locals

    if "self" in caller_locals:
        instance = caller_locals["self"]
        if hasattr(instance, "name") and instance.name:
            caller_name = instance.name
        elif isinstance(instance, Runnable):
            caller_name = getattr(instance, "name", type(instance).__name__)

    if settings.env == "production":
        print(
            json.dumps(
                {
                    "log_level": log_level,
                    "called_name": caller_name,
                    "message": message,
                },
                default=str,
            )
        )
    if settings.env != "production" and settings.log_level == log_level:
        print(f"({log_level}) [{caller_name}] {str(message)}")
