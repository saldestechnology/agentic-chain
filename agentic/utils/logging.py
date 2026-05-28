from agentic.utils.config import Settings
import sys
import json

settings = Settings()

def log(message: str, log_level: str = "INFO", as_json: bool = False) -> None:
    if not settings.debug:
        return
    # Check if instance of Runnable or a function and print name or function name
    caller_frame = sys._getframe(1)
    caller_name = caller_frame.f_code.co_name
    caller_locals = caller_frame.f_locals
    
    if "self" in caller_locals:
        instance = caller_locals["self"]
        if hasattr(instance, "name") and instance.name:
            caller_name = instance.name
        elif type(instance).__name__ == "Runnable" or hasattr(instance, "invoke"):
            caller_name = getattr(instance, "name", type(instance).__name__)
    if settings.env == 'production':
        print(json.dumps({ "log_level": log_level, "called_name": caller_name, "message": message }))
    if settings.env != 'production' and settings.log_level == log_level:
        print(f"({log_level}) [{caller_name}] {message}")