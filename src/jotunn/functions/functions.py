import daft
import magic


@daft.func
def detect_file_type(sample) -> str:
    if sample is None:
        return None
    if len(sample) == 0:
        return None
    mime = magic.from_buffer(sample, mime=True)
    return mime
