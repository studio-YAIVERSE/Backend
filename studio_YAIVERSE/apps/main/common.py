import datetime
import os
import uuid


def typename(instance):
    return type(instance).__name__


def file_upload_path(instance, filename):
    d = datetime.datetime.now()
    file_base_path = d.strftime("%Y-%m-%d")
    file_entry_path = "%s_%s" % (uuid.uuid4().hex, d.strftime("%Y%m%d%H%M%S"))
    return os.path.join(typename(instance), file_base_path, file_entry_path, filename)
