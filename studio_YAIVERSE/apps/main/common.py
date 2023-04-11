import datetime
import os
import uuid


def typename(instance):
    return type(instance).__name__


def file_upload_path(instance, filename):
    ext = filename.split('.')[-1]
    d = datetime.datetime.now()
    filepath = d.strftime("%Y/%m/%d")
    suffix = d.strftime("%Y%m%d%H%M%S")
    filename = "%s_%s.%s" % (uuid.uuid4().hex, suffix, ext)
    return os.path.join(typename(instance), filepath, filename)
