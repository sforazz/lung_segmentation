import requests
import tarfile
import os


def get_weights(url, location):
    r = requests.get(url)
    with open(os.path.join(location, 'weights.tar.gz'), 'wb') as f:
        f.write(r.content)
    print(r.status_code)
    print(r.headers['content-type'])
    print(r.encoding)
    
    return os.path.join(location, 'weights.tar.gz')

 
def untar(fname):
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()
        print("Extracted in Current Directory")
    else:
        print("Not a tar.gz file: {}".format(fname))