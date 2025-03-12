from revproxy.views import ProxyView
import os
orthanc_url = os.environ.get('ORTHANC_URL')
Nifi_URL = os.environ.get('Nifi_URL')

class OrthancProxyView(ProxyView):
    upstream = orthanc_url



class NiftiProxyView(ProxyView):
    upstream = Nifi_URL