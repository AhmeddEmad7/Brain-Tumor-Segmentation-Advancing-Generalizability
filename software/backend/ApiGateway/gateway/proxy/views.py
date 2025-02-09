from revproxy.views import ProxyView
import os
orthanc_url = os.environ.get('ORTHANC_URL')


class OrthancProxyView(ProxyView):
    upstream = orthanc_url


