
class Shared:
    def __init__(self):
        self.llm = None
        self.loaded_docs = []
        self.tunnel = False


shared_obj = Shared()
