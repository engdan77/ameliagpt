from pathlib import Path
import appdirs

docs_sources = []


class Settings:
    def __init__(self):
        self.file = Path(appdirs.user_data_dir('llmdocs')) / 'config.txt'

    def read(self) -> str:
        if self.file.exists():
            r = self.file.read_text()
            return r
        else:
            ''

    def write(self, src: str):
        self.file.write_text(src)


settings = Settings()
