import enum
import json
import subprocess


class TokenizationFormat(enum.Enum):
    """Result format of tokenization"""
    JSON = 'json'
    MECAB = 'mecab'
    CONLL = 'conllu'
    CABOCHA = 'cabocha'


def tokenize(path: str, result_format: TokenizationFormat = TokenizationFormat.JSON):
    """Tokenize"""
    cmd = f'cat {path} | ginza -m ja_ginza_electra -f {result_format.value}'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result_format == TokenizationFormat.JSON and len(result.stdout) != 0:
        result.stdout = json.loads(result.stdout)

    return {'stdout': result.stdout, 'stderr': result.stderr}


# NOTE: test subprocess
def lsl():
    result = subprocess.run(['ls -l'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {'stdout': result.stdout, 'stderr': result.stderr}


if __name__ == '__main__':
    # print(tokenize('data/sample.txt'))
    print(lsl())
