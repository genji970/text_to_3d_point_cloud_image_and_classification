import logging
import warnings
from diffusers.utils import logging as diffusers_logging

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO)

# diffusers의 로깅 수준을 INFO로 변경
diffusers_logging.set_verbosity_info()

# PyTorch에서 경고 메시지도 보이게 설정
warnings.simplefilter("default")